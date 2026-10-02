"""GPU differential tests for the direct kernel used by patch 0002.

Run explicitly on a CUDA host. Missing GPU/dependencies are failures, not skips.
This tests the kernel; full BC_Attention/model integration needs the campaign gate.
"""
import unittest
import torch
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode


class DirectAttentionGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; this gate cannot pass on CPU')

    def test_direct_matches_split_and_reference(self):
        for dim in (64, 80, 128):
            for past in (31, 127, 255, 256, 511):
                for window in (None, (63, 0)):
                    for has_sinks in (False, True):
                        with self.subTest(dim=dim, past=past, window=window, sinks=has_sinks):
                            torch.manual_seed(dim + past)
                            total = past + 1
                            pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
                            kv_heads, q_heads = 2, 8
                            device = 'cuda:0'
                            k = torch.randn(pages, PAGE_SIZE, kv_heads, dim,
                                            device=device, dtype=torch.float16)
                            v = torch.randn_like(k)
                            # Non-identity physical page order exercises paged addressing.
                            table = torch.arange(pages - 1, -1, -1, device=device,
                                                 dtype=torch.int32).unsqueeze(0)
                            lengths = torch.tensor([past], device=device, dtype=torch.int32)
                            q = torch.randn(1, 1, q_heads, dim, device=device, dtype=torch.float16)
                            sinks = torch.randn(q_heads, device=device, dtype=torch.float32) if has_sinks else None
                            kwargs = dict(causal=True, window_size=window, sinks=sinks,
                                          pre_appended_len=1, block_n=max(16, 8192 // (1 << (dim-1).bit_length())))
                            direct = paged_attn_triton_decode(q, None, None, k, v, table, lengths,
                                                              num_splits=1, **kwargs)
                            split = paged_attn_triton_decode(q, None, None, k, v, table, lengths,
                                                             num_splits=4, **kwargs)
                            torch.testing.assert_close(direct, split, atol=8e-3, rtol=8e-3)
                            flat_k = k[table.long().flatten()].reshape(-1, kv_heads, dim)[:total]
                            flat_v = v[table.long().flatten()].reshape(-1, kv_heads, dim)[:total]
                            flat_k = flat_k.repeat_interleave(q_heads // kv_heads, dim=1).float()
                            flat_v = flat_v.repeat_interleave(q_heads // kv_heads, dim=1).float()
                            scores = torch.einsum('hd,thd->ht', q[0, 0].float(), flat_k) / dim**0.5
                            if window:
                                scores[:, :max(0, past-window[0])] = -torch.inf
                            if sinks is not None:
                                augmented = torch.cat((scores, sinks[:, None]), dim=1)
                                weights = augmented.softmax(dim=-1)[:, :-1]
                            else:
                                weights = scores.softmax(dim=-1)
                            reference = torch.einsum('ht,thd->hd', weights, flat_v)[None, None]
                            torch.testing.assert_close(direct.float(), reference, atol=8e-3, rtol=8e-3)


    def test_quantized_cache_direct_matches_split(self):
        from exllamav3.ext import exllamav3_ext as ext
        for bits in (4, 8):
            for dim in (96, 128):
                for past in (255, 511):
                    with self.subTest(bits=bits, dim=dim, past=past):
                        torch.manual_seed(bits + dim + past)
                        pages = (past + 1 + PAGE_SIZE - 1) // PAGE_SIZE
                        kv_heads, q_heads = 2, 8
                        rows = pages * PAGE_SIZE
                        packed, scales, decoded = [], [], []
                        for _ in range(2):
                            values = torch.randn(rows, kv_heads * dim, device='cuda:0', dtype=torch.float16)
                            pq = torch.empty(rows, kv_heads * dim // 32 * bits,
                                             device='cuda:0', dtype=torch.int32)
                            sc = torch.empty(rows, kv_heads * dim // 32,
                                             device='cuda:0', dtype=torch.float16)
                            deq = torch.empty_like(values)
                            ext.quant_cache_cont(values, pq, sc, 0.0)
                            ext.dequant_cache_cont(pq, sc, deq, 0.0)
                            packed.append(pq.view(pages, PAGE_SIZE, -1))
                            scales.append(sc.view(pages, PAGE_SIZE, -1))
                            decoded.append(deq.view(pages, PAGE_SIZE, kv_heads, dim))
                        table = torch.arange(pages - 1, -1, -1, device='cuda:0',
                                             dtype=torch.int32).unsqueeze(0)
                        lengths = torch.tensor([past], device='cuda:0', dtype=torch.int32)
                        q = torch.randn(1, 1, q_heads, dim, device='cuda:0', dtype=torch.float16)
                        kwargs = dict(causal=True, pre_appended_len=1,
                                      qc=(scales[0], scales[1], bits, bits),
                                      n_kv_heads_override=kv_heads)
                        direct = paged_attn_triton_decode(q, None, None, *packed, table,
                                                         lengths, num_splits=1, **kwargs)
                        split = paged_attn_triton_decode(q, None, None, *packed, table,
                                                        lengths, num_splits=4, **kwargs)
                        torch.testing.assert_close(direct, split, atol=1.2e-2, rtol=1.2e-2)
                        reference = paged_attn_triton_decode(q, None, None, *decoded, table,
                                                            lengths, causal=True,
                                                            pre_appended_len=1, num_splits=4)
                        torch.testing.assert_close(direct, reference, atol=1.2e-2, rtol=1.2e-2)


if __name__ == '__main__':
    unittest.main()
