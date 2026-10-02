"""CPU validation. This suite does not certify CUDA compilation or inference."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pradium_build', HERE / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class PatchSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='pradium-validation-')
        cls.source = Path(cls.temp.name) / 'source'
        cls.manifest = build.materialize(cls.source)
        policy_path = cls.source / 'exllamav3/modules/attention_fn/pradium_policy.py'
        spec = importlib.util.spec_from_file_location('pradium_policy', policy_path)
        cls.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.policy)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_patch_python_syntax(self):
        for relative in ['exllamav3/modules/attention_fn/bc_attn.py',
                         'exllamav3/modules/attention_fn/pradium_policy.py']:
            path = self.source / relative
            compile(path.read_text(), str(path), 'exec')
        gpu_test = HERE / 'tests/gpu_direct_attention.py'
        compile(gpu_test.read_text(), str(gpu_test), 'exec')
        self.assertEqual(self.manifest['upstream_commit'], build.PIN)
        self.assertFalse(self.manifest['gpu_validated'])

    def eligible(self, **overrides):
        args = dict(enabled_limit=513, bsz=1, q_len=1, table_pages=2,
                    page_size=256, head_dim=128, v_head_dim=128, qsa=False)
        args.update(overrides)
        return self.policy.direct_attention_eligible(**args)

    def test_direct_boundary_and_context_growth(self):
        self.assertTrue(self.eligible())
        self.assertFalse(self.eligible(enabled_limit=512))
        self.assertFalse(self.eligible(table_pages=3))
        self.assertFalse(self.eligible(enabled_limit=0))
        self.assertFalse(self.eligible(enabled_limit=-1))

    def test_unsupported_direct_shapes(self):
        for override in [dict(bsz=2), dict(q_len=2), dict(v_head_dim=64),
                         dict(qsa=True), dict(table_pages=0)]:
            with self.subTest(override=override):
                self.assertFalse(self.eligible(**override))

    def test_existing_destination_is_preserved(self):
        marker = self.source / 'KEEP'
        marker.write_text('user data')
        with self.assertRaises(ValueError):
            build.materialize(self.source)
        self.assertEqual(marker.read_text(), 'user data')

    def test_replay_bindings_native(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'A C++ compiler is required for the replay-plan test')
        binary = Path(self.temp.name) / 'replay_plan'
        include = self.source / 'exllamav3/exllamav3_ext'
        subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-I'+str(include),
                        str(HERE / 'tests/replay_plan.cpp'), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
