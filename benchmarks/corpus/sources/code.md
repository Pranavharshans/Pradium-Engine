# Code Corpus

Realistic code fragments in several languages: Python, C/C++, JSON, shell, and
configuration. Coding and agent workloads process exactly this kind of mixed
source material.

---

## Python: a small token budget scheduler

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import PriorityQueue
from threading import Lock


@dataclass(order=True)
class Job:
    priority: int
    submitted_at: float = field(compare=False)
    prompt_tokens: int = field(compare=False)
    max_new_tokens: int = field(compare=False)
    job_id: str = field(compare=False)


class TokenBudgetScheduler:
    """Admits jobs while the projected token budget is not exceeded."""

    def __init__(self, budget_tokens: int, rate_tokens_per_s: float) -> None:
        self.budget_tokens = budget_tokens
        self.rate = rate_tokens_per_s
        self._queue: PriorityQueue[Job] = PriorityQueue()
        self._lock = Lock()
        self._inflight_tokens = 0

    def submit(self, job: Job) -> bool:
        cost = job.prompt_tokens + job.max_new_tokens
        with self._lock:
            if self._inflight_tokens + cost > self.budget_tokens:
                self._queue.put(job)
                return False
            self._inflight_tokens += cost
            return True

    def drain(self, now: float) -> list[Job]:
        admitted: list[Job] = []
        with self._lock:
            while not self._queue.empty():
                job = self._queue.queue[0]
                cost = job.prompt_tokens + job.max_new_tokens
                if self._inflight_tokens + cost > self.budget_tokens:
                    break
                self._queue.get()
                self._inflight_tokens += cost
                admitted.append(job)
        return admitted

    def complete(self, job: Job) -> None:
        with self._lock:
            self._inflight_tokens -= job.prompt_tokens + job.max_new_tokens

    def projected_finish(self, job: Job) -> float:
        queued_tokens = self._inflight_tokens
        return job.submitted_at + (queued_tokens + job.prompt_tokens) / self.rate
```

## Python: deterministic hash utilities

```python
import hashlib
import json
from typing import Any, Iterable


def stable_digest(payload: Any) -> str:
    """SHA-256 over a canonical JSON encoding; stable across processes."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def digest_token_ids(token_ids: Iterable[int]) -> str:
    return stable_digest(list(token_ids))


def short_id(digest: str, length: int = 8) -> str:
    return digest[:length]


class RollingHash:
    """Rabin-style rolling hash over integer tokens for prefix matching."""

    def __init__(self, base: int = 257, modulus: int = (1 << 61) - 1) -> None:
        self.base = base
        self.modulus = modulus
        self.value = 0
        self.length = 0

    def push(self, token: int) -> None:
        self.value = (self.value * self.base + (token + 1)) % self.modulus
        self.length += 1

    def digest(self) -> tuple[int, int]:
        return self.length, self.value
```

## C++: a tiny ring buffer

```cpp
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>

template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    static_assert(Capacity > 0, "capacity must be positive");

    void push(const T& value) {
        data_[head_] = value;
        head_ = (head_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        } else {
            tail_ = (tail_ + 1) % Capacity;
        }
    }

    std::optional<T> pop() {
        if (size_ == 0) {
            return std::nullopt;
        }
        T value = data_[tail_];
        tail_ = (tail_ + 1) % Capacity;
        --size_;
        return value;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    std::array<T, Capacity> data_{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
};
```

## C: bit manipulation helpers

```c
#include <stdint.h>
#include <stddef.h>

/* Count leading zeros for a 32-bit word; undefined for x == 0. */
static inline int clz32(uint32_t x) {
    int n = 0;
    if (x == 0) return 32;
    if ((x >> 16) == 0) { n += 16; x <<= 16; }
    if ((x >> 24) == 0) { n +=  8; x <<=  8; }
    if ((x >> 28) == 0) { n +=  4; x <<=  4; }
    if ((x >> 30) == 0) { n +=  2; x <<=  2; }
    if ((x >> 31) == 0) { n +=  1; }
    return n;
}

size_t popcount(const uint8_t* bytes, size_t len) {
    size_t total = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t b = bytes[i];
        while (b) {
            total += b & 1u;
            b >>= 1;
        }
    }
    return total;
}
```

## JSON: service configuration

```json
{
  "service": {
    "name": "token-gateway",
    "version": "2.4.1",
    "listen": {"host": "0.0.0.0", "port": 8080},
    "limits": {
      "max_concurrent_requests": 64,
      "max_prompt_tokens": 32768,
      "max_new_tokens": 4096,
      "request_timeout_ms": 120000
    },
    "queue": {
      "policy": "priority",
      "max_depth": 512,
      "starvation_guard": true
    },
    "telemetry": {
      "metrics_interval_ms": 100,
      "traces_sample_rate": 0.05
    }
  }
}
```

## JSON: a benchmark result fragment

```json
{
  "profile": "MM",
  "prompt_tokens": 1024,
  "requested_output_tokens": 256,
  "actual_output_tokens": 256,
  "ttft_ms": 84.2,
  "decode_tok_s": 61.7,
  "tpot_ms": 16.2,
  "execution_status": "SUCCESS",
  "correctness_status": "PASS",
  "metric_provenance": {
    "prefill_ms": "REPORTED_RUNTIME",
    "queue_ms": "UNAVAILABLE"
  }
}
```

## Shell: deployment helper

```bash
#!/usr/bin/env bash
set -euo pipefail

SERVICE="${1:?usage: deploy.sh <service> <version>}"
VERSION="${2:?usage: deploy.sh <service> <version>}"
REGION="${REGION:-us-east-1}"
ARTIFACT="dist/${SERVICE}-${VERSION}.tar.zst"

if [[ ! -f "${ARTIFACT}" ]]; then
  echo "building artifact for ${SERVICE} ${VERSION}" >&2
  make -s "dist/${SERVICE}-${VERSION}.tar.zst"
fi

sha256sum "${ARTIFACT}" | tee "${ARTIFACT}.sha256"

for attempt in 1 2 3; do
  if aws s3 cp "${ARTIFACT}" "s3://build-artifacts/${REGION}/${SERVICE}/"; then
    break
  fi
  echo "upload failed (attempt ${attempt}), retrying in $((attempt * 5))s" >&2
  sleep $((attempt * 5))
done

curl -fsS -X POST "https://deploy.internal/v1/releases" \
  -H "content-type: application/json" \
  -d "{\"service\": \"${SERVICE}\", \"version\": \"${VERSION}\", \"region\": \"${REGION}\"}"
```

## YAML: service configuration fragment

```yaml
server:
  workers: 8
  threads_per_worker: 4
  max_request_body: 4mb
  keepalive_timeout: 75s

logging:
  level: info
  format: json
  sampling:
    enabled: true
    initial: 100
    thereafter: 100

healthcheck:
  path: /healthz
  interval: 10s
  timeout: 2s
  unhealthy_threshold: 3
```

## Python: async request fan-out

```python
import asyncio
from dataclasses import dataclass


@dataclass
class Endpoint:
    name: str
    url: str
    timeout_s: float = 5.0


async def probe(session, endpoint: Endpoint) -> tuple[str, bool, float]:
    start = asyncio.get_event_loop().time()
    try:
        async with session.get(endpoint.url, timeout=endpoint.timeout_s) as resp:
            await resp.read()
            ok = resp.status < 500
    except Exception:
        ok = False
    elapsed = asyncio.get_event_loop().time() - start
    return endpoint.name, ok, elapsed


async def probe_all(endpoints: list[Endpoint]) -> dict[str, tuple[bool, float]]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(probe(session, ep) for ep in endpoints))
    return {name: (ok, elapsed) for name, ok, elapsed in results}
```
