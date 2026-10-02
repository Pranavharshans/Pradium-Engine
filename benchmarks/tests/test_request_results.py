"""Request / result serialization and JSON-schema tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmarks.core.capabilities import (
    CAPABILITY_NAMES,
    CapabilityStatus,
    RuntimeCapabilities,
    UnknownCapabilityError,
)
from benchmarks.core.request import (
    BenchmarkRequest,
    RequestValidationError,
    hash_token_ids,
    new_request_id,
)
from benchmarks.core.results import (
    CORRECT_PASS,
    EXEC_OOM,
    EXEC_SUCCESS,
    FailureRecord,
    RawResult,
)
from benchmarks.core.schema import SchemaError, load_schema, validate_schema

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


def make_request(**kwargs) -> BenchmarkRequest:
    defaults = dict(
        request_id=new_request_id(),
        input_ids=list(range(128)),
        prompt_token_count=128,
        max_new_tokens=64,
        profile="SS",
        input_class="S",
        output_class="S",
    )
    defaults.update(kwargs)
    return BenchmarkRequest(**defaults)


class TestRequest(unittest.TestCase):
    def test_round_trip(self) -> None:
        req = make_request()
        data = req.to_dict(include_input_ids=True)
        again = BenchmarkRequest.from_dict(data)
        self.assertEqual(again.input_ids, req.input_ids)
        self.assertEqual(again.input_ids_hash, req.input_ids_hash)
        self.assertEqual(again.request_id, req.request_id)

    def test_hash_stable(self) -> None:
        self.assertEqual(hash_token_ids([1, 2, 3]), hash_token_ids([1, 2, 3]))
        self.assertNotEqual(hash_token_ids([1, 2, 3]), hash_token_ids([1, 2, 4]))

    def test_validation_errors(self) -> None:
        with self.assertRaises(RequestValidationError):
            make_request(prompt_token_count=64)
        with self.assertRaises(RequestValidationError):
            make_request(max_new_tokens=0)
        with self.assertRaises(RequestValidationError):
            make_request(prefix_token_count=10)
        with self.assertRaises(RequestValidationError):
            make_request(temperature=0.0, greedy=False)

    def test_optional_fields(self) -> None:
        req = make_request(
            prefix_id="p1",
            prefix_token_count=64,
            fixed_decode_length=True,
            ignore_eos=True,
            concurrency_group="C2",
            metadata={"note": "x"},
        )
        data = req.to_dict()
        self.assertEqual(data["prefix_id"], "p1")
        self.assertEqual(data["prefix_token_count"], 64)
        self.assertNotIn("input_ids", data)
        self.assertIn("input_ids_hash", data)


class TestCapabilities(unittest.TestCase):
    def test_all_names_declared(self) -> None:
        caps = RuntimeCapabilities()
        for name in CAPABILITY_NAMES:
            self.assertEqual(caps.status(name), CapabilityStatus.UNKNOWN)
        data = caps.to_dict()
        self.assertEqual(set(data) - {"notes"}, set(CAPABILITY_NAMES))

    def test_round_trip(self) -> None:
        caps = RuntimeCapabilities(supports_streaming=CapabilityStatus.SUPPORTED)
        again = RuntimeCapabilities.from_dict(caps.to_dict())
        self.assertEqual(again.supports_streaming, CapabilityStatus.SUPPORTED)

    def test_unknown_capability_rejected(self) -> None:
        with self.assertRaises(UnknownCapabilityError):
            RuntimeCapabilities.from_dict({"supports_telepathy": "SUPPORTED"})
        with self.assertRaises(UnknownCapabilityError):
            RuntimeCapabilities(notes={"nope": "x"})


class TestRawResultSchema(unittest.TestCase):
    def make_result(self, **kwargs) -> RawResult:
        defaults = dict(
            benchmark_version="PRADIUM-RUNTIME-BENCH-v1",
            session_id="20261002-000000_mock_validation_abc123",
            timestamp="2026-10-02T00:00:00+00:00",
            suite="matrix",
            runtime="mock",
            profile="SS",
            input_class="S",
            output_class="S",
            run_index=0,
            warmup=False,
            benchmark_mode="validation",
            performance_valid=False,
            prompt_tokens=128,
            requested_output_tokens=64,
            actual_output_tokens=64,
            concurrency=1,
            execution_status=EXEC_SUCCESS,
            correctness_status=CORRECT_PASS,
        )
        defaults.update(kwargs)
        return RawResult(**defaults)

    def test_serialization_round_trip(self) -> None:
        record = self.make_result(ttft_ms=12.5, generated_token_ids=[1, 2, 3])
        payload = json.loads(record.to_json())
        again = RawResult.from_dict(payload)
        self.assertEqual(again.ttft_ms, 12.5)
        self.assertEqual(again.generated_token_ids, [1, 2, 3])
        self.assertEqual(again.execution_status, EXEC_SUCCESS)

    def test_matches_schema(self) -> None:
        schema = load_schema(SCHEMAS_DIR / "raw_result.schema.json")
        record = self.make_result()
        validate_schema(record.to_dict(), schema)

    def test_schema_rejects_missing_required(self) -> None:
        schema = load_schema(SCHEMAS_DIR / "raw_result.schema.json")
        record = self.make_result()
        data = record.to_dict()
        del data["actual_output_tokens"]
        with self.assertRaises(SchemaError):
            validate_schema(data, schema)

    def test_schema_enums(self) -> None:
        schema = load_schema(SCHEMAS_DIR / "raw_result.schema.json")
        record = self.make_result(execution_status="EXPLODED")
        with self.assertRaises(SchemaError):
            validate_schema(record.to_dict(), schema)

    def test_failure_record(self) -> None:
        record = FailureRecord(
            benchmark_version="PRADIUM-RUNTIME-BENCH-v1",
            session_id="s",
            timestamp="t",
            suite="context",
            runtime="mock",
            model=None,
            profile="CTX-32768",
            concurrency=1,
            input_tokens=32768,
            requested_output_tokens=256,
            status=EXEC_OOM,
            error="simulated OOM",
        )
        payload = json.loads(record.to_json())
        self.assertEqual(payload["status"], "OOM")
        self.assertEqual(payload["input_tokens"], 32768)


if __name__ == "__main__":
    unittest.main()
