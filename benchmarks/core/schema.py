"""Minimal JSON-Schema validator for benchmark result artifacts.

Supports the schema subset used by ``schemas/*.json``: ``type``, ``required``,
``properties``, ``items``, ``enum``, ``minimum``/``maximum``, ``additionalProperties``
(as a bool). Full JSON-Schema compliance is intentionally out of scope; the
schemas are also consumable by external validators.
"""

from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    """Raised when data does not conform to the schema."""


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``data`` against ``schema``; raise :class:`SchemaError` on mismatch."""
    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        for t in types:
            check = _TYPE_CHECKS.get(t)
            if check is None:
                raise SchemaError(f"{path}: unsupported schema type {t!r}")
            if check(data):
                break
        else:
            raise SchemaError(
                f"{path}: expected type {types}, got {type(data).__name__}"
            )

    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(f"{path}: value {data!r} not in enum {schema['enum']}")

    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            raise SchemaError(f"{path}: {data} < minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            raise SchemaError(f"{path}: {data} > maximum {schema['maximum']}")

    if isinstance(data, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                raise SchemaError(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                validate_schema(value, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise SchemaError(f"{path}: unexpected property {key!r}")

    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            validate_schema(item, schema["items"], f"{path}[{i}]")


def load_schema(path: Any) -> dict[str, Any]:
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))
