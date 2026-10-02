"""Minimal, dependency-free YAML subset loader.

The benchmark configuration files use a small, deterministic subset of YAML:

* nested mappings (``key: value``)
* block sequences of scalars (``- item``)
* inline/flow sequences (``[a, b, c]``)
* scalars: int, float, bool, null, quoted and plain strings
* ``#`` comments and blank lines

Anything outside this subset raises :class:`YamlError` so configuration files
stay simple and parsing stays deterministic. This keeps the benchmark suite
free of third-party dependencies.
"""

from __future__ import annotations

import re
from typing import Any


class YamlError(ValueError):
    """Raised when a document is not valid within the supported YAML subset."""


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    """Strip a trailing comment, respecting single/double quoted scalars."""
    out: list[str] = []
    quote: str | None = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (not out or out[-1] in (" ", "\t")):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in ("~", "null", "Null", "NULL"):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_flow(inner)]
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1]
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        return text[1:-1]
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _split_flow(inner: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in inner:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
        elif ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _split_key(content: str, lineno: int) -> tuple[str, str]:
    """Split ``key: rest`` handling quoted keys and bare keys."""
    if content.startswith(("'", '"')):
        quote = content[0]
        end = content.find(quote, 1)
        if end < 0:
            raise YamlError(f"line {lineno}: unterminated quoted key")
        key = content[1:end]
        rest = content[end + 1 :]
        if not rest.startswith(":"):
            raise YamlError(f"line {lineno}: expected ':' after key {key!r}")
        return key, rest[1:].strip()
    idx = content.find(":")
    if idx < 0:
        raise YamlError(f"line {lineno}: expected 'key: value', got {content!r}")
    key = content[:idx].strip()
    rest = content[idx + 1 :].strip()
    if not key:
        raise YamlError(f"line {lineno}: empty key")
    return key, rest


def _logical_lines(text: str) -> list[tuple[int, str, int]]:
    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        leading = stripped[: len(stripped) - len(stripped.lstrip(" \t"))]
        if "\t" in leading:
            raise YamlError(f"line {lineno}: tabs are not allowed in indentation")
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip(), lineno))
    return lines


def loads(text: str) -> Any:
    """Parse a YAML-subset document into Python objects."""
    lines = _logical_lines(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise YamlError(f"line {lines[index][2]}: unexpected trailing content")
    return value


def load_file(path: Any) -> Any:
    """Parse a YAML-subset file."""
    from pathlib import Path

    return loads(Path(path).read_text(encoding="utf-8"))


def _parse_block(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[Any, int]:
    _, content, _ = lines[index]
    if content == "-" or content.startswith("- "):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content, lineno = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"line {lineno}: unexpected indentation")
        if content == "-" or content.startswith("- "):
            raise YamlError(f"line {lineno}: sequence item inside mapping block")
        key, rest = _split_key(content, lineno)
        if key in result:
            raise YamlError(f"line {lineno}: duplicate key {key!r}")
        if rest:
            result[key] = _parse_scalar(rest)
            index += 1
        else:
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                result[key], index = _parse_block(lines, index, lines[index][0])
            else:
                result[key] = None
    return result, index


def _parse_sequence(
    lines: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line_indent, content, lineno = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"line {lineno}: unexpected indentation")
        if not (content == "-" or content.startswith("- ")):
            break
        rest = content[1:].strip()
        if rest:
            if ":" in rest and not rest.startswith(("'", '"', "[")):
                raise YamlError(
                    f"line {lineno}: nested mappings in sequences are not "
                    "supported by the minimal YAML loader"
                )
            items.append(_parse_scalar(rest))
            index += 1
        else:
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                item, index = _parse_block(lines, index, lines[index][0])
                items.append(item)
            else:
                items.append(None)
    return items, index
