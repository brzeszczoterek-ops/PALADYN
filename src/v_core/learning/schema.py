from __future__ import annotations

import json
import math
from typing import Any


class SchemaError(ValueError):
    pass


_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "additionalProperties",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
}


def validate_schema(schema: dict[str, Any], *, depth: int = 0) -> None:
    if depth > 8:
        raise SchemaError("schema nesting exceeds eight levels")
    if not isinstance(schema, dict):
        raise SchemaError("schema must be an object")
    unknown = set(schema) - _KEYS
    if unknown:
        raise SchemaError(f"unsupported schema keys: {sorted(unknown)}")
    kind = schema.get("type")
    if kind not in _TYPES:
        raise SchemaError(f"unsupported or missing schema type: {kind!r}")
    applicable = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
        "boolean": set(),
        "null": set(),
    }[kind] | {"type", "enum"}
    irrelevant = set(schema) - applicable
    if irrelevant:
        raise SchemaError(
            f"schema keys are not valid for type {kind}: {sorted(irrelevant)}"
        )
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > 100:
            raise SchemaError("enum must contain between 1 and 100 values")
        try:
            json.dumps(enum, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SchemaError("enum contains a non-JSON value") from error
        for item in enum:
            if not _matches_kind(item, kind):
                raise SchemaError("enum contains a value of the wrong type")
    _validate_bounds(schema, kind)
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or len(properties) > 64:
            raise SchemaError("object properties must be a mapping of at most 64 items")
        for name, child in properties.items():
            if not isinstance(name, str) or not name or len(name) > 128:
                raise SchemaError("property names must be non-empty strings")
            validate_schema(child, depth=depth + 1)
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(required) != len(set(required))
            or any(item not in properties for item in required)
        ):
            raise SchemaError("required must contain declared property names")
        if not isinstance(schema.get("additionalProperties", False), bool):
            raise SchemaError("additionalProperties must be boolean")
    if kind == "array":
        if "items" not in schema:
            raise SchemaError("array schemas require items")
        validate_schema(schema["items"], depth=depth + 1)


def validate_instance(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> None:
    validate_schema(schema)
    kind = schema["type"]
    valid = _matches_kind(value, kind)
    if not valid:
        raise SchemaError(f"{path} must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path} is not an allowed enum value")
    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise SchemaError(f"{path} is missing required fields: {sorted(missing)}")
        if not schema.get("additionalProperties", False):
            extra = set(value) - set(properties)
            if extra:
                raise SchemaError(f"{path} has unexpected fields: {sorted(extra)}")
        for name, item in value.items():
            if name in properties:
                validate_instance(item, properties[name], path=f"{path}.{name}")
    elif kind == "array":
        length = len(value)
        _range(length, schema.get("minItems"), schema.get("maxItems"), path)
        for index, item in enumerate(value):
            validate_instance(item, schema["items"], path=f"{path}[{index}]")
    elif kind == "string":
        _range(len(value), schema.get("minLength"), schema.get("maxLength"), path)
    elif kind in {"integer", "number"}:
        _range(value, schema.get("minimum"), schema.get("maximum"), path)


def _matches_kind(value: Any, kind: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": _is_finite_number(value),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[kind]


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _range(value: float, minimum: Any, maximum: Any, path: str) -> None:
    if minimum is not None and value < minimum:
        raise SchemaError(f"{path} is below its minimum")
    if maximum is not None and value > maximum:
        raise SchemaError(f"{path} exceeds its maximum")


def _validate_bounds(schema: dict[str, Any], kind: str) -> None:
    pairs = []
    if kind in {"integer", "number"}:
        pairs.append(("minimum", "maximum", (int, float)))
    if kind == "string":
        pairs.append(("minLength", "maxLength", (int,)))
    if kind == "array":
        pairs.append(("minItems", "maxItems", (int,)))
    for lower_name, upper_name, expected_types in pairs:
        lower = schema.get(lower_name)
        upper = schema.get(upper_name)
        for name, value in ((lower_name, lower), (upper_name, upper)):
            if value is None:
                continue
            if (
                not isinstance(value, expected_types)
                or isinstance(value, bool)
                or (isinstance(value, float) and not math.isfinite(value))
                or value < 0 and kind in {"string", "array"}
            ):
                raise SchemaError(f"{name} has an invalid value")
        if lower is not None and upper is not None and lower > upper:
            raise SchemaError(f"{lower_name} cannot exceed {upper_name}")
