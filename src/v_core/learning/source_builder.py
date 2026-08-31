from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


class SourceBlueprintError(ValueError):
    """The runtime could not build a grounded tool contract from source."""


@dataclass(frozen=True, slots=True)
class SourceBlueprint:
    name: str
    description: str
    arguments: dict[str, Any]
    expected: dict[str, Any] | None
    oracle: str


_EXPLICIT_NAME = re.compile(
    r"(?:\b(?:called|named|name)\b|\b(?:nazw(?:a|ij|any|ane)|o\s+nazwie)\b)"
    r"[\s:=-]*[`\"']?([a-z][a-z0-9_]{2,63})",
    re.IGNORECASE,
)
_SNAKE_NAME = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9]+(?:_[a-z0-9]+)+)(?![A-Za-z0-9_])")
_IGNORED_NAMES = {
    "learning_create_tool",
    "snapshot_text",
    "timeout_seconds",
    "input_schema",
    "output_schema",
    "test_arguments",
    "test_expected",
}


def json_assignments(text: str) -> dict[str, list[Any]]:
    """Extract exact ``name = JSON`` literals without interpreting prose."""

    decoder = json.JSONDecoder()
    assignments: dict[str, list[Any]] = {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    )
    for match in pattern.finditer(text):
        remainder = text[match.end() :]
        leading = len(remainder) - len(remainder.lstrip())
        try:
            value, _ = decoder.raw_decode(remainder[leading:])
        except (TypeError, json.JSONDecodeError):
            continue
        assignments.setdefault(match.group(1), []).append(value)
    return assignments


def source_argument_defaults(source: str) -> tuple[set[str], dict[str, Any]]:
    """Return argument keys used by ``run`` and literal ``dict.get`` defaults."""

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise SourceBlueprintError(f"generated tool has invalid Python: {error}") from error

    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run is None:
        raise SourceBlueprintError("generated tool must define run(arguments)")
    if not run.args.args:
        raise SourceBlueprintError("run must accept one arguments object")
    argument_name = run.args.args[0].arg
    fields: set[str] = set()
    defaults: dict[str, Any] = {}
    for node in ast.walk(run):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id != argument_name:
                continue
            key = _literal_string(node.slice)
            if key:
                fields.add(key)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr != "get"
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id != argument_name
            or not node.args
        ):
            continue
        key = _literal_string(node.args[0])
        if not key:
            continue
        fields.add(key)
        if len(node.args) >= 2:
            try:
                default = ast.literal_eval(node.args[1])
                json.dumps(default, allow_nan=False)
            except (TypeError, ValueError):
                continue
            defaults[key] = default
    return fields, defaults


def build_source_blueprint(
    *,
    objective: str,
    source: str,
    observed_snapshot: str = "",
    name_hint: str = "",
    description_hint: str = "",
) -> SourceBlueprint:
    """Build identity and a concrete test fixture from trusted runtime context.

    The model supplies source only. Names and fixture values come from the owner's
    immutable objective, an actual observed browser snapshot, or literal defaults
    in the generated function. No domain-specific patching is involved.
    """

    assignments = json_assignments(objective)
    fields, defaults = source_argument_defaults(source)
    objective_fields = {
        name
        for name in assignments
        if name
        not in {
            "expected",
            "test_expected",
            "version",
            "timeout_seconds",
        }
    }
    ignored_fixture_fields = sorted(objective_fields - fields)
    if ignored_fixture_fields:
        rendered = ", ".join(ignored_fixture_fields)
        raise SourceBlueprintError(
            "generated source ignores concrete objective fixture fields: "
            f"{rendered}. Read every fixture field from the arguments object; "
            "do not hard-code the expected result."
        )
    name = _tool_name(objective, source, name_hint)
    description = description_hint.strip() or (
        f"Process bounded JSON input for the current PALADYN task with {name}."
    )
    arguments: dict[str, Any] = {}
    missing: list[str] = []
    for field in sorted(fields):
        values = assignments.get(field)
        if values:
            arguments[field] = values[0]
        elif field == "snapshot_text" and observed_snapshot:
            arguments[field] = observed_snapshot
        elif field in defaults:
            arguments[field] = defaults[field]
        else:
            missing.append(field)
    if missing:
        rendered = ", ".join(missing)
        raise SourceBlueprintError(
            "PALADYN cannot derive a concrete test fixture for generated-tool "
            f"input fields: {rendered}. Supply exact JSON assignments in the "
            "objective or use literal defaults in source."
        )

    expected_values = assignments.get("expected")
    expected = (
        expected_values[0]
        if expected_values and isinstance(expected_values[0], dict)
        else None
    )
    return SourceBlueprint(
        name=name,
        description=description[:500],
        arguments=arguments,
        expected=expected,
        oracle=("owner_expected" if expected is not None else "deterministic_smoke"),
    )


def schema_from_example(value: Any) -> dict[str, Any]:
    """Derive PALADYN's strict JSON-schema subset from one concrete value."""

    if isinstance(value, dict):
        properties = {
            str(name): schema_from_example(item) for name, item in value.items()
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        schemas = [schema_from_example(item) for item in value]
        return {
            "type": "array",
            "items": merge_example_schemas(schemas) if schemas else {"type": "null"},
        }
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    return {"type": "string"}


def merge_example_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    if not schemas:
        return {"type": "null"}
    kinds = {str(schema.get("type", "")) for schema in schemas}
    if kinds <= {"integer", "number"}:
        return {"type": "number" if "number" in kinds else "integer"}
    if len(kinds) != 1:
        return schemas[0]
    kind = next(iter(kinds))
    if kind == "object":
        names = {
            name for schema in schemas for name in schema.get("properties", {})
        }
        properties: dict[str, Any] = {}
        for name in sorted(names):
            children = [
                schema["properties"][name]
                for schema in schemas
                if name in schema.get("properties", {})
            ]
            properties[name] = merge_example_schemas(children)
        required_sets = [set(schema.get("required", [])) for schema in schemas]
        required = sorted(set.intersection(*required_sets)) if required_sets else []
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    if kind == "array":
        return {
            "type": "array",
            "items": merge_example_schemas([schema["items"] for schema in schemas]),
        }
    return {"type": kind}


def _tool_name(objective: str, source: str, hint: str) -> str:
    candidates = [hint.strip()]
    explicit = _EXPLICIT_NAME.search(objective)
    if explicit:
        candidates.append(explicit.group(1))
    candidates.extend(
        match.group(1) for match in _SNAKE_NAME.finditer(objective)
    )
    for candidate in candidates:
        lowered = candidate.casefold()
        if (
            lowered not in _IGNORED_NAMES
            and re.fullmatch(r"[a-z][a-z0-9_]{2,63}", lowered)
        ):
            return lowered
    digest = hashlib.sha256((objective + "\0" + source).encode("utf-8")).hexdigest()
    return f"generated_{digest[:12]}"


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Python <3.9 compatibility is harmless and keeps AST fixtures portable.
    if hasattr(ast, "Index") and isinstance(node, ast.Index):  # pragma: no cover
        return _literal_string(node.value)
    return ""
