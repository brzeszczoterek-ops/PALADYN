"""Static tool-interface review. Never invokes a tool or claims functional tests."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter


def review_catalog(definitions: list[dict], prompt: str) -> tuple[str, dict]:
    functions = [item.get("function", {}) for item in definitions if isinstance(item, dict)]
    functions = [item for item in functions if isinstance(item, dict) and isinstance(item.get("name"), str)]
    named = [item for item in functions if re.search(
        r"(?<!\w)" + re.escape(item["name"]) + r"(?!\w)", prompt, re.IGNORECASE
    )]
    selected = named or functions
    fingerprint = hashlib.sha256(json.dumps(
        selected, sort_keys=True, ensure_ascii=False, default=str
    ).encode()).hexdigest()
    if not selected:
        return (
            "Boss, discovery returned no tool definitions. I cannot assess their interfaces. No functional tests ran.",
            {"status": "empty", "count": 0, "schema_sha256": fingerprint},
        )
    counts = Counter(item["name"] for item in selected)
    findings = []
    lines = [
        f"Boss, I inspected {len(selected)} discovered tool definitions — metadata only, not their implementation.",
        "No tool was executed or tested. Runtime behavior, speed and reliability remain unverified.",
    ]
    for function in selected:
        name = function["name"]
        schema = function.get("parameters")
        issues = []
        if not str(function.get("description", "")).strip():
            issues.append("description is missing; document purpose and limitations")
        if counts[name] > 1:
            issues.append("duplicate name in catalog; resolve the ambiguous registration")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            issues.append("input schema is missing or not an object; publish a complete input contract")
        else:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                issues.append("properties or required has an invalid structure")
            else:
                undefined = [key for key in required if isinstance(key, str) and key not in properties]
                if undefined:
                    issues.append("required lists undeclared fields: " + ", ".join(undefined))
                for key, value in properties.items():
                    if not isinstance(value, dict):
                        issues.append(f"{key}: field schema is not an object")
                    elif not str(value.get("description", "")).strip():
                        issues.append(f"{key}: parameter meaning is undocumented; add a description")
        findings.append({"tool": name, "issues": issues})
        lines.append(f"- {name}: " + ("; ".join(issues) if issues else "no issue found by these limited metadata checks; functionality not tested"))
    lines.append("These are interface findings, not proof that a tool works or fails. No changes were made.")
    return "\n\n".join(lines), {
        "status": "reviewed", "scope": "discovered_metadata_only",
        "count": len(selected), "schema_sha256": fingerprint, "findings": findings,
        "functional_tests": 0,
    }
