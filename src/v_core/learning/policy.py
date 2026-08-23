from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from typing import Iterable

from v_core.autonomy import AuthorizationGuard

from .models import ArtifactScope, SkillManifest, ToolManifest
from .schema import validate_schema


class ArtifactPolicyError(PermissionError):
    pass


_SAFE_IMPORTS = {
    "base64",
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
    "sys",
    "time",
    "typing",
    "urllib",
}
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}
_PROTECTED_SKILL_PHRASES = {
    "ignore previous",
    "ignore the constitution",
    "ignore system",
    "override system",
    "disable safety",
    "disable the kill switch",
    "bypass permission",
    "reveal credentials",
    "read private key",
    "modify persona",
    "change persona",
}
_RESERVED_TOOL_NAMES = {
    "browser_click",
    "browser_find",
    "browser_navigate",
    "browser_press_key",
    "browser_snapshot",
    "create_directory",
    "directory_tree",
    "edit_file",
    "get_file_info",
    "list_directory",
    "move_file",
    "read_file",
    "sandbox_execute_offline",
    "search_files",
    "write_file",
}


@dataclass(slots=True)
class ArtifactPolicy:
    authorization: AuthorizationGuard

    def may_stage(self, scope: ArtifactScope) -> None:
        self.authorization.require("create_task_tools")
        if scope is ArtifactScope.PERSISTENT:
            self.authorization.require("owner:create_persistent_artifacts")

    def may_validate(self) -> None:
        self.authorization.require("run_task_tests")
        self.authorization.require("run_sandboxed_code")

    def may_activate(self, scope: ArtifactScope) -> None:
        if scope is ArtifactScope.TASK:
            self.authorization.require("activate_task_artifacts")
        else:
            self.authorization.require("owner:activate_persistent_artifacts")

    @staticmethod
    def validate_tool_manifest(manifest: ToolManifest) -> None:
        if (
            manifest.name in _RESERVED_TOOL_NAMES
            or manifest.name.startswith(("evm_", "learning_", "paladyn_"))
        ):
            raise ArtifactPolicyError(
                f"generated tool name is reserved: {manifest.name}"
            )
        validate_schema(manifest.input_schema)
        validate_schema(manifest.output_schema)
        if manifest.input_schema.get("type") != "object":
            raise ArtifactPolicyError("generated tool input must be a JSON object")
        if manifest.output_schema.get("type") != "object":
            raise ArtifactPolicyError("generated tool output must be a JSON object")
        if len(manifest.tests) > 100:
            raise ArtifactPolicyError("generated tools may define at most 100 tests")
        encoded = json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 2_000_000:
            raise ArtifactPolicyError("generated tool manifest exceeds 2 MB")

    @staticmethod
    def validate_tool_source_envelope(source: str) -> None:
        if not isinstance(source, str):
            raise ArtifactPolicyError("generated tool source must be text")
        if len(source.encode("utf-8")) > 200_000:
            raise ArtifactPolicyError("generated tool source exceeds 200 KB")
        if "\x00" in source:
            raise ArtifactPolicyError("generated tool source contains a NUL byte")

    @staticmethod
    def validate_tool_source(source: str) -> None:
        ArtifactPolicy.validate_tool_source_envelope(source)
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as error:
            raise ArtifactPolicyError(f"generated tool has invalid Python: {error}") from error

        has_run = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "run":
                    has_run = True
                if node.name.startswith("__") and node.name != "__init__":
                    raise ArtifactPolicyError("generated tool defines a protected dunder")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    ArtifactPolicy._validate_import(alias.name)
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ArtifactPolicyError("relative imports are not allowed")
                ArtifactPolicy._validate_import(node.module or "")
            if isinstance(node, ast.Call):
                name = ArtifactPolicy._call_name(node.func)
                if name in _FORBIDDEN_CALLS:
                    raise ArtifactPolicyError(f"forbidden generated-tool call: {name}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ArtifactPolicyError("dunder attribute access is not allowed")

        if not has_run:
            raise ArtifactPolicyError("generated tool must define run(arguments)")

    @staticmethod
    def validate_skill(
        manifest: SkillManifest,
        available_tools: Iterable[str],
    ) -> None:
        available = set(available_tools)
        missing = set(manifest.required_tools) - available
        if missing:
            raise ArtifactPolicyError(
                f"skill requires unavailable tools: {sorted(missing)}"
            )
        combined = " ".join(manifest.steps).casefold()
        found = sorted(phrase for phrase in _PROTECTED_SKILL_PHRASES if phrase in combined)
        if found:
            raise ArtifactPolicyError(
                f"skill attempts to modify protected policy: {found}"
            )
        encoded = json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 500_000:
            raise ArtifactPolicyError("generated skill manifest exceeds 500 KB")

    @staticmethod
    def skill_matches(manifest: SkillManifest, user_input: str) -> bool:
        normalized = " ".join(user_input.casefold().split())
        return any(trigger in normalized for trigger in manifest.triggers)

    @staticmethod
    def _validate_import(module: str) -> None:
        root = module.split(".", 1)[0]
        if root not in _SAFE_IMPORTS:
            raise ArtifactPolicyError(f"generated tool import is not allowed: {module}")

    @staticmethod
    def _call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""
