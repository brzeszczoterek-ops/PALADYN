from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote_plus, unquote, urlsplit
from uuid import uuid4

from .autonomy import AuthorizationEnvelope, AuthorizationGuard, review_task
from .config import Config
from .evm import (
    EVMAccessProfile,
    EVMCapability,
    EVMToolkit,
    FoundrySandboxRunner,
    FoundryUnavailable,
    OraclePolicy,
    OracleRound,
    SequencerStatus,
)
from .mcp_client import MCPClient
from .learning import (
    ArtifactScope,
    EvidenceOutcome,
    EvidenceSource,
    LearningEvidence,
    LearningRuntime,
    SkillManifest,
    ToolManifest,
    ToolTestCase,
)
from .learning.snapshot_extractor import (
    ACCESSIBILITY_PRODUCT_CARD_SOURCE,
    extract_accessibility_product_cards,
    product_card_fixture,
)
from .sandbox import (
    BubblewrapBackend,
    SandboxExecutor,
    SandboxLimits,
    SandboxSpec,
    SandboxUnavailable,
)
from .tools.filesystem import Filesystem


class MCPToolExecutionError(RuntimeError):
    pass


class MCPTools:

    def __init__(
        self,
        config: Config,
    ):
        self.learning_profile = getattr(config, "learning_profile", "client")
        #
        # MCP clients
        #

        self.filesystem_client = MCPClient(
            config.filesystem_server
        )

        self.browser_client = MCPClient(
            config.browser_server
        )

        #
        # Browser session state
        #

        self.browser_session = None
        self.browser_ready = False
        self.interaction_id = ""
        self.interaction_prompt = ""
        self._observed_browser_snapshot = ""
        self._web_discovered_urls: dict[str, str] = {}
        self._web_search_performed = False
        self._tool_definitions_cache: list[dict[str, Any]] | None = None
        autonomy_root = Path(
            getattr(config, "autonomy_root", config.workspace / ".paladyn_autonomy")
        )
        self.interactive_trace_root = autonomy_root / "interactive"

        #
        # Wrappers
        #

        self.filesystem = Filesystem(
            self.filesystem_client
        )

        envelope = AuthorizationEnvelope(
            workspace=str(config.workspace),
        )
        profile = (
            EVMAccessProfile.owner_lab()
            if config.evm_profile == "owner_lab"
            else EVMAccessProfile.client()
        )
        profile.apply(envelope)
        if self.learning_profile == "owner_lab":
            persistent_learning = {
                "owner:create_persistent_artifacts",
                "owner:activate_persistent_artifacts",
                "owner:privileged_generated_code",
            }
            envelope.capabilities.update(persistent_learning)
            envelope.owner_approved_capabilities.update(persistent_learning)
        self.authorization = AuthorizationGuard(
            Path.cwd(),
            envelope,
        )
        self.evm = EVMToolkit(self.authorization)
        try:
            backend = BubblewrapBackend()
            self.sandbox = SandboxExecutor(
                self.authorization,
                backend,
            )
            self.sandbox_error = ""
            try:
                self.foundry = FoundrySandboxRunner(backend)
                self.foundry_error = ""
            except FoundryUnavailable as exc:
                self.foundry = None
                self.foundry_error = str(exc)
            learning_root = Path(
                getattr(config, "learning_root", config.workspace / ".paladyn_learning")
            )
            self.learning = LearningRuntime(
                learning_root,
                self.authorization,
                backend,
            )
            self.learning_error = ""
        except SandboxUnavailable as exc:
            self.sandbox = None
            self.sandbox_error = str(exc)
            self.foundry = None
            self.foundry_error = str(exc)
            self.learning = None
            self.learning_error = str(exc)

    async def ensure_browser_session(self) -> None:

        if (
            self.browser_ready
            and self.browser_session is not None
        ):
            return

        params = self.browser_client.server_command

        from mcp import ClientSession
        from mcp.client.stdio import (
            StdioServerParameters,
            stdio_client,
        )

        self._browser_stdio = stdio_client(
            StdioServerParameters(
                command=params[0],
                args=params[1:],
            )
        )

        self._browser_streams = (
            await self._browser_stdio.__aenter__()
        )

        read_stream, write_stream = (
            self._browser_streams
        )

        self.browser_session = ClientSession(
            read_stream,
            write_stream,
        )

        await self.browser_session.__aenter__()

        print("[MCP] Initializing browser session...")

        await self.browser_session.initialize()

        print("[MCP] Browser session ready")

        self.browser_ready = True

    async def close_browser_session(self) -> None:

        if (
            not self.browser_ready
            or self.browser_session is None
        ):
            return

        try:

            await self.browser_session.__aexit__(
                None,
                None,
                None,
            )

        finally:

            if hasattr(
                self,
                "_browser_stdio",
            ):
                await self._browser_stdio.__aexit__(
                    None,
                    None,
                    None,
                )

            self.browser_session = None
            self.browser_ready = False

    def begin_interaction(self, interaction_id: str, prompt: str) -> None:
        self.interaction_id = str(interaction_id)[:128]
        self.interaction_prompt = str(prompt)[:20_000]
        self._web_discovered_urls = {}
        self._web_search_performed = False
        self._observed_browser_snapshot = ""

    def observe_browser_snapshot(self, snapshot_text: str) -> None:
        """Retain bounded, runtime-observed text for deterministic builders.

        The model never supplies this evidence back to PALADYN. This prevents a
        generated extractor test from silently replacing page facts with
        invented or truncated fixtures.
        """

        self._observed_browser_snapshot = str(snapshot_text)[:20_000]

    #
    # Filesystem shortcuts
    #

    async def ls(
        self,
        path: str = ".",
    ) -> list[str]:

        return await self.filesystem.list_directory(
            path
        )

    async def cat(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.read_file(
            path
        )

    async def write(
        self,
        path: str,
        content: str,
    ) -> str:

        return await self.filesystem.write_file(
            path,
            content,
        )

    async def edit(
        self,
        path: str,
        edits: list[dict],
        dry_run: bool = False,
    ) -> str:

        return await self.filesystem.edit_file(
            path,
            edits,
            dry_run,
        )

    async def mkdir(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.make_directory(
            path
        )

    async def move(
        self,
        source: str,
        destination: str,
    ) -> str:

        return await self.filesystem.move_file(
            source,
            destination,
        )

    async def search(
        self,
        path: str,
        pattern: str,
    ) -> str:

        return await self.filesystem.search_files(
            path,
            pattern,
        )

    async def tree(
        self,
        path: str = ".",
    ) -> str:

        return await self.filesystem.directory_tree(
            path
        )

    async def info(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.get_file_info(
            path
        )

    #
    # Browser MCP
    #

    async def browser_call(
        self,
        tool: str,
        arguments: dict,
    ) -> str:

        await self.ensure_browser_session()

        result = await self.browser_session.call_tool(
            tool,
            arguments,
        )

        text = []

        for item in result.content or []:

            if hasattr(
                item,
                "text",
            ):
                text.append(
                    item.text
                )

        output = "\n".join(text)
        if bool(
            getattr(result, "isError", False)
            or getattr(result, "is_error", False)
        ):
            raise MCPToolExecutionError(
                output[:2_000] or f"browser tool {tool} returned an error"
            )
        return output

    @staticmethod
    def _normalized_web_url(url: str) -> str:
        try:
            parsed = urlsplit(url.strip())
        except ValueError:
            return ""
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        path = parsed.path.rstrip("/") or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{parsed.scheme.casefold()}://{parsed.netloc.casefold()}{path}{query}"

    @staticmethod
    def _url_copy_fingerprint(url: str) -> str:
        """Ignore punctuation that local models commonly corrupt while copying."""

        try:
            parsed = urlsplit(str(url).strip())
        except ValueError:
            return ""
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        evidence = unquote(
            f"{parsed.netloc}{parsed.path}?{parsed.query}"
        ).casefold()
        return "".join(character for character in evidence if character.isalnum())

    def _unique_discovered_copy_match(self, requested_url: str) -> str:
        """Return one exact punctuation-insensitive match from search evidence."""

        fingerprint = self._url_copy_fingerprint(requested_url)
        if not fingerprint:
            return ""
        matches = {
            original
            for original in self._web_discovered_urls.values()
            if self._url_copy_fingerprint(original) == fingerprint
        }
        return next(iter(matches)) if len(matches) == 1 else ""

    @classmethod
    def _search_results_from_snapshot(
        cls,
        snapshot: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Extract grounded result URLs and labels from a DuckDuckGo snapshot."""

        results: list[dict[str, Any]] = []
        by_url: dict[str, dict[str, Any]] = {}
        pending_label = ""

        for line in snapshot.splitlines():
            link = re.search(r'\blink\s+"([^"]+)"', line, re.IGNORECASE)
            if link is not None:
                pending_label = " ".join(link.group(1).split()).strip()

            target = re.search(r"- /url:\s*(https?://\S+)", line, re.IGNORECASE)
            if target is None:
                continue
            url = target.group(1).rstrip("`\"'),.;:]}\\")
            normalized = cls._normalized_web_url(url)
            if not normalized:
                continue
            hostname = (urlsplit(normalized).hostname or "").casefold()
            if hostname.endswith("duckduckgo.com"):
                continue

            label = pending_label
            if (
                not label
                or label.casefold().startswith("search domain ")
                or label.casefold().startswith(("http://", "https://"))
                or " › " in label
            ):
                label = hostname.removeprefix("www.")

            existing = by_url.get(normalized)
            if existing is None:
                existing = {
                    "rank": len(results) + 1,
                    "title": label,
                    "url": url,
                }
                by_url[normalized] = existing
                results.append(existing)
            elif existing["title"] in {
                hostname,
                hostname.removeprefix("www."),
            } and label not in {
                hostname,
                hostname.removeprefix("www."),
            }:
                existing["title"] = label

        return results[:limit]

    async def web_search(self, query: str, max_results: int = 6) -> str:
        """Run a real search and return a compact, model-friendly result list."""

        query = " ".join(str(query).split()).strip()
        if not query:
            return self._json({"error": "web_search requires a non-empty query."})
        limit = max(1, min(int(max_results), 10))
        search_url = "https://duckduckgo.com/?q=" + quote_plus(query) + "&ia=web"
        self._web_search_performed = True
        await self.browser_call("browser_navigate", {"url": search_url})
        snapshot = await self.browser_call("browser_snapshot", {})
        results = self._search_results_from_snapshot(snapshot, limit=limit)
        for result in results:
            normalized = self._normalized_web_url(str(result["url"]))
            if normalized:
                self._web_discovered_urls[normalized] = str(result["url"])
        return self._json(
            {
                "query": query,
                "engine": "duckduckgo",
                "search_url": search_url,
                "result_count": len(results),
                "results": results,
            }
        )

    async def web_read(self, url: str) -> str:
        """Open a verified result and return its actual accessibility snapshot."""

        model_requested_url = url
        normalized = self._normalized_web_url(url)
        if not normalized:
            return self._json({"error": "web_read requires an HTTP or HTTPS URL."})

        if self._web_search_performed and normalized not in self._web_discovered_urls:
            corrected = self._unique_discovered_copy_match(url)
            if corrected:
                url = corrected
                normalized = self._normalized_web_url(url)
            else:
                return self._json(
                    {
                        "error": "web_read rejected a URL absent from web_search evidence.",
                        "requested_url": model_requested_url,
                        "allowed_urls": list(self._web_discovered_urls.values())[:10],
                    }
                )

        navigation = await self.browser_call("browser_navigate", {"url": url})
        snapshot = await self.browser_call("browser_snapshot", {})
        actual_url = re.search(r"^- Page URL:\s*(\S+)", snapshot, re.MULTILINE)
        page_title = re.search(r"^- Page Title:\s*(.+)$", snapshot, re.MULTILINE)
        for candidate in re.findall(r"https?://[^\s<>\[\](){}\"']+", snapshot):
            discovered = candidate.rstrip("`\"'),.;:]}\\")
            key = self._normalized_web_url(discovered)
            if key:
                self._web_discovered_urls[key] = discovered
        return self._json(
            {
                "requested_url": model_requested_url,
                "corrected_url": url if url != model_requested_url else "",
                "url": actual_url.group(1) if actual_url else url,
                "title": page_title.group(1).strip() if page_title else "",
                "navigation": navigation[:1_000],
                "content": snapshot,
            }
        )

    #
    # Metadata
    #

    async def tools(self) -> list[str]:

        try:
            filesystem_tools = await self.filesystem_client.list_tools()
            filesystem_names = [tool.name for tool in filesystem_tools.tools]
        except Exception:
            filesystem_names = []

        try:
            browser_tools = await self.browser_client.list_tools()
            browser_names = [tool.name for tool in browser_tools.tools]
        except Exception:
            browser_names = []

        return self.local_tool_names() + filesystem_names + browser_names

    async def openai_tool_definitions(self) -> list[dict[str, Any]]:
        """Return executable tool metadata in the OpenAI function format.

        The MCP servers own their schemas. PALADYN only supplies local schemas
        for capabilities implemented directly in this process. The result is
        cached because spawning MCP discovery processes on every reasoning step
        is both slow and a source of noisy shutdown failures.
        """

        if self._tool_definitions_cache is not None:
            return self._tool_definitions_cache

        definitions = self._local_tool_definitions()
        discovered_names = {item["function"]["name"] for item in definitions}
        callable_names = self._known_tool_names()
        for client in (self.filesystem_client, self.browser_client):
            try:
                result = await client.list_tools()
            except Exception as error:
                print(f"[MCP] Tool schema discovery failed: {type(error).__name__}: {error}")
                continue
            for tool in result.tools:
                name = str(getattr(tool, "name", "") or "").strip()
                if (
                    not name
                    or name in discovered_names
                    or name not in callable_names
                ):
                    continue
                schema = getattr(tool, "inputSchema", None)
                if schema is None:
                    schema = getattr(tool, "input_schema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                definitions.append(
                    self._tool_definition(
                        name,
                        str(getattr(tool, "description", "") or f"Execute {name}."),
                        schema,
                    )
                )
                discovered_names.add(name)

        # Keep the central protocol available even when an MCP discovery
        # subprocess is temporarily unavailable.
        for item in self._fallback_mcp_tool_definitions():
            name = item["function"]["name"]
            if name not in discovered_names:
                definitions.append(item)
                discovered_names.add(name)

        self._tool_definitions_cache = definitions
        return definitions

    @staticmethod
    def _tool_definition(
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    def _local_tool_definitions(self) -> list[dict[str, Any]]:
        object_schema = {"type": "object", "properties": {}}
        generated_schema = {
            "type": "object",
            "additionalProperties": True,
        }
        tool_test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object", "additionalProperties": True},
                "expected": {"type": "object", "additionalProperties": True},
            },
            "required": ["name", "arguments", "expected"],
            "additionalProperties": False,
        }
        tool_manifest_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "input_schema": generated_schema,
                "output_schema": generated_schema,
                "tests": {"type": "array", "items": tool_test_schema},
                "scope": {"type": "string", "enum": ["task", "persistent"]},
                "lesson_ids": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "number", "minimum": 0.05, "maximum": 120},
            },
            "required": [
                "name",
                "version",
                "description",
                "input_schema",
                "output_schema",
                "tests",
            ],
            "additionalProperties": False,
        }
        skill_test_schema = {
            "type": "object",
            "properties": {
                "user_input": {"type": "string"},
                "should_match": {"type": "boolean"},
            },
            "required": ["user_input", "should_match"],
            "additionalProperties": False,
        }
        skill_manifest_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "triggers": {"type": "array", "items": {"type": "string"}},
                "steps": {"type": "array", "items": {"type": "string"}},
                "required_tools": {"type": "array", "items": {"type": "string"}},
                "tests": {"type": "array", "items": skill_test_schema},
                "scope": {"type": "string", "enum": ["task", "persistent"]},
                "lesson_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "name",
                "version",
                "description",
                "triggers",
                "steps",
                "required_tools",
                "tests",
            ],
            "additionalProperties": False,
        }
        evidence_schema = {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["self_review", "user_correction"],
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "correction", "regression"],
                },
                "summary": {"type": "string", "minLength": 1},
                "expected": {"type": "string"},
                "actual": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object", "additionalProperties": True},
            },
            "required": ["source", "outcome", "summary"],
            "additionalProperties": False,
        }
        lesson_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "hypothesis": {"type": "string", "minLength": 1},
                "trigger": {"type": "string", "minLength": 1},
                "action": {"type": "string", "minLength": 1},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": [
                "title",
                "hypothesis",
                "trigger",
                "action",
                "evidence_ids",
            ],
            "additionalProperties": False,
        }
        schemas: dict[str, tuple[str, dict[str, Any]]] = {
            "web_search": (
                "Search the public web through DuckDuckGo and return grounded result "
                "titles and exact URLs. Use this before web_read when the user did "
                "not provide a URL.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            "web_read": (
                "Open an exact URL returned by web_search (or explicitly supplied "
                "by the user) and return grounded page content. Never invent the URL.",
                {
                    "type": "object",
                    "properties": {"url": {"type": "string", "minLength": 1}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
            "evm_analyze_erc20_abi": (
                "Analyze an ERC-20 ABI locally.",
                {
                    "type": "object",
                    "properties": {"abi": {"type": "array", "items": {}}},
                    "required": ["abi"],
                    "additionalProperties": False,
                },
            ),
            "evm_analyze_solidity_security": (
                "Run local static security checks over Solidity source.",
                {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "additionalProperties": False,
                },
            ),
            "sandbox_execute_offline": (
                "Execute a command in PALADYN's isolated offline sandbox.",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"}},
                        "workspace": {"type": "string"},
                        "working_directory": {"type": "string"},
                        "environment": {"type": "object"},
                        "timeout_seconds": {"type": "number", "minimum": 1},
                    },
                    "required": ["command", "workspace"],
                    "additionalProperties": True,
                },
            ),
            "learning_create_tool": (
                "Create, quarantine, test, and activate a deterministic offline "
                "Python tool from source defining run(arguments). The normal agent "
                "path supplies source only: PALADYN derives the name, description, "
                "concrete fixture, strict schemas, validation contract, and lifecycle "
                "from immutable task context and runtime-observed data. An optional "
                "explicit test remains available to expert callers. "
                + (
                    "OWNER LAB: generated code may use arbitrary Python imports, "
                    "dynamic code, subprocesses, and file operations inside the "
                    "isolated sandbox."
                    if self.learning_profile == "owner_lab"
                    else "CLIENT: generated code uses the restricted source policy."
                ),
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "test": tool_test_schema,
                        "version": {"type": "string"},
                        "scope": {"type": "string", "enum": ["task", "persistent"]},
                        "timeout_seconds": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 120,
                        },
                    },
                    "required": ["source"],
                    "additionalProperties": False,
                },
            ),
            "learning_create_snapshot_extractor": (
                "Create, quarantine, validate, and activate a task-scoped offline "
                "product-card extractor from PALADYN's latest observed accessibility "
                "snapshot. PALADYN writes the Python and exact regression fixture; "
                "provide only the generated tool name.",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            ),
            "learning_record_evidence": (
                "Record bounded self-review or Boss-supplied correction evidence.",
                evidence_schema,
            ),
            "learning_propose_lesson": (
                "Propose a reusable lesson grounded in existing evidence IDs.",
                lesson_schema,
            ),
            "learning_stage_tool": (
                "Stage a deterministic offline tool in quarantine without activating it.",
                {
                    "type": "object",
                    "properties": {
                        "manifest": tool_manifest_schema,
                        "source": {"type": "string"},
                    },
                    "required": ["manifest", "source"],
                    "additionalProperties": False,
                },
            ),
            "learning_create_skill": (
                "Create, validate, and activate a reusable orchestration skill.",
                {
                    "type": "object",
                    "properties": {"manifest": skill_manifest_schema},
                    "required": ["manifest"],
                    "additionalProperties": False,
                },
            ),
            "learning_stage_skill": (
                "Stage a reusable orchestration skill in quarantine.",
                {
                    "type": "object",
                    "properties": {"manifest": skill_manifest_schema},
                    "required": ["manifest"],
                    "additionalProperties": False,
                },
            ),
            "learning_validate_artifact": (
                "Validate a quarantined generated artifact.",
                {
                    "type": "object",
                    "properties": {"artifact_id": {"type": "string"}},
                    "required": ["artifact_id"],
                    "additionalProperties": False,
                },
            ),
            "learning_activate_artifact": (
                "Activate a generated artifact that passed validation.",
                {
                    "type": "object",
                    "properties": {"artifact_id": {"type": "string"}},
                    "required": ["artifact_id"],
                    "additionalProperties": False,
                },
            ),
            "learning_retire_artifact": (
                "Retire an active generated artifact.",
                {
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["artifact_id", "reason"],
                    "additionalProperties": False,
                },
            ),
            "learning_list_artifacts": (
                "List generated artifacts visible in the current scope.",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            "runtime_review_task": (
                "Audit PALADYN's own prior interactive execution log. Returns only "
                "runtime-grounded findings with exact tool-call or context-rollover "
                "references. Omit task_id to review the most recent prior task.",
                {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "pattern": "^interactive-[A-Za-z0-9_-]+$",
                        }
                    },
                    "additionalProperties": False,
                },
            ),
        }
        descriptions = {
            "learning_record_evidence": "Record bounded evidence for later learning.",
            "learning_propose_lesson": "Propose a lesson grounded in recorded evidence.",
            "learning_stage_tool": "Stage generated tool code in quarantine.",
            "learning_stage_skill": "Stage a generated skill in quarantine.",
            "learning_list_artifacts": "List generated artifact lifecycle records.",
            "evm_validate_oracle": "Validate oracle round data against a local policy.",
            "evm_decode_uniswap_v4_hook": "Decode Uniswap v4 hook address flags.",
            "evm_quote_flash_swap": "Calculate a local flash-swap repayment quote.",
            "evm_foundry_test_offline": "Run Foundry tests inside the offline sandbox.",
        }
        if self.learning is not None:
            for definition in self.learning.active_tool_definitions():
                schemas[str(definition["name"])] = (
                    str(definition["description"]),
                    dict(definition["parameters"]),
                )
        definitions: list[dict[str, Any]] = []
        for name in self.local_tool_names():
            description, schema = schemas.get(
                name,
                (descriptions.get(name, f"Execute PALADYN local tool {name}."), object_schema),
            )
            definitions.append(self._tool_definition(name, description, schema))
        return definitions

    def _fallback_mcp_tool_definitions(self) -> list[dict[str, Any]]:
        properties = lambda **items: {
            "type": "object",
            "properties": items,
            "required": list(items),
            "additionalProperties": False,
        }
        specs = {
            "browser_navigate": (
                "Navigate the controlled browser to a URL.",
                properties(url={"type": "string"}),
            ),
            "browser_snapshot": (
                "Capture the current page as grounded textual evidence.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "browser_click": (
                "Click an element from the current browser snapshot.",
                properties(element={"type": "string"}),
            ),
            "browser_find": (
                "Find text in the current browser page.",
                properties(text={"type": "string"}),
            ),
            "browser_press_key": (
                "Press a key in the controlled browser.",
                properties(key={"type": "string"}),
            ),
            "browser_type": (
                "Type text into an editable browser element and optionally submit it.",
                {
                    "type": "object",
                    "properties": {
                        "element": {"type": "string"},
                        "target": {"type": "string"},
                        "text": {"type": "string"},
                        "submit": {"type": "boolean"},
                        "slowly": {"type": "boolean"},
                    },
                    "required": ["target", "text"],
                    "additionalProperties": False,
                },
            ),
            "read_file": (
                "Read a local workspace file.",
                properties(path={"type": "string"}),
            ),
            "write_file": (
                "Write a local workspace file.",
                properties(path={"type": "string"}, content={"type": "string"}),
            ),
            "list_directory": (
                "List a local workspace directory.",
                properties(path={"type": "string"}),
            ),
            "directory_tree": (
                "Return a local workspace directory tree.",
                properties(path={"type": "string"}),
            ),
            "search_files": (
                "Search for files under a local workspace path.",
                properties(path={"type": "string"}, pattern={"type": "string"}),
            ),
            "get_file_info": (
                "Read local workspace file metadata.",
                properties(path={"type": "string"}),
            ),
            "create_directory": (
                "Create a local workspace directory.",
                properties(path={"type": "string"}),
            ),
            "move_file": (
                "Move a local workspace file.",
                properties(source={"type": "string"}, destination={"type": "string"}),
            ),
            "edit_file": (
                "Apply structured edits to a local workspace file.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "edits": {"type": "array", "items": {"type": "object"}},
                        "dry_run": {"type": "boolean"},
                    },
                    "required": ["path", "edits"],
                    "additionalProperties": False,
                },
            ),
        }
        return [self._tool_definition(name, *spec) for name, spec in specs.items()]

    def local_tool_names(self) -> list[str]:
        names = [
            "web_search",
            "web_read",
            "evm_analyze_erc20_abi",
            "evm_validate_oracle",
            "evm_analyze_solidity_security",
            "sandbox_execute_offline",
            "learning_record_evidence",
            "learning_propose_lesson",
            "learning_stage_tool",
            "learning_stage_skill",
            "learning_create_tool",
            "learning_create_snapshot_extractor",
            "learning_create_skill",
            "learning_validate_artifact",
            "learning_activate_artifact",
            "learning_retire_artifact",
            "learning_list_artifacts",
            "runtime_review_task",
        ]
        if self.authorization.envelope.allows(
            EVMCapability.UNISWAP_HOOKS_SIMULATE.value
        ):
            names.extend(
                [
                    "evm_decode_uniswap_v4_hook",
                    "evm_quote_flash_swap",
                    "evm_foundry_test_offline",
                ]
            )
        if self.learning is not None:
            names.extend(self.learning.active_tool_names())
        return list(dict.fromkeys(names))

    def render_matching_skills(self, prompt: str) -> str:
        if self.learning is None:
            return ""
        return self.learning.render_matching_skills(prompt)

    def capture_tool_failure(
        self,
        *,
        task_id: str,
        tool: str,
        arguments: dict[str, Any],
        error: str,
    ) -> dict[str, Any] | None:
        if self.learning is None:
            return None
        evidence = self.learning.capture_tool_failure(
            task_id=task_id,
            tool=tool,
            arguments=arguments,
            error=error,
        )
        return evidence.to_dict()

    def _known_tool_names(self) -> set[str]:
        return set(self.local_tool_names()) | {
            "browser_click",
            "browser_find",
            "browser_navigate",
            "browser_press_key",
            "browser_snapshot",
            "browser_type",
            "create_directory",
            "directory_tree",
            "edit_file",
            "get_file_info",
            "list_directory",
            "move_file",
            "read_file",
            "search_files",
            "write_file",
        }

    async def tool_info(
        self,
        name: str,
    ):

        filesystem_tools = (
            await self.filesystem_client.list_tools()
        )

        for tool in filesystem_tools.tools:

            if tool.name == name:
                return tool

        browser_tools = (
            await self.browser_client.list_tools()
        )

        for tool in browser_tools.tools:

            if tool.name == name:
                return tool

        return None

    #
    # Dispatcher
    #

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any] | str = "",
    ):

        tool = tool.strip()

        structured = arguments if isinstance(arguments, dict) else None

        if self.learning is not None and tool in self.learning.active_tool_names():
            if structured is None:
                return f"Generated tool {tool} requires structured JSON arguments."
            result = await self.learning.execute_tool(tool, structured)
            return self._json(result)

        def value(name: str, default: str = "") -> str:
            if structured is not None:
                item = structured.get(name, default)
                return str(item) if item is not None else default
            return arguments or default

        #
        # Filesystem aliases + native MCP names
        #

        match tool:

            case "web_search":
                if structured is None:
                    return "web_search requires structured arguments."
                try:
                    max_results = int(structured.get("max_results", 6))
                except (TypeError, ValueError):
                    return "web_search requires max_results to be an integer."
                return await self.web_search(value("query"), max_results)

            case "web_read":
                if structured is None:
                    return "web_read requires structured arguments."
                return await self.web_read(value("url"))

            case "learning_record_evidence":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return "learning_record_evidence requires structured arguments."
                requested_source = EvidenceSource(value("source", "self_review"))
                is_correction = (
                    requested_source is EvidenceSource.USER_CORRECTION
                    and bool(self.interaction_prompt)
                )
                evidence = LearningEvidence(
                    task_id=(self.interaction_id or f"unbound-{uuid4().hex}"),
                    source=(
                        EvidenceSource.USER_CORRECTION
                        if is_correction
                        else EvidenceSource.SELF_REVIEW
                    ),
                    outcome=(
                        EvidenceOutcome.CORRECTION
                        if is_correction
                        else EvidenceOutcome(value("outcome"))
                    ),
                    summary=(
                        f"Boss's directly supplied correction: {self.interaction_prompt}"
                        if is_correction
                        else value("summary")
                    ),
                    expected=value("expected"),
                    actual=(self.interaction_prompt if is_correction else value("actual")),
                    confidence=min(float(structured.get("confidence", 0.0)), 0.85),
                    verified=False,
                    metadata=(
                        structured.get("metadata", {})
                        if isinstance(structured.get("metadata", {}), dict)
                        else {}
                    ),
                )
                return self._json(self.learning.record_evidence(evidence).to_dict())

            case "learning_propose_lesson":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None or not isinstance(
                    structured.get("evidence_ids"), list
                ):
                    return "learning_propose_lesson requires an evidence_ids array."
                lesson = self.learning.propose_lesson(
                    title=value("title"),
                    hypothesis=value("hypothesis"),
                    trigger=value("trigger"),
                    action=value("action"),
                    evidence_ids=[str(item) for item in structured["evidence_ids"]],
                )
                return self._json(lesson.to_dict())

            case "learning_stage_tool":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None or not isinstance(structured.get("manifest"), dict):
                    return "learning_stage_tool requires manifest and source."
                record = self.learning.stage_tool(
                    ToolManifest.from_dict(structured["manifest"]),
                    value("source"),
                )
                return self._json(record.to_dict())

            case "learning_create_tool":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return "learning_create_tool requires structured arguments."
                raw_manifest = structured.get("manifest")
                if isinstance(raw_manifest, dict):
                    # Backwards-compatible expert API. The runtime-facing model
                    # receives the smaller blueprint schema above.
                    manifest = ToolManifest.from_dict(raw_manifest)
                    record = await self.learning.create_tool(
                        manifest,
                        value("source"),
                    )
                elif isinstance(structured.get("test"), dict):
                    raw_test = structured.get("test")
                    assert isinstance(raw_test, dict)
                    test = ToolTestCase(
                        name=str(raw_test.get("name", "generated tool test")),
                        arguments=dict(raw_test.get("arguments", {})),
                        expected=dict(raw_test.get("expected", {})),
                    )
                    raw_version = value("version", "1.0.0")
                    manifest = ToolManifest(
                        name=value("name"),
                        version=(
                            raw_version
                            if re.fullmatch(r"\d+\.\d+\.\d+", raw_version)
                            else "1.0.0"
                        ),
                        description=value("description"),
                        input_schema=self._schema_from_example(test.arguments),
                        output_schema=self._schema_from_example(test.expected),
                        tests=(test,),
                        scope=ArtifactScope(value("scope", "task")),
                        lesson_ids=(),
                        timeout_seconds=float(structured.get("timeout_seconds", 10.0)),
                    )
                    record = await self.learning.create_tool(
                        manifest,
                        value("source"),
                    )
                else:
                    raw_version = value("version", "1.0.0")
                    version = (
                        raw_version
                        if re.fullmatch(r"\d+\.\d+\.\d+", raw_version)
                        else "1.0.0"
                    )
                    record = await self.learning.create_tool_from_source(
                        value("source"),
                        objective=self.interaction_prompt,
                        observed_snapshot=self._observed_browser_snapshot,
                        name_hint=value("name"),
                        description_hint=value("description"),
                        version=version,
                        scope=ArtifactScope(value("scope", "task")),
                        timeout_seconds=float(structured.get("timeout_seconds", 10.0)),
                    )
                self._tool_definitions_cache = None
                return self._json(record.to_dict())

            case "learning_create_snapshot_extractor":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return (
                        "learning_create_snapshot_extractor requires structured "
                        "arguments."
                    )
                fixture = product_card_fixture(
                    self._observed_browser_snapshot,
                    maximum_records=3,
                )
                records = extract_accessibility_product_cards(fixture)
                if len(records) < 3:
                    raise MCPToolExecutionError(
                        "the latest runtime-observed accessibility snapshot does not "
                        "contain three complete product cards with title, price, "
                        "availability, and relative URL"
                    )
                test = ToolTestCase(
                    name="extract three observed product cards",
                    arguments={"snapshot_text": fixture},
                    expected={"records": records[:3]},
                )
                manifest = ToolManifest(
                    name=value("name"),
                    version="1.0.0",
                    description=(
                        "Extract title, price, availability, and relative product URL "
                        "from accessibility product-card text."
                    ),
                    input_schema=self._schema_from_example(test.arguments),
                    output_schema=self._schema_from_example(test.expected),
                    tests=(test,),
                    scope=ArtifactScope.TASK,
                    lesson_ids=(),
                    timeout_seconds=10.0,
                )
                record = await self.learning.create_tool(
                    manifest,
                    ACCESSIBILITY_PRODUCT_CARD_SOURCE,
                )
                self._tool_definitions_cache = None
                return self._json(record.to_dict())

            case "learning_stage_skill":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None or not isinstance(structured.get("manifest"), dict):
                    return "learning_stage_skill requires a manifest."
                record = self.learning.stage_skill(
                    SkillManifest.from_dict(structured["manifest"])
                )
                return self._json(record.to_dict())

            case "learning_create_skill":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None or not isinstance(structured.get("manifest"), dict):
                    return "learning_create_skill requires a manifest."
                record = await self.learning.create_skill(
                    SkillManifest.from_dict(structured["manifest"]),
                    available_tools=self._known_tool_names(),
                )
                return self._json(record.to_dict())

            case "learning_validate_artifact":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return "learning_validate_artifact requires artifact_id."
                record = await self.learning.validate_artifact(
                    value("artifact_id"),
                    available_tools=self._known_tool_names(),
                )
                return self._json(record.to_dict())

            case "learning_activate_artifact":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return "learning_activate_artifact requires artifact_id."
                record = self.learning.activate_artifact(value("artifact_id"))
                self._tool_definitions_cache = None
                return self._json(record.to_dict())

            case "learning_retire_artifact":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                if structured is None:
                    return "learning_retire_artifact requires artifact_id and reason."
                return self._json(
                    self.learning.retire_artifact(
                        value("artifact_id"), value("reason")
                    ).to_dict()
                )

            case "learning_list_artifacts":
                if self.learning is None:
                    return f"Learning runtime unavailable: {self.learning_error}"
                return self._json({"artifacts": self.learning.list_artifacts()})

            case "runtime_review_task":
                try:
                    report = review_task(
                        self.interactive_trace_root,
                        task_id=value("task_id"),
                        exclude_task_id=("" if value("task_id") else self.interaction_id),
                    )
                except ValueError as error:
                    return self._json({"error": str(error), "findings": []})
                return self._json(report)

            case "evm_analyze_erc20_abi":
                if structured is None or not isinstance(structured.get("abi"), list):
                    return "evm_analyze_erc20_abi requires an ABI array."
                report = self.evm.analyze_erc20(structured["abi"])
                return self._json(
                    {
                        "interface_conformant": report.interface_conformant,
                        "functions": sorted(report.functions),
                        "events": sorted(report.events),
                        "optional_metadata": sorted(report.optional_metadata),
                        "findings": [asdict(item) for item in report.findings],
                    }
                )

            case "evm_validate_oracle":
                if structured is None:
                    return "evm_validate_oracle requires structured arguments."
                data = OracleRound(
                    round_id=int(structured["round_id"]),
                    answer=int(structured["answer"]),
                    started_at=int(structured.get("started_at", 0)),
                    updated_at=int(structured["updated_at"]),
                    answered_in_round=int(structured["answered_in_round"]),
                    decimals=int(structured["decimals"]),
                )
                policy = OraclePolicy(
                    max_age_seconds=int(structured["max_age_seconds"]),
                    require_positive=bool(structured.get("require_positive", True)),
                    minimum_answer=self._decimal_or_none(
                        structured.get("minimum_answer")
                    ),
                    maximum_answer=self._decimal_or_none(
                        structured.get("maximum_answer")
                    ),
                )
                sequencer_data = structured.get("sequencer")
                sequencer = None
                if isinstance(sequencer_data, dict):
                    sequencer = SequencerStatus(
                        is_up=bool(sequencer_data["is_up"]),
                        started_at=int(sequencer_data["started_at"]),
                        grace_period_seconds=int(
                            sequencer_data.get("grace_period_seconds", 3_600)
                        ),
                    )
                report = self.evm.validate_oracle(
                    data,
                    policy,
                    now=int(structured["now"]),
                    sequencer=sequencer,
                )
                return self._json(
                    {
                        "acceptable": report.acceptable,
                        "value": report.value,
                        "age_seconds": report.age_seconds,
                        "findings": [asdict(item) for item in report.findings],
                    }
                )

            case "evm_analyze_solidity_security":
                if structured is None or not isinstance(structured.get("source"), str):
                    return "evm_analyze_solidity_security requires Solidity source."
                return self._json(
                    {
                        "findings": [
                            asdict(item)
                            for item in self.evm.analyze_security(structured["source"])
                        ]
                    }
                )

            case "evm_decode_uniswap_v4_hook":
                if structured is None:
                    return "evm_decode_uniswap_v4_hook requires an address."
                return self._json(asdict(self.evm.decode_hook(value("address"))))

            case "evm_quote_flash_swap":
                if structured is None:
                    return "evm_quote_flash_swap requires structured arguments."
                protocol = value("protocol").lower()
                if protocol == "v2_same_token":
                    amount = self.evm.quote_v2_same_token_flash(
                        int(structured["amount_out"])
                    )
                    return self._json({"minimum_repayment": amount})
                if protocol == "v2_cross_token":
                    amount = self.evm.quote_v2_cross_token_flash(
                        int(structured["amount_out"]),
                        int(structured["reserve_in"]),
                        int(structured["reserve_out"]),
                    )
                    return self._json({"minimum_repayment": amount})
                if protocol == "v3":
                    fee = self.evm.quote_v3_flash_fee(
                        int(structured["amount"]),
                        int(structured["fee_pips"]),
                    )
                    return self._json(
                        {
                            "fee": fee,
                            "total_owed": int(structured["amount"]) + fee,
                        }
                    )
                return "Unknown protocol. Use v2_same_token, v2_cross_token, or v3."

            case "evm_foundry_test_offline":
                self.authorization.require(
                    EVMCapability.ARBITRARY_HARNESS.value
                )
                if structured is None:
                    return "evm_foundry_test_offline requires structured arguments."
                if self.foundry is None:
                    return f"Foundry unavailable: {self.foundry_error}"
                project = self.authorization.resolve_task_path(
                    value("project", "evm_lab"),
                    write=True,
                )
                result = await self.foundry.test(
                    project,
                    fuzz_runs=int(structured.get("fuzz_runs", 256)),
                    invariant_runs=int(structured.get("invariant_runs", 64)),
                    timeout_seconds=float(structured.get("timeout_seconds", 300)),
                )
                return self._json(asdict(result))

            case "sandbox_execute_offline":
                if structured is None or not isinstance(structured.get("command"), list):
                    return "sandbox_execute_offline requires a command array."
                if self.sandbox is None:
                    return f"Sandbox unavailable: {self.sandbox_error}"
                command = tuple(str(item) for item in structured["command"])
                result = await self.sandbox.execute(
                    SandboxSpec(
                        command=command,
                        workspace=Path(value("workspace", "sandbox")),
                        working_directory=value("working_directory", "."),
                        environment={
                            str(name): str(item)
                            for name, item in structured.get("environment", {}).items()
                        },
                        limits=SandboxLimits(
                            timeout_seconds=float(structured.get("timeout_seconds", 120)),
                            cpu_seconds=int(structured.get("cpu_seconds", 60)),
                            memory_mb=int(structured.get("memory_mb", 1_024)),
                            max_output_bytes=int(
                                structured.get("max_output_bytes", 2_000_000)
                            ),
                            max_workspace_bytes=int(
                                structured.get(
                                    "max_workspace_bytes", 512 * 1024 * 1024
                                )
                            ),
                        ),
                    )
                )
                return self._json(asdict(result))

            case "ls" | "list_directory":
                return await self.ls(
                    value("path", ".")
                )

            case "tree" | "directory_tree":
                return await self.tree(
                    value("path", ".")
                )

            case "cat" | "read_file":
                return await self.cat(
                    value("path")
                )

            case "mkdir" | "create_directory":
                return await self.mkdir(
                    value("path")
                )

            case "info" | "get_file_info":
                return await self.info(
                    value("path")
                )

            case "search" | "search_files":

                if structured is not None:
                    return await self.search(
                        value("path", "."),
                        value("pattern"),
                    )

                try:
                    path, pattern = (
                        arguments.split(
                            ",",
                            1,
                        )
                    )

                except ValueError:
                    return (
                        "Invalid arguments for "
                        "search_files. Expected: "
                        "path,pattern"
                    )

                return await self.search(
                    path.strip(),
                    pattern.strip(),
                )

            case "write" | "write_file":

                if structured is not None:
                    return await self.write(
                        value("path"),
                        value("content"),
                    )

                try:
                    path, content = (
                        arguments.split(
                            "|",
                            1,
                        )
                    )

                except ValueError:
                    return (
                        "Invalid arguments for "
                        "write_file. Expected: "
                        "path|content"
                    )

                return await self.write(
                    path.strip(),
                    content,
                )

            case "edit" | "edit_file":

                if structured is not None:
                    edits = structured.get("edits", [])
                    if not isinstance(edits, list):
                        return "Invalid edits: expected a list."
                    return await self.edit(
                        value("path"),
                        edits,
                        bool(structured.get("dry_run", False)),
                    )

                return (
                    "edit_file requires structured "
                    "arguments and is not supported "
                    "through the legacy TOOL:name:string "
                    "format."
                )

            case "move" | "move_file":

                if structured is not None:
                    return await self.move(
                        value("source"),
                        value("destination"),
                    )

                try:
                    source, destination = (
                        arguments.split(
                            ",",
                            1,
                        )
                    )

                except ValueError:
                    return (
                        "Invalid arguments for "
                        "move_file. Expected: "
                        "source,destination"
                    )

                return await self.move(
                    source.strip(),
                    destination.strip(),
                )

            #
            # Browser
            #

            case "browser_navigate":
                return await self.browser_call(
                    tool,
                    {
                        "url": value("url"),
                    },
                )

            case "browser_snapshot":
                return await self.browser_call(
                    tool,
                    {},
                )

            case "browser_find":
                return await self.browser_call(
                    tool,
                    {
                        "text": value("text"),
                    },
                )

            case "browser_click":

                try:
                    element = value("element").strip()

                    if not element:
                        return (
                            "browser_click requires "
                            "an element identifier."
                        )

                    return await self.browser_call(
                        tool,
                        {
                            "element": element,
                        },
                    )

                except Exception as exc:

                    return (
                        "browser_click failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            case "browser_press_key":
                return await self.browser_call(
                    tool,
                    {
                        "key": value("key"),
                    },
                )

            case "browser_type":
                if structured is None:
                    return "browser_type requires structured JSON arguments."
                return await self.browser_call(tool, structured)

            #
            # Unknown tool
            #

            case _:

                return (
                    f"Unknown MCP tool: {tool}"
                )

    @staticmethod
    def _json(value: Any) -> str:
        def encode(item: Any) -> Any:
            if hasattr(item, "value"):
                return item.value
            if isinstance(item, Decimal):
                return str(item)
            if isinstance(item, Path):
                return str(item)
            raise TypeError(f"cannot JSON encode {type(item).__name__}")

        return json.dumps(value, default=encode, sort_keys=True)

    @staticmethod
    def _schema_from_example(value: Any) -> dict[str, Any]:
        """Derive a strict JSON schema from a concrete generated-tool test value."""

        if isinstance(value, dict):
            properties = {
                str(name): MCPTools._schema_from_example(item)
                for name, item in value.items()
            }
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        if isinstance(value, list):
            item_schema = MCPTools._merge_example_schemas(
                [MCPTools._schema_from_example(item) for item in value]
            ) if value else {"type": "null"}
            return {"type": "array", "items": item_schema}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if value is None:
            return {"type": "null"}
        return {"type": "string"}

    @staticmethod
    def _merge_example_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge same-kind examples, including heterogeneous object-list items."""

        if not schemas:
            return {"type": "null"}
        kinds = {str(schema.get("type", "")) for schema in schemas}
        if kinds <= {"integer", "number"}:
            return {"type": "number" if "number" in kinds else "integer"}
        if len(kinds) != 1:
            # The learning schema intentionally supports a compact JSON-Schema
            # subset without anyOf. Keep the first concrete kind; validation
            # will surface a mixed-type fixture rather than silently broadening.
            return schemas[0]
        kind = next(iter(kinds))
        if kind == "object":
            names = {
                name
                for schema in schemas
                for name in schema.get("properties", {})
            }
            properties: dict[str, Any] = {}
            for name in sorted(names):
                children = [
                    schema["properties"][name]
                    for schema in schemas
                    if name in schema.get("properties", {})
                ]
                properties[name] = MCPTools._merge_example_schemas(children)
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
                "items": MCPTools._merge_example_schemas(
                    [schema["items"] for schema in schemas]
                ),
            }
        return {"type": kind}

    @staticmethod
    def _decimal_or_none(value: Any) -> Decimal | None:
        return None if value is None else Decimal(str(value))
