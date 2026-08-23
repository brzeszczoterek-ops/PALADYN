from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .autonomy import AuthorizationEnvelope, AuthorizationGuard
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
    EvidenceOutcome,
    EvidenceSource,
    LearningEvidence,
    LearningRuntime,
    SkillManifest,
    ToolManifest,
)
from .sandbox import (
    BubblewrapBackend,
    SandboxExecutor,
    SandboxLimits,
    SandboxSpec,
    SandboxUnavailable,
)
from .tools.filesystem import Filesystem


class MCPTools:

    def __init__(
        self,
        config: Config,
    ):
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
        if getattr(config, "learning_profile", "client") == "owner_lab":
            persistent_learning = {
                "owner:create_persistent_artifacts",
                "owner:activate_persistent_artifacts",
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

        if not result.content:
            return ""

        text = []

        for item in result.content:

            if hasattr(
                item,
                "text",
            ):
                text.append(
                    item.text
                )

        return "\n".join(text)

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

    def local_tool_names(self) -> list[str]:
        names = [
            "evm_analyze_erc20_abi",
            "evm_validate_oracle",
            "evm_analyze_solidity_security",
            "sandbox_execute_offline",
            "learning_record_evidence",
            "learning_propose_lesson",
            "learning_stage_tool",
            "learning_stage_skill",
            "learning_create_tool",
            "learning_create_skill",
            "learning_validate_artifact",
            "learning_activate_artifact",
            "learning_retire_artifact",
            "learning_list_artifacts",
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

    def _known_tool_names(self) -> set[str]:
        return set(self.local_tool_names()) | {
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
                if structured is None or not isinstance(structured.get("manifest"), dict):
                    return "learning_create_tool requires manifest and source."
                record = await self.learning.create_tool(
                    ToolManifest.from_dict(structured["manifest"]),
                    value("source"),
                )
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
                return self._json(
                    self.learning.activate_artifact(value("artifact_id")).to_dict()
                )

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
    def _decimal_or_none(value: Any) -> Decimal | None:
        return None if value is None else Decimal(str(value))
