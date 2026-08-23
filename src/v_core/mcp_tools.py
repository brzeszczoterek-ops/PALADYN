from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

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
        except SandboxUnavailable as exc:
            self.sandbox = None
            self.sandbox_error = str(exc)
            self.foundry = None
            self.foundry_error = str(exc)

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
        return names

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

        def value(name: str, default: str = "") -> str:
            if structured is not None:
                item = structured.get(name, default)
                return str(item) if item is not None else default
            return arguments or default

        #
        # Filesystem aliases + native MCP names
        #

        match tool:

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
