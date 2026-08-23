from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.policy import AuthorizationDenied
from v_core.evm import (
    EVMAccessProfile,
    EVMToolkit,
    FoundrySandboxRunner,
    HookPermissions,
    LiveGrantStore,
    LiveAction,
    LiveActionDenied,
    LiveActionGrant,
    LiveActionRequest,
    LiveOperationGuard,
    PendingTransactionObserver,
    ReadOnlyRPCClient,
    RPCPolicyError,
    TransactionSimulator,
    OraclePolicy,
    OracleRound,
    SequencerStatus,
    analyze_erc20_abi,
    analyze_solidity_security,
    decode_v4_hook_address,
    uniswap_v2_cross_token_repayment,
    uniswap_v2_same_token_repayment,
    uniswap_v3_flash_fee,
)
from v_core.mcp_tools import MCPTools


def _erc20_abi() -> list[dict[str, object]]:
    def function(
        name: str,
        inputs: list[str],
        outputs: list[str],
    ) -> dict[str, object]:
        return {
            "type": "function",
            "name": name,
            "inputs": [{"type": item} for item in inputs],
            "outputs": [{"type": item} for item in outputs],
        }

    def event(name: str, inputs: list[str]) -> dict[str, object]:
        return {
            "type": "event",
            "name": name,
            "inputs": [{"type": item} for item in inputs],
        }

    return [
        function("totalSupply", [], ["uint256"]),
        function("balanceOf", ["address"], ["uint256"]),
        function("transfer", ["address", "uint256"], ["bool"]),
        function("allowance", ["address", "address"], ["uint256"]),
        function("approve", ["address", "uint256"], ["bool"]),
        function(
            "transferFrom",
            ["address", "address", "uint256"],
            ["bool"],
        ),
        event("Transfer", ["address", "address", "uint256"]),
        event("Approval", ["address", "address", "uint256"]),
    ]


def test_erc20_abi_analysis_checks_exact_standard_interface() -> None:
    report = analyze_erc20_abi(_erc20_abi())
    broken = analyze_erc20_abi(_erc20_abi()[:-1])

    assert report.interface_conformant
    assert not broken.interface_conformant
    assert any(item.code == "erc20.missing_event.Approval" for item in broken.findings)


def test_oracle_validation_rejects_stale_and_l2_recovery_data() -> None:
    envelope = EVMAccessProfile.client().apply(AuthorizationEnvelope())
    toolkit = EVMToolkit(AuthorizationGuard(Path.cwd(), envelope))
    data = OracleRound(
        round_id=10,
        answer=2_500_00000000,
        started_at=800,
        updated_at=900,
        answered_in_round=10,
        decimals=8,
    )
    report = toolkit.validate_oracle(
        data,
        OraclePolicy(max_age_seconds=30, minimum_answer=Decimal("1")),
        now=1_000,
        sequencer=SequencerStatus(is_up=True, started_at=950),
    )

    assert report.value == Decimal("2500")
    assert not report.acceptable
    assert {item.code for item in report.findings} >= {
        "oracle.stale",
        "oracle.sequencer_grace_period",
    }


def test_security_lint_flags_high_signal_boundary_errors() -> None:
    findings = analyze_solidity_security(
        """
        contract Unsafe {
          function pull(IERC20 token, address from, uint amount) external {
            require(tx.origin == owner);
            token.transferFrom(from, address(this), amount);
          }
        }
        """
    )

    codes = {item.code for item in findings}
    assert "security.tx_origin" in codes
    assert "security.unsafe_erc20_call" in codes


def test_v4_hook_permission_bits_match_official_example() -> None:
    permissions = decode_v4_hook_address(
        "0x0000000000000000000000000000000000002400"
    )

    assert permissions == HookPermissions(
        before_initialize=True,
        after_add_liquidity=True,
    )
    assert permissions.mask == 0x2400


def test_flash_swap_math_is_integer_safe() -> None:
    assert uniswap_v2_same_token_repayment(1_000) == 1_004
    assert uniswap_v2_cross_token_repayment(100, 10_000, 5_000) == 205
    assert uniswap_v3_flash_fee(1_000_000, 3_000) == 3_000
    assert uniswap_v3_flash_fee(1, 500) == 1


def test_client_profile_cannot_use_owner_only_simulation(tmp_path: Path) -> None:
    client = EVMAccessProfile.client().apply(
        AuthorizationEnvelope(workspace=str(tmp_path))
    )
    toolkit = EVMToolkit(AuthorizationGuard(tmp_path, client))

    with pytest.raises(AuthorizationDenied):
        toolkit.quote_v2_same_token_flash(1_000)


def test_owner_profile_explicitly_approves_simulation_but_not_broadcast(
    tmp_path: Path,
) -> None:
    owner = EVMAccessProfile.owner_lab().apply(
        AuthorizationEnvelope(workspace=str(tmp_path))
    )
    toolkit = EVMToolkit(AuthorizationGuard(tmp_path, owner))

    assert toolkit.quote_v2_same_token_flash(1_000) == 1_004
    assert not owner.allows("owner:evm.live_broadcast")


def test_owner_prefixed_capability_requires_second_approval() -> None:
    capability = "owner:evm.flash_swap.simulate"
    envelope = AuthorizationEnvelope(capabilities={capability})

    assert not envelope.allows(capability)
    envelope.owner_approved_capabilities.add(capability)
    assert envelope.allows(capability)


@pytest.mark.asyncio
async def test_owner_evm_tools_are_callable_through_agent_tool_runtime(
    tmp_path: Path,
) -> None:
    tools = MCPTools(
        SimpleNamespace(
            filesystem_server=["/usr/bin/false"],
            browser_server=["/usr/bin/false"],
            workspace=tmp_path,
            evm_profile="owner_lab",
        )
    )

    result = json.loads(
        await tools.call(
            "evm_quote_flash_swap",
            {"protocol": "v3", "amount": 1_000_000, "fee_pips": 3_000},
        )
    )

    assert result == {"fee": 3_000, "total_owed": 1_003_000}
    assert "evm_decode_uniswap_v4_hook" in tools.local_tool_names()
    assert "evm_foundry_test_offline" in tools.local_tool_names()


def test_client_tool_catalog_hides_advanced_evm_tools(tmp_path: Path) -> None:
    tools = MCPTools(
        SimpleNamespace(
            filesystem_server=["/usr/bin/false"],
            browser_server=["/usr/bin/false"],
            workspace=tmp_path,
            evm_profile="client",
        )
    )

    assert "evm_analyze_erc20_abi" in tools.local_tool_names()
    assert "evm_quote_flash_swap" not in tools.local_tool_names()
    assert "evm_decode_uniswap_v4_hook" not in tools.local_tool_names()


def test_live_observation_and_state_change_have_separate_owner_grants(
    tmp_path: Path,
) -> None:
    target = "0x1111111111111111111111111111111111111111"
    selector = "0xa9059cbb"
    grant = LiveActionGrant(
        chain_id=1,
        issued_at=1_000,
        expires_at=1_600,
        actions=frozenset({LiveAction.OBSERVE, LiveAction.BROADCAST}),
        allowed_targets=frozenset({target}),
        allowed_selectors=frozenset({selector}),
        max_value_wei=0,
    )
    capabilities = {
        "owner:evm.live_observe",
        "owner:evm.live_broadcast",
    }
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path),
        capabilities=set(capabilities),
        owner_approved_capabilities=set(capabilities),
    )
    guard = LiveOperationGuard(AuthorizationGuard(tmp_path, envelope), grant)

    guard.authorize(
        LiveActionRequest(
            LiveAction.OBSERVE,
            chain_id=1,
            value_wei=10**24,
        ),
        now=1_100,
    )
    with pytest.raises(LiveActionDenied):
        guard.authorize(
            LiveActionRequest(
                LiveAction.BROADCAST,
                chain_id=1,
                to_address=target,
                selector=selector,
                owner_confirmed=False,
            ),
            now=1_100,
        )
    guard.authorize(
        LiveActionRequest(
            LiveAction.BROADCAST,
            chain_id=1,
            to_address=target,
            selector=selector,
            owner_confirmed=True,
        ),
        now=1_100,
    )


def test_live_owner_grant_is_short_lived() -> None:
    with pytest.raises(ValueError, match="lifetime"):
        LiveActionGrant(
            chain_id=1,
            issued_at=1_000,
            expires_at=2_000,
            actions=frozenset({LiveAction.OBSERVE}),
        )


def test_live_grant_store_is_private_and_removes_expired_grants(
    tmp_path: Path,
) -> None:
    store = LiveGrantStore(tmp_path / "grants")
    grant = LiveActionGrant(
        chain_id=1,
        issued_at=1_000,
        expires_at=1_100,
        actions=frozenset({LiveAction.OBSERVE}),
    )

    path = store.save(grant)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert store.load(grant.grant_id, now=1_050) == grant
    assert store.load(grant.grant_id, now=1_100) is None
    assert not path.exists()


@pytest.mark.asyncio
async def test_read_only_rpc_observes_and_simulates_without_send_methods(
    tmp_path: Path,
) -> None:
    target = "0x1111111111111111111111111111111111111111"
    selector = "0xa9059cbb"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        result = {
            "eth_chainId": "0x1",
            "eth_getBlockByNumber": {"number": None, "transactions": []},
            "eth_call": "0x01",
            "eth_estimateGas": "0x5208",
        }[payload["method"]]
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rpc = ReadOnlyRPCClient("http://rpc.test", 1, client=client)
    grant = LiveActionGrant(
        chain_id=1,
        issued_at=1_000,
        expires_at=1_600,
        actions=frozenset({LiveAction.OBSERVE, LiveAction.SIMULATE}),
        allowed_targets=frozenset({target}),
        allowed_selectors=frozenset({selector}),
        max_value_wei=0,
    )
    capabilities = {"owner:evm.live_observe", "owner:evm.live_simulate"}
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path),
        capabilities=set(capabilities),
        owner_approved_capabilities=set(capabilities),
    )
    guard = LiveOperationGuard(AuthorizationGuard(tmp_path, envelope), grant)

    block = await PendingTransactionObserver(rpc, guard).pending_block(now=1_100)
    result = await TransactionSimulator(rpc, guard).simulate(
        {"to": target, "data": selector, "value": "0x0"},
        now=1_100,
    )

    assert block == {"number": None, "transactions": []}
    assert result["output"] == "0x01"
    assert result["estimated_gas"] == 21_000
    with pytest.raises(RPCPolicyError):
        await rpc.call("eth_sendRawTransaction", ["0xdead"])
    await client.aclose()


@pytest.mark.asyncio
async def test_rpc_fails_closed_on_wrong_chain() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": "0x89"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rpc = ReadOnlyRPCClient("https://rpc.test", 1, client=client)
    with pytest.raises(RPCPolicyError, match="chain mismatch"):
        await rpc.verify_chain()
    await client.aclose()


@pytest.mark.asyncio
async def test_foundry_runner_forces_offline_sandbox_and_pinned_solc(
    tmp_path: Path,
) -> None:
    from v_core.sandbox import SandboxResult

    forge = tmp_path / "forge"
    solc = tmp_path / "solc"
    forge.write_bytes(b"forge")
    solc.write_bytes(b"solc")
    project = tmp_path / "project"
    project.mkdir()
    (project / "foundry.toml").write_text("[profile.default]\n", encoding="utf-8")

    class BackendStub:
        def __init__(self) -> None:
            self.spec = None

        async def run(self, spec):
            self.spec = spec
            return SandboxResult(spec.command, 0, "ok", "", 0.1, "stub")

    backend = BackendStub()
    runner = FoundrySandboxRunner(backend, forge, solc)
    result = await runner.test(project, fuzz_runs=10, invariant_runs=5)

    assert result.succeeded
    assert backend.spec.network.value == "offline"
    assert backend.spec.read_only_inputs == (forge.resolve(), solc.resolve())
    assert "--offline" in backend.spec.command
    assert backend.spec.command[
        backend.spec.command.index("--use") + 1
    ] == "/inputs/1-solc"
