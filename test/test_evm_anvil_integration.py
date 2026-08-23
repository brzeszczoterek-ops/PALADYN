from __future__ import annotations

import asyncio
from pathlib import Path
import socket

import httpx
import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.evm import (
    LiveAction,
    LiveActionGrant,
    LiveOperationGuard,
    PendingTransactionObserver,
    ReadOnlyRPCClient,
    RPCResponseError,
    TransactionSimulator,
)


ANVIL = Path.home() / ".foundry" / "bin" / "anvil"

pytestmark = pytest.mark.skipif(
    not ANVIL.is_file(),
    reason="local Foundry Anvil is not installed",
)


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.asyncio
async def test_real_anvil_observation_and_simulation_are_read_only(
    tmp_path: Path,
) -> None:
    port = _available_port()
    process = await asyncio.create_subprocess_exec(
        str(ANVIL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--silent",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    endpoint = f"http://127.0.0.1:{port}"
    target = "0x1111111111111111111111111111111111111111"
    client = httpx.AsyncClient(timeout=2, trust_env=False)
    rpc = ReadOnlyRPCClient(endpoint, 31_337, client=client)
    try:
        for _ in range(50):
            try:
                await rpc.verify_chain()
                break
            except (RPCResponseError, OSError):
                if process.returncode is not None:
                    stderr = await process.stderr.read()
                    pytest.fail(
                        "Anvil exited during startup: "
                        + stderr.decode("utf-8", errors="replace")
                    )
                await asyncio.sleep(0.02)
        else:
            pytest.fail("Anvil did not start")

        capabilities = {"owner:evm.live_observe", "owner:evm.live_simulate"}
        envelope = AuthorizationEnvelope(
            workspace=str(tmp_path),
            capabilities=set(capabilities),
            owner_approved_capabilities=set(capabilities),
        )
        grant = LiveActionGrant(
            chain_id=31_337,
            issued_at=1_000,
            expires_at=1_600,
            actions=frozenset({LiveAction.OBSERVE, LiveAction.SIMULATE}),
            allowed_targets=frozenset({target}),
        )
        guard = LiveOperationGuard(AuthorizationGuard(tmp_path, envelope), grant)

        pending = await PendingTransactionObserver(rpc, guard).pending_block(now=1_100)
        simulation = await TransactionSimulator(rpc, guard).simulate(
            {"to": target, "data": "0x", "value": "0x0"},
            now=1_100,
        )

        assert pending is not None
        assert isinstance(pending.get("transactions"), list)
        assert simulation["output"] == "0x"
        assert simulation["estimated_gas"] >= 21_000
    finally:
        await client.aclose()
        if process.returncode is None:
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=2)
