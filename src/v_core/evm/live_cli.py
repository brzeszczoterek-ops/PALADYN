from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Awaitable, Callable

import httpx
import typer
from rich.console import Console

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard

from .access import EVMCapability
from .grants import LiveGrantStore
from .live import LiveAction, LiveActionDenied, LiveOperationGuard
from .rpc import (
    PendingTransactionObserver,
    ReadOnlyRPCClient,
    RPCPolicyError,
    RPCResponseError,
    TransactionSimulator,
)


app = typer.Typer(add_completion=False)
console = Console(stderr=True)


def default_root() -> Path:
    return Path(os.getenv("PALADYN_AUTONOMY_ROOT", "autonomy")).expanduser()


def _operation_guard(root: Path, grant_id: str, *, now: int) -> LiveOperationGuard:
    grant = LiveGrantStore(root / "live_grants").load(grant_id, now=now)
    if grant is None:
        raise LiveActionDenied("live grant is missing, revoked, or expired")
    capability_by_action = {
        LiveAction.OBSERVE: EVMCapability.LIVE_OBSERVE.value,
        LiveAction.SIMULATE: EVMCapability.LIVE_SIMULATE.value,
        LiveAction.SIGN: EVMCapability.LIVE_SIGN.value,
        LiveAction.BROADCAST: EVMCapability.LIVE_BROADCAST.value,
    }
    capabilities = {
        capability_by_action[action]
        for action in grant.actions
    }
    envelope = AuthorizationEnvelope(
        workspace=str(Path.cwd()),
        capabilities=capabilities,
        owner_approved_capabilities=set(capabilities),
    )
    return LiveOperationGuard(AuthorizationGuard(Path.cwd(), envelope), grant)


def _guard_or_exit(root: Path, grant_id: str, *, now: int) -> LiveOperationGuard:
    try:
        return _operation_guard(root, grant_id, now=now)
    except (LiveActionDenied, ValueError, OSError) as exc:
        console.print(f"Cannot load live owner grant: {exc}")
        raise typer.Exit(code=1) from exc


async def _with_rpc(
    endpoint: str,
    chain_id: int,
    operation: Callable[[ReadOnlyRPCClient], Awaitable[Any]],
) -> Any:
    rpc = ReadOnlyRPCClient(endpoint, chain_id)
    try:
        return await operation(rpc)
    finally:
        await rpc.close()


def _run(operation: Awaitable[Any]) -> None:
    try:
        result = asyncio.run(operation)
    except (
        LiveActionDenied,
        RPCPolicyError,
        RPCResponseError,
        httpx.HTTPError,
        ValueError,
    ) as exc:
        console.print(f"Live operation denied/failed: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("pending-block")
def pending_block(
    endpoint: str = typer.Option(..., help="HTTP(S) EVM JSON-RPC endpoint."),
    chain_id: int = typer.Option(..., help="Expected chain ID."),
    grant_id: str = typer.Option(..., help="Active owner grant ID."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    now = int(time.time())
    guard = _guard_or_exit(root, grant_id, now=now)

    async def operation(rpc: ReadOnlyRPCClient) -> Any:
        return await PendingTransactionObserver(rpc, guard).pending_block(now=now)

    _run(_with_rpc(endpoint, chain_id, operation))


@app.command("txpool")
def txpool(
    endpoint: str = typer.Option(..., help="Geth-compatible HTTP(S) RPC endpoint."),
    chain_id: int = typer.Option(..., help="Expected chain ID."),
    grant_id: str = typer.Option(..., help="Active owner grant ID."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    now = int(time.time())
    guard = _guard_or_exit(root, grant_id, now=now)

    async def operation(rpc: ReadOnlyRPCClient) -> Any:
        return await PendingTransactionObserver(rpc, guard).geth_txpool(now=now)

    _run(_with_rpc(endpoint, chain_id, operation))


@app.command("transaction")
def transaction(
    tx_hash: str,
    endpoint: str = typer.Option(..., help="HTTP(S) EVM JSON-RPC endpoint."),
    chain_id: int = typer.Option(..., help="Expected chain ID."),
    grant_id: str = typer.Option(..., help="Active owner grant ID."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    now = int(time.time())
    guard = _guard_or_exit(root, grant_id, now=now)

    async def operation(rpc: ReadOnlyRPCClient) -> Any:
        return await PendingTransactionObserver(rpc, guard).transaction(
            tx_hash,
            now=now,
        )

    _run(_with_rpc(endpoint, chain_id, operation))


@app.command("simulate")
def simulate(
    tx_json: Path = typer.Option(..., help="JSON file containing transaction fields."),
    endpoint: str = typer.Option(..., help="HTTP(S) EVM JSON-RPC endpoint."),
    chain_id: int = typer.Option(..., help="Expected chain ID."),
    grant_id: str = typer.Option(..., help="Active owner grant ID."),
    block: str = typer.Option("pending", help="Block tag or hexadecimal block number."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    try:
        transaction_data = json.loads(tx_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"Cannot read transaction JSON: {exc}")
        raise typer.Exit(code=1) from exc
    if not isinstance(transaction_data, dict):
        console.print("Transaction JSON must be an object.")
        raise typer.Exit(code=1)

    now = int(time.time())
    guard = _guard_or_exit(root, grant_id, now=now)

    async def operation(rpc: ReadOnlyRPCClient) -> Any:
        return await TransactionSimulator(rpc, guard).simulate(
            transaction_data,
            now=now,
            block=block,
        )

    _run(_with_rpc(endpoint, chain_id, operation))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
