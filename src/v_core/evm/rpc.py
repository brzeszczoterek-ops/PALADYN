from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from .live import LiveAction, LiveActionRequest, LiveOperationGuard


READ_ONLY_RPC_METHODS = frozenset(
    {
        "eth_chainId",
        "eth_call",
        "eth_estimateGas",
        "eth_getBlockByNumber",
        "eth_getCode",
        "eth_getLogs",
        "eth_getTransactionByHash",
        "eth_getTransactionReceipt",
        "txpool_content",
    }
)
_TX_HASH = re.compile(r"^0x[0-9a-fA-F]{64}$")


class RPCPolicyError(PermissionError):
    pass


class RPCResponseError(RuntimeError):
    pass


@dataclass(slots=True)
class ReadOnlyRPCClient:
    endpoint: str
    expected_chain_id: int
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 15.0
    max_response_bytes: int = 10 * 1024 * 1024
    _request_id: int = field(default=0, init=False)
    _owns_client: bool = field(default=False, init=False)
    _chain_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("RPC endpoint must be an http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("RPC credentials must not be embedded in the URL")
        if self.expected_chain_id <= 0:
            raise ValueError("expected_chain_id must be positive")
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            )
            self._owns_client = True

    async def close(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    async def verify_chain(self) -> None:
        if self._chain_verified:
            return
        chain_hex = await self.call("eth_chainId", [], verify_chain=False)
        try:
            actual = int(str(chain_hex), 16)
        except ValueError as exc:
            raise RPCResponseError("RPC returned an invalid chain ID") from exc
        if actual != self.expected_chain_id:
            raise RPCPolicyError(
                f"RPC chain mismatch: expected {self.expected_chain_id}, got {actual}"
            )
        self._chain_verified = True

    async def call(
        self,
        method: str,
        params: list[Any],
        *,
        verify_chain: bool = True,
    ) -> Any:
        if method not in READ_ONLY_RPC_METHODS:
            raise RPCPolicyError(f"JSON-RPC method is not read-only: {method}")
        if verify_chain and method != "eth_chainId":
            await self.verify_chain()
        self._request_id += 1
        assert self.client is not None
        try:
            response = await self.client.post(
                self.endpoint,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": method,
                    "params": params,
                },
            )
        except httpx.HTTPError as exc:
            raise RPCResponseError("RPC transport request failed") from exc
        if response.is_error:
            raise RPCResponseError(f"RPC HTTP status {response.status_code}")
        if len(response.content) > self.max_response_bytes:
            raise RPCResponseError("RPC response exceeds configured size limit")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RPCResponseError("RPC returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            raise RPCResponseError("RPC returned an invalid envelope")
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                code = error.get("code", "unknown")
                message = error.get("message", "RPC error")
                raise RPCResponseError(f"RPC error {code}: {message}")
            raise RPCResponseError("RPC returned an unspecified error")
        if "result" not in payload:
            raise RPCResponseError("RPC response has no result")
        return payload["result"]


@dataclass(slots=True)
class PendingTransactionObserver:
    rpc: ReadOnlyRPCClient
    authorization: LiveOperationGuard

    async def pending_block(self, *, now: int) -> dict[str, Any] | None:
        self.authorization.authorize(
            LiveActionRequest(LiveAction.OBSERVE, self.rpc.expected_chain_id),
            now=now,
        )
        result = await self.rpc.call("eth_getBlockByNumber", ["pending", True])
        if result is not None and not isinstance(result, dict):
            raise RPCResponseError("pending block is not an object")
        return result

    async def geth_txpool(self, *, now: int) -> dict[str, Any]:
        self.authorization.authorize(
            LiveActionRequest(LiveAction.OBSERVE, self.rpc.expected_chain_id),
            now=now,
        )
        result = await self.rpc.call("txpool_content", [])
        if not isinstance(result, dict):
            raise RPCResponseError("txpool_content is not an object")
        return result

    async def transaction(self, tx_hash: str, *, now: int) -> dict[str, Any] | None:
        if not _TX_HASH.fullmatch(tx_hash):
            raise ValueError("invalid transaction hash")
        self.authorization.authorize(
            LiveActionRequest(LiveAction.OBSERVE, self.rpc.expected_chain_id),
            now=now,
        )
        result = await self.rpc.call("eth_getTransactionByHash", [tx_hash])
        if result is not None and not isinstance(result, dict):
            raise RPCResponseError("transaction response is not an object")
        return result


@dataclass(slots=True)
class TransactionSimulator:
    rpc: ReadOnlyRPCClient
    authorization: LiveOperationGuard

    async def simulate(
        self,
        transaction: dict[str, Any],
        *,
        now: int,
        block: str = "pending",
    ) -> dict[str, Any]:
        target = transaction.get("to")
        data = transaction.get("data", transaction.get("input", "0x"))
        selector = data[:10] if isinstance(data, str) and len(data) >= 10 else None
        value = transaction.get("value", "0x0")
        value_wei = int(value, 16) if isinstance(value, str) else int(value)
        request = LiveActionRequest(
            LiveAction.SIMULATE,
            self.rpc.expected_chain_id,
            to_address=target,
            selector=selector,
            value_wei=value_wei,
        )
        self.authorization.authorize(request, now=now)
        call_transaction = {
            key: transaction[key]
            for key in (
                "from",
                "to",
                "gas",
                "gasPrice",
                "value",
                "maxFeePerGas",
                "maxPriorityFeePerGas",
                "accessList",
                "type",
            )
            if key in transaction
        }
        call_transaction["data"] = data
        output = await self.rpc.call("eth_call", [call_transaction, block])
        gas = await self.rpc.call("eth_estimateGas", [call_transaction, block])
        return {
            "output": output,
            "estimated_gas": int(str(gas), 16),
            "block": block,
        }
