from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from uuid import uuid4

from v_core.autonomy import AuthorizationGuard

from .access import EVMCapability

MAX_LIVE_GRANT_SECONDS = 15 * 60
_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SELECTOR = re.compile(r"^0x[0-9a-fA-F]{8}$")


class LiveAction(str, Enum):
    OBSERVE = "observe"
    SIMULATE = "simulate"
    SIGN = "sign"
    BROADCAST = "broadcast"


class LiveActionDenied(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class LiveActionRequest:
    action: LiveAction
    chain_id: int
    to_address: str | None = None
    selector: str | None = None
    value_wei: int = 0
    owner_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.chain_id <= 0:
            raise ValueError("chain_id must be positive")
        if self.value_wei < 0:
            raise ValueError("value_wei cannot be negative")
        if self.to_address is not None and not _ADDRESS.fullmatch(self.to_address):
            raise ValueError("invalid EVM target address")
        if self.selector is not None and not _SELECTOR.fullmatch(self.selector):
            raise ValueError("invalid EVM function selector")


@dataclass(frozen=True, slots=True)
class LiveActionGrant:
    """Short-lived owner grant consumed by a future out-of-process RPC bridge.

    The grant carries no private key and cannot sign anything by itself. It is a
    policy object that a separate signer/broadcaster must verify immediately
    before an action.
    """

    chain_id: int
    issued_at: int
    expires_at: int
    actions: frozenset[LiveAction]
    allowed_targets: frozenset[str] = field(default_factory=frozenset)
    allowed_selectors: frozenset[str] = field(default_factory=frozenset)
    max_value_wei: int = 0
    grant_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if self.chain_id <= 0:
            raise ValueError("chain_id must be positive")
        lifetime = self.expires_at - self.issued_at
        if lifetime <= 0 or lifetime > MAX_LIVE_GRANT_SECONDS:
            raise ValueError(
                f"live grant lifetime must be 1..{MAX_LIVE_GRANT_SECONDS} seconds"
            )
        if not self.actions:
            raise ValueError("live grant must contain at least one action")
        if self.max_value_wei < 0:
            raise ValueError("max_value_wei cannot be negative")

        normalized_targets = frozenset(item.lower() for item in self.allowed_targets)
        normalized_selectors = frozenset(item.lower() for item in self.allowed_selectors)
        if any(not _ADDRESS.fullmatch(item) for item in normalized_targets):
            raise ValueError("grant contains an invalid EVM target address")
        if any(not _SELECTOR.fullmatch(item) for item in normalized_selectors):
            raise ValueError("grant contains an invalid function selector")
        object.__setattr__(self, "allowed_targets", normalized_targets)
        object.__setattr__(self, "allowed_selectors", normalized_selectors)

    def authorize(self, request: LiveActionRequest, *, now: int) -> None:
        if now < self.issued_at or now >= self.expires_at:
            raise LiveActionDenied("live owner grant is not currently valid")
        if request.chain_id != self.chain_id:
            raise LiveActionDenied("chain_id is outside the live owner grant")
        if request.action not in self.actions:
            raise LiveActionDenied("action is outside the live owner grant")
        if (
            request.action is not LiveAction.OBSERVE
            and request.value_wei > self.max_value_wei
        ):
            raise LiveActionDenied("transaction value exceeds the live owner grant")

        state_changing = request.action in {LiveAction.SIGN, LiveAction.BROADCAST}
        if state_changing:
            if not request.owner_confirmed:
                raise LiveActionDenied("state-changing live action needs owner confirmation")
            if request.to_address is None or request.selector is None:
                raise LiveActionDenied("state-changing live action needs target and selector")
            if not self.allowed_targets or request.to_address.lower() not in self.allowed_targets:
                raise LiveActionDenied("target is outside the live owner grant")
            if (
                not self.allowed_selectors
                or request.selector.lower() not in self.allowed_selectors
            ):
                raise LiveActionDenied("selector is outside the live owner grant")

        # Public observation and eth_call-style simulation may cover arbitrary
        # targets, but remain non-signing and non-broadcasting operations.
        if request.to_address is not None and self.allowed_targets:
            if request.to_address.lower() not in self.allowed_targets:
                raise LiveActionDenied("target is outside the live owner grant")
        if request.selector is not None and self.allowed_selectors:
            if request.selector.lower() not in self.allowed_selectors:
                raise LiveActionDenied("selector is outside the live owner grant")


@dataclass(frozen=True, slots=True)
class LiveOperationGuard:
    authorization: AuthorizationGuard
    grant: LiveActionGrant

    def authorize(self, request: LiveActionRequest, *, now: int) -> None:
        capability = {
            LiveAction.OBSERVE: EVMCapability.LIVE_OBSERVE,
            LiveAction.SIMULATE: EVMCapability.LIVE_SIMULATE,
            LiveAction.SIGN: EVMCapability.LIVE_SIGN,
            LiveAction.BROADCAST: EVMCapability.LIVE_BROADCAST,
        }[request.action]
        self.authorization.require(capability.value)
        self.grant.authorize(request, now=now)
