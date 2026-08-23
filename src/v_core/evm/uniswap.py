from __future__ import annotations

from dataclasses import dataclass, fields


_HOOK_FLAGS = {
    "before_initialize": 1 << 13,
    "after_initialize": 1 << 12,
    "before_add_liquidity": 1 << 11,
    "after_add_liquidity": 1 << 10,
    "before_remove_liquidity": 1 << 9,
    "after_remove_liquidity": 1 << 8,
    "before_swap": 1 << 7,
    "after_swap": 1 << 6,
    "before_donate": 1 << 5,
    "after_donate": 1 << 4,
    "before_swap_returns_delta": 1 << 3,
    "after_swap_returns_delta": 1 << 2,
    "after_add_liquidity_returns_delta": 1 << 1,
    "after_remove_liquidity_returns_delta": 1,
}


@dataclass(frozen=True, slots=True)
class HookPermissions:
    before_initialize: bool = False
    after_initialize: bool = False
    before_add_liquidity: bool = False
    after_add_liquidity: bool = False
    before_remove_liquidity: bool = False
    after_remove_liquidity: bool = False
    before_swap: bool = False
    after_swap: bool = False
    before_donate: bool = False
    after_donate: bool = False
    before_swap_returns_delta: bool = False
    after_swap_returns_delta: bool = False
    after_add_liquidity_returns_delta: bool = False
    after_remove_liquidity_returns_delta: bool = False

    @property
    def mask(self) -> int:
        return sum(
            _HOOK_FLAGS[item.name]
            for item in fields(self)
            if getattr(self, item.name)
        )


def decode_v4_hook_address(address: str) -> HookPermissions:
    value = _parse_address(address)
    return HookPermissions(
        **{name: bool(value & flag) for name, flag in _HOOK_FLAGS.items()}
    )


def validate_v4_hook_address(
    address: str,
    expected: HookPermissions,
) -> bool:
    return decode_v4_hook_address(address) == expected


def _parse_address(address: str) -> int:
    raw = address[2:] if address.startswith("0x") else address
    if len(raw) != 40:
        raise ValueError("EVM address must contain exactly 20 bytes")
    try:
        return int(raw, 16)
    except ValueError as exc:
        raise ValueError("invalid hexadecimal EVM address") from exc
