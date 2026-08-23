from __future__ import annotations


def uniswap_v2_same_token_repayment(amount_out: int) -> int:
    """Minimum same-token repayment for the canonical 0.30% v2 fee."""

    _positive(amount_out, "amount_out")
    return _ceil_div(amount_out * 1_000, 997)


def uniswap_v2_cross_token_repayment(
    amount_out: int,
    reserve_in: int,
    reserve_out: int,
) -> int:
    """Canonical v2 getAmountIn calculation for a flash-swap repayment."""

    _positive(amount_out, "amount_out")
    _positive(reserve_in, "reserve_in")
    _positive(reserve_out, "reserve_out")
    if amount_out >= reserve_out:
        raise ValueError("amount_out must be below reserve_out")
    numerator = reserve_in * amount_out * 1_000
    denominator = (reserve_out - amount_out) * 997
    return numerator // denominator + 1


def uniswap_v3_flash_fee(amount: int, fee_pips: int) -> int:
    """Fee owed for a v3 flash amount, rounded up like FullMath.mulDivRoundingUp."""

    if amount < 0:
        raise ValueError("amount cannot be negative")
    if not 0 <= fee_pips <= 1_000_000:
        raise ValueError("fee_pips must be between 0 and 1,000,000")
    return _ceil_div(amount * fee_pips, 1_000_000)


def _positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
