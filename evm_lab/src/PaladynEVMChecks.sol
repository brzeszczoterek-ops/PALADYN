// SPDX-License-Identifier: MIT
pragma solidity ^0.8.35;

library PaladynEVMChecks {
    uint256 internal constant V2_FEE_DENOMINATOR = 1_000;
    uint256 internal constant V2_FEE_MULTIPLIER = 997;
    uint160 internal constant V4_ALL_HOOK_MASK = (1 << 14) - 1;

    function v2SameTokenRepayment(uint256 amountOut) internal pure returns (uint256) {
        require(amountOut > 0, "amount=0");
        return ceilDiv(amountOut * V2_FEE_DENOMINATOR, V2_FEE_MULTIPLIER);
    }

    function v2CrossTokenRepayment(
        uint256 amountOut,
        uint256 reserveIn,
        uint256 reserveOut
    ) internal pure returns (uint256) {
        require(amountOut > 0 && reserveIn > 0 && amountOut < reserveOut, "bad reserves");
        uint256 numerator = reserveIn * amountOut * V2_FEE_DENOMINATOR;
        uint256 denominator = (reserveOut - amountOut) * V2_FEE_MULTIPLIER;
        return numerator / denominator + 1;
    }

    function v3FlashFee(uint256 amount, uint24 feePips) internal pure returns (uint256) {
        require(feePips <= 1_000_000, "bad fee");
        return ceilDiv(amount * uint256(feePips), 1_000_000);
    }

    function validOracleRound(
        int256 answer,
        uint256 updatedAt,
        uint80 roundId,
        uint80 answeredInRound,
        uint256 currentTime,
        uint256 maxAge
    ) internal pure returns (bool) {
        if (answer <= 0 || updatedAt == 0 || updatedAt > currentTime) return false;
        if (answeredInRound < roundId) return false;
        return currentTime - updatedAt <= maxAge;
    }

    function hookFlags(address hook) internal pure returns (uint160) {
        return uint160(hook) & V4_ALL_HOOK_MASK;
    }

    function ceilDiv(uint256 numerator, uint256 denominator) internal pure returns (uint256) {
        require(denominator != 0, "division by zero");
        if (numerator == 0) return 0;
        return (numerator - 1) / denominator + 1;
    }
}
