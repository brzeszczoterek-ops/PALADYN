// SPDX-License-Identifier: MIT
pragma solidity ^0.8.35;

import {PaladynEVMChecks} from "../src/PaladynEVMChecks.sol";

contract HookHandler {
    uint160 public flags;

    function setHook(address hook) external {
        flags = PaladynEVMChecks.hookFlags(hook);
    }
}

contract PaladynEVMChecksTest {
    HookHandler internal handler;

    function setUp() public {
        handler = new HookHandler();
    }

    function targetContracts() public view returns (address[] memory targets) {
        targets = new address[](1);
        targets[0] = address(handler);
    }

    function test_v2KnownRepayment() public pure {
        require(PaladynEVMChecks.v2SameTokenRepayment(1_000) == 1_004);
        require(PaladynEVMChecks.v2CrossTokenRepayment(100, 10_000, 5_000) == 205);
    }

    function test_v3KnownFee() public pure {
        require(PaladynEVMChecks.v3FlashFee(1_000_000, 3_000) == 3_000);
        require(PaladynEVMChecks.v3FlashFee(1, 500) == 1);
    }

    function test_oracleRejectsStaleRound() public pure {
        require(!PaladynEVMChecks.validOracleRound(2_500e8, 900, 10, 10, 1_000, 30));
        require(PaladynEVMChecks.validOracleRound(2_500e8, 990, 10, 10, 1_000, 30));
    }

    function test_v4OfficialHookExample() public pure {
        address hook = address(0x2400);
        require(PaladynEVMChecks.hookFlags(hook) == 0x2400);
    }

    function testFuzz_v3FeeRoundsUp(uint128 amount, uint24 feePips) public pure {
        if (feePips > 1_000_000) return;
        uint256 fee = PaladynEVMChecks.v3FlashFee(amount, feePips);
        require(fee * 1_000_000 >= uint256(amount) * uint256(feePips));
        if (fee > 0) {
            require((fee - 1) * 1_000_000 < uint256(amount) * uint256(feePips));
        }
    }

    function invariant_hookMaskNeverEscapesFourteenBits() public view {
        require(handler.flags() <= 0x3fff);
    }
}
