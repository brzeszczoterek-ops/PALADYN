from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v_core.autonomy import AuthorizationEnvelope


class EVMCapability(str, Enum):
    ERC20_ANALYZE = "evm.erc20.analyze"
    ORACLE_VALIDATE = "evm.oracle.validate"
    SECURITY_ANALYZE = "evm.security.analyze"
    UNISWAP_HOOKS_SIMULATE = "owner:evm.uniswap_hooks.simulate"
    FLASH_SWAP_SIMULATE = "owner:evm.flash_swap.simulate"
    STATE_FORK = "owner:evm.state_fork"
    ARBITRARY_HARNESS = "owner:evm.arbitrary_harness"
    LIVE_OBSERVE = "owner:evm.live_observe"
    LIVE_SIMULATE = "owner:evm.live_simulate"
    LIVE_SIGN = "owner:evm.live_sign"
    LIVE_BROADCAST = "owner:evm.live_broadcast"


GENERAL_CAPABILITIES = frozenset(
    {
        EVMCapability.ERC20_ANALYZE.value,
        EVMCapability.ORACLE_VALIDATE.value,
        EVMCapability.SECURITY_ANALYZE.value,
    }
)
ADVANCED_SIMULATION_CAPABILITIES = frozenset(
    {
        EVMCapability.UNISWAP_HOOKS_SIMULATE.value,
        EVMCapability.FLASH_SWAP_SIMULATE.value,
    }
)
OWNER_LAB_CAPABILITIES = frozenset(
    {
        *ADVANCED_SIMULATION_CAPABILITIES,
        EVMCapability.STATE_FORK.value,
        EVMCapability.ARBITRARY_HARNESS.value,
    }
)


@dataclass(frozen=True, slots=True)
class EVMAccessProfile:
    name: str
    capabilities: frozenset[str]

    @classmethod
    def client(cls) -> "EVMAccessProfile":
        return cls("client", GENERAL_CAPABILITIES)

    @classmethod
    def owner_lab(cls) -> "EVMAccessProfile":
        # Signing and broadcasting are intentionally absent. They require a
        # separate, explicit session grant and never become model defaults.
        return cls("owner_lab", GENERAL_CAPABILITIES | OWNER_LAB_CAPABILITIES)

    def apply(self, envelope: AuthorizationEnvelope) -> AuthorizationEnvelope:
        envelope.capabilities.update(self.capabilities)
        envelope.owner_approved_capabilities.update(
            capability
            for capability in self.capabilities
            if capability.startswith("owner:")
        )
        return envelope
