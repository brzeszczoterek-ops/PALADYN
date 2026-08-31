from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v_core.autonomy import AuthorizationEnvelope


class EVMCapability(str, Enum):
    ERC20_ANALYZE = "evm.erc20.analyze"
    ORACLE_VALIDATE = "evm.oracle.validate"
    SECURITY_ANALYZE = "evm.security.analyze"


GENERAL_CAPABILITIES = frozenset(
    {
        EVMCapability.ERC20_ANALYZE.value,
        EVMCapability.ORACLE_VALIDATE.value,
        EVMCapability.SECURITY_ANALYZE.value,
    }
)


@dataclass(frozen=True, slots=True)
class EVMAccessProfile:
    name: str
    capabilities: frozenset[str]

    @classmethod
    def client(cls) -> "EVMAccessProfile":
        return cls("client", GENERAL_CAPABILITIES)

    def apply(self, envelope: AuthorizationEnvelope) -> AuthorizationEnvelope:
        envelope.capabilities.update(self.capabilities)
        envelope.owner_approved_capabilities.update(
            capability
            for capability in self.capabilities
            if capability.startswith("owner:")
        )
        return envelope
