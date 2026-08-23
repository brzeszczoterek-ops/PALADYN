# V-Core Changelog

## Unreleased

### Added
- Full Autonomous execution mode foundation
- Authorization envelopes with explicit capabilities and budgets
- Durable task checkpoints and append-only JSONL action journals
- External PAUSE, RESUME, STOP, and PANIC control channel
- Multi-step autonomous runner with retry and resume support
- Protected task-workspace path guard
- External Bubblewrap sandbox with offline namespaces and resource limits
- Client and owner-lab EVM capability profiles with double approval for
  `owner:` capabilities
- ERC-20 ABI conformance analyzer
- Oracle freshness, bounds, round, and L2 sequencer validator
- Solidity security-wrapper heuristic analyzer
- Uniswap v4 hook-permission decoder and v2/v3 flash-swap math tools
- Local EVM and sandbox tools exposed to the agent runtime
- Global latched PANIC across every autonomous task
- Linux input-event emergency chord (`Q+P+0` by default)
- PID plus process-start validation before terminating PALADYN runtimes
- Short-lived live-chain owner grants separated into observe, simulate, sign,
  and broadcast actions
- Private, expiring live-grant store and owner CLI commands
- Separate read-only `paladyn-live` RPC observer/simulator
- Chain-ID enforcement and JSON-RPC method allowlisting
- Offline Foundry runner with pinned read-only forge/solc binaries
- Dependency-free Solidity unit, fuzz, and invariant harness
- Real local Anvil integration test
- Evidence-gated relationship-state updates and qualitative persona stages
- Versioned, atomic relationship persistence with private permissions

### Fixed
- Browser MCP sessions now close explicitly during V-Core shutdown
- Rejected low-confidence experiences can no longer change V's relationship
- Relationship updates no longer mutate live state before persistence succeeds
- Preferred forms of address cannot be accepted from unsupported inferences
- Relationship evidence is delimited as untrusted data in model prompts
- Non-finite confidence and relationship values are rejected or normalized
- Real-Anvil readiness test now handles PALADYN's RPC error boundary reliably

## 1.0.0

First stable foundation for continued PALADYN development.

### Added
- V constitution separated from identity and speaking style
- Few-shot personality anchors for smaller local models
- Structured JSON tool actions
- Environment-based model and workspace configuration
- Automated runtime, memory, routing, and persona tests

### Fixed
- Duplicate session memory entries
- Memory tasks being lost during shutdown
- Low-confidence reflections polluting durable memory
- General questions being incorrectly routed to URL research
- Tool and research paths bypassing V's persona
- Tool output being injected with system-message authority

### Changed
- All user-visible results now pass through V's voice
- V is user-aligned without being blindly obedient
- Profanity is an expected but contextual part of V's informal voice
- Development and launch documentation now reflects the working runtime

## 0.7.3

### Added
- Conversation history
- Async Memory Engine
- MCP integration improvements

### Fixed
- Reflection JSON parsing
- ToolDispatcher stability
- Import issues
- Reasoning mode disabled

### Changed
- Agent architecture
- LLM interface
