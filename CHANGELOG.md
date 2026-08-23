# V-Core Changelog

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
