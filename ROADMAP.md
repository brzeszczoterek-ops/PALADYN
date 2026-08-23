# V-Core Roadmap

## Current Goal

Build on the stable 1.5 foundation toward an autonomous agent capable of
completing and verifying real-world tasks.

No new features unless they improve:
- stability
- reliability
- task completion
- memory quality
- tool usage

---

# After 1.5 (Must Have)

## Completed foundation

- [x] Full Autonomous task state machine, checkpoints, budgets, and PANIC
- [x] External simultaneous-key emergency stop
- [x] Offline Bubblewrap execution backend
- [x] Client/owner EVM capability boundary
- [x] ERC-20, oracle, security, Uniswap hook, and flash-swap analysis tools
- [x] Offline Foundry unit/fuzz/invariant lab
- [x] Read-only live RPC observer and transaction simulator
- [x] Evidence-gated persistent relationship state and persona-stage rendering
- [x] Evidence-driven lesson validation and immutable artifact registry
- [x] Offline generated-tool validation, activation, and automatic rollback
- [x] Declarative generated skills with tested triggers and protected boundaries
- [x] Interactive local GGUF discovery, llama.cpp profiles, and managed startup

## Agent

- [ ] Improve ToolDispatcher
- [ ] JSON tool calls
- [ ] Better tool routing
- [ ] Retry failed tools

## Reliability

- [ ] Never hallucinate filesystem contents
- [ ] Never hallucinate web search results
- [ ] Always prefer tools over model knowledge
- [ ] Validate tool output before storing memories

## Memory

- [ ] Better Reflection prompts
- [ ] Experience ranking
- [ ] Memory consolidation
- [ ] Experience deduplication
- [ ] Store only useful lessons
- [ ] Knowledge retrieval
- [ ] Shared JSON parser for all memory modules
- [x] Verified task outcomes feed the learning plane

## MCP

- [ ] Persistent MCP session
- [ ] Shared timeout configuration
- [ ] Stable Web Search
- [ ] Browser integration
- [ ] Better filesystem support
- [ ] Better tool error handling

---

# Later (Should Have)

## Planning

- [ ] Multi-step task execution
- [ ] Task decomposition
- [ ] Self verification
- [ ] Automatic retry on failure

## Web

- [ ] Website navigation
- [ ] Better scraping
- [ ] Search result ranking
- [ ] Session persistence

## Memory

- [ ] Memory confidence scoring
- [ ] Memory source validation
- [ ] Long conversation summarization

---

# Future

## Intelligence

- [ ] Intent Engine
- [ ] Autonomous planning
- [ ] Plugin system
- [ ] Long-term memory optimization
- [ ] Dynamic tool selection
- [ ] Self-improvement loop

---

# Long-Term Vision

## V-Core

The goal is not to build another chatbot.

The goal is to build a reliable local autonomous AI agent framework capable of solving real-world tasks using tools, memory and reasoning.

Future specializations (OSINT, blockchain, crypto analysis, web automation, research, etc.) should be implemented as independent modules without modifying the V-Core architecture.
