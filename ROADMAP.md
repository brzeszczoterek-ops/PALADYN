# V-Core Roadmap

## Current Goal

Deliver a stable autonomous AI agent capable of completing real-world tasks for sponsor demonstrations.

No new features unless they improve:
- stability
- reliability
- task completion
- memory quality
- tool usage

---

# 0.7.4 (Must Have)

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

## MCP

- [ ] Persistent MCP session
- [ ] Shared timeout configuration
- [ ] Stable Web Search
- [ ] Browser integration
- [ ] Better filesystem support
- [ ] Better tool error handling

---

# 0.7.5 (Should Have)

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

# 0.8.0 (Future)

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
