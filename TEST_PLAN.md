# V-Core Test Plan

The purpose of this document is to verify that every release behaves correctly.

Every item should pass before creating a new release.

---

# Conversation

## Memory

- [ ] Remembers previous question.
- [ ] Remembers previous answer.
- [ ] Maintains context across multiple turns.
- [ ] Does not confuse unrelated conversations.

---

# Filesystem

- [ ] Shows directory tree using MCP.
- [ ] Reads files using MCP.
- [ ] Creates files.
- [ ] Edits files.
- [ ] Moves files.
- [ ] Searches project files.
- [ ] Never hallucinates project structure.

---

# Internet

- [ ] Uses web tools when information is unavailable.
- [ ] Does not invent search results.
- [ ] Distinguishes current knowledge from retrieved knowledge.

---

# Tool Usage

- [ ] Correctly decides when a tool is required.
- [ ] Does not call tools unnecessarily.
- [ ] Correctly parses tool arguments.
- [ ] Handles tool failures gracefully.
- [ ] Continues after recoverable errors.

---

# Memory Engine

## Reflection

- [ ] Detects successful tasks.
- [ ] Detects failed tasks.
- [ ] Produces useful lessons.

## Experience

- [ ] Removes duplicate lessons.
- [ ] Stores only valuable experiences.

## Knowledge

- [ ] Stores long-term principles.
- [ ] Rejects temporary observations.
- [ ] Avoids memory pollution.

## Relationship

- [x] Rejected experiences cannot modify relationship state.
- [x] Numeric deltas are finite, bounded, and confidence-scaled.
- [x] Routine interactions cannot fabricate shared history.
- [x] Forms of address require a reliable directly stated preference.
- [x] Legacy state loads and new state saves atomically with private permissions.
- [x] Relationship stage, metrics, history, and address reach the persona prompt.

---

# Planning

- [ ] Executes multi-step tasks.
- [ ] Continues unfinished work.
- [ ] Requests clarification only when necessary.

---

# Reliability

- [ ] No uncaught exceptions.
- [ ] No hallucinated filesystem data.
- [ ] No hallucinated tool results.
- [ ] No fabricated web search results.

---

# Performance

- [ ] LLM response time acceptable.
- [ ] MCP communication stable.
- [ ] Memory pipeline completes successfully.
- [ ] No unnecessary LLM calls.

---

# Sponsor Demonstration

The agent must successfully complete:

- [ ] Find information on the web.
- [ ] Search local project files.
- [ ] Explain previous answers.
- [ ] Maintain conversation context.
- [ ] Execute MCP tools.
- [ ] Modify project files.
- [ ] Perform multi-step tasks.
- [ ] Produce correct summaries.

---

# Release Checklist

Before releasing a version:

- [ ] All tests pass.
- [ ] No critical bugs.
- [ ] Roadmap updated.
- [ ] Changelog updated.
- [ ] Sponsor scenarios verified.
- [ ] Version tagged.

---

# EVM and Sandbox

- [x] Client profile cannot see or invoke owner-only Uniswap/flash tools.
- [x] Owner capability requires explicit owner approval in the envelope.
- [x] ERC-20 analyzer rejects missing or malformed required ABI entries.
- [x] Oracle validator rejects stale, invalid, or sequencer-unsafe rounds.
- [x] Hook permission bits match the canonical Uniswap v4 mapping.
- [x] Flash-swap arithmetic rounds repayment and fees upward.
- [x] Sandbox cannot see the host home directory.
- [x] Sandbox has no host network.
- [x] Sandbox kills timed-out and output-flooding processes.
- [x] Sandbox workspace cannot escape the task authorization envelope.
- [x] Emergency chord triggers only while all configured keys are held.
- [x] Global PANIC cancels every active autonomous runner.
- [x] Runtime termination records validate PID and process start identity.
- [x] Live observation cannot implicitly authorize signing or broadcasting.
- [x] Live owner grants expire within at most 15 minutes.
- [x] Foundry compilation/fuzz/invariant harness runs offline in Bubblewrap.
- [x] Real local Anvil pending-block observation and `eth_call` simulation.
- [x] `paladyn-live` works end to end with a stored owner grant.
- [ ] Rootless container backend with seccomp and cgroup-v2 quotas.
- [ ] MicroVM backend for hostile native binaries.
