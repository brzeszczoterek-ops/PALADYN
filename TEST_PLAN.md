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

# Learning, Generated Tools, and Skills

- [x] Unsupported success does not become learning evidence.
- [x] Model-authored JSON cannot mark its own evidence as verified.
- [x] User corrections retain framework task identity and the raw user message.
- [x] Autonomous failures and explicitly verified results are recorded.
- [x] Lessons require independent tasks, fingerprints, and verified evidence.
- [x] Evidence and artifact-journal tampering is detected.
- [x] Artifact source and manifest tampering is detected before activation.
- [x] Artifact versions are immutable within a scope.
- [x] Task artifacts are invisible outside their authorized workspace.
- [x] Persistent artifacts require a validated lesson and double owner approval.
- [x] Generated code with forbidden imports or calls is rejected.
- [x] Generated tools execute offline with time, memory, and output limits.
- [x] Sandbox process-count limits are enforced for generated code.
- [x] Generated manifests, schemas, source, metadata, and arguments are bounded.
- [x] Generated tools cannot exceed the total workspace disk limit.
- [x] Infinite generated code is terminated by the sandbox.
- [x] Input and output JSON schemas are enforced.
- [x] Failed tool tests prevent activation.
- [x] Active generated tools survive a runtime restart.
- [x] Three runtime failures automatically retire an active generated tool.
- [x] A regressing replacement rolls back to its previously active version.
- [x] Invalid caller input does not count as a generated-tool failure.
- [x] A new active version retires the previous version.
- [x] Skills require available tools and passing trigger cases.
- [x] Skills cannot override persona, policy, permissions, or kill switch.
- [x] Matching skills reach the agent prompt as lower-authority workflow data.
- [x] The owner audit verifies every journal, record, manifest, and source digest.
- [x] Parallel generated-tool runs update lifecycle counters transactionally.
- [x] Independent runtime instances serialize lifecycle updates without loss.

---

# Reliability

- [ ] No uncaught exceptions.
- [ ] No hallucinated filesystem data.
- [ ] No hallucinated tool results.
- [ ] No fabricated web search results.

---

# Local Model Loader

- [x] Recursive GGUF discovery rejects empty/fake files and auxiliary shards.
- [x] Model directory, profiles, binary, and last selection persist privately.
- [x] Profile parameters are range-checked and bounded.
- [x] Extra arguments cannot override the local loader boundary.
- [x] `llama-server` launches without a shell and only on loopback.
- [x] Inherited `LLAMA_ARG_*` overrides are removed from the child environment.
- [x] V is not initialized before both `/health` and `/v1/models` pass.
- [x] Early server failure reports the private log tail.
- [x] A managed server is terminated during shutdown and failed startup.
- [x] Noninteractive runs never hang waiting for input.

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
