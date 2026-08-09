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
