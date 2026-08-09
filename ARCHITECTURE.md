# V-Core Architecture

## Overview

V-Core is a local autonomous AI agent framework.

Its purpose is not to be another chatbot.

Its purpose is to execute real-world tasks by combining:

- LLM
- Memory
- Tools
- Planning
- MCP

The LLM is only one component of the system.

The intelligence comes from the cooperation of all modules.

---

# High-Level Architecture

```
                User
                  │
                  ▼
              Agent Core
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
    LLM        Memory      Tool Dispatcher
                  │              │
                  ▼              ▼
           Memory Engine      MCP Tools
                  │              │
        ┌─────────┼───────┐      ▼
        ▼         ▼       ▼   MCP Client
   Reflection Experience Knowledge │
                  │                ▼
                  └────────── Filesystem
```

---

# Responsibilities

## Agent

Coordinates the entire system.

The Agent never performs work directly.

It decides what should happen next.

---

## LLM

Responsible only for language generation and reasoning.

It should never replace tools.

It should never fabricate external information.

---

## Session

Stores short-term conversation history.

Used to preserve conversational context.

---

## Memory Engine

Coordinates long-term learning.

Pipeline:

Task
↓

Reflection
↓

Experience
↓

Knowledge

---

## Reflection

Evaluates completed work.

Finds mistakes.

Extracts lessons.

---

## Experience

Determines whether a lesson is useful.

Ranks importance.

Removes repetition.

---

## Knowledge

Stores durable long-term knowledge.

Knowledge should represent principles, not events.

---

## Tool Dispatcher

Decides whether a tool should be used.

Routes requests to MCP.

Never executes tools directly.

---

## MCP Tools

High-level wrapper around available tools.

Provides a stable API for the Agent.

---

## MCP Client

Low-level communication layer.

Responsible only for communication with MCP servers.

---

# Design Principles

- Single responsibility.
- Tools before guessing.
- Memory before repetition.
- Modular architecture.
- Local-first.
- LLM is a reasoning engine, not a database.
- Every module should be independently replaceable.

---

# Future

The architecture is designed to support multiple agents.

Example:

agents/

    V/
    Atlas/
    Nova/
    Ghost/

Each agent will define:

- personality
- behavior
- constitution
- prompts
- memory profile

without changing the V-Core engine.
