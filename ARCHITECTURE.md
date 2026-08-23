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

# Persona Architecture

V is implemented as separate layers so that speaking style cannot silently
replace judgment:

- `IdentityKernel` defines who V is and what she values.
- `Constitution` defines truth, user alignment, autonomy, risk judgment, and
  the narrow conditions in which V should object to a requested action.
- `VoiceProfile` defines V's direct, rebellious, contemporary, and naturally
  profane conversational register.
- `PersonaRuntime` combines those layers with the real relationship state and
  supplies few-shot anchors for smaller local models.

Every user-visible response must pass through the persona layer. This includes
normal chat, browser research, tool output, and error reporting. Untrusted tool
content remains data and cannot redefine V or her constitution.

## Relationship state

Relationship development is evidence-driven rather than a free-form roleplay
claim. The memory pipeline first reflects on an interaction, creates an
experience, and subjects it to the durable-memory confidence threshold. A
rejected experience cannot reach `RelationshipUpdater`.

For accepted experiences, the updater works on a copy of the current state,
clamps and confidence-scales numeric deltas, restricts shared-history events to
meaningful experience kinds, and accepts a form of address only when a reliable
`directly_told` preference contains that form in its evidence. The candidate is
atomically persisted before it replaces the in-memory state. Legacy flat YAML
loads remain supported; new writes use a versioned private state file.

`PersonaRuntime` converts the numeric state into `new`, `familiar`,
`established`, or `close` and includes the underlying scores and remembered
evidence. This makes the change visible to the model without permitting it to
perform intimacy unsupported by memory.

---

# Full Autonomous Control Plane

Full Autonomous is a runtime execution mode, not a system-prompt instruction.

Each task owns:

- an `AuthorizationEnvelope` describing pre-authorized capabilities;
- runtime, action, failure, disk, and external-cost budgets;
- an atomic JSON checkpoint;
- an append-only JSONL action journal;
- a control channel for `PAUSE`, `RESUME`, `STOP`, and `PANIC`;
- a task workspace separated from protected PALADYN state.

The autonomous runner checks control signals during active steps and cancels
in-flight work on `STOP` or `PANIC`. The LLM does not own this mechanism and
cannot grant itself new capabilities. The owner may change the envelope, while
worker models can only propose actions within it.

The emergency control plane also has a latched global PANIC. A separate Linux
input-event watcher detects a real simultaneous key chord and then:

1. writes the global panic latch consumed by every runner;
2. cancels all active task steps;
3. terminates registered PALADYN processes after validating both PID and Linux
   process start identity, preventing stale-PID termination.

The latch survives the stopped process and requires an explicit owner reset.

---

# Evidence-Driven Learning Plane

Learning is implemented outside the LLM as a state machine with six separate
objects: evidence, lesson, quarantined artifact, validation report, active
artifact, and retirement record. The separation prevents a model-generated
reflection from becoming executable authority.

Evidence and lifecycle events use append-only SHA-256 hash-chain journals.
Artifact records are checked against the journal, while manifests and source
are checked against their immutable bundle digest. Task artifacts carry a
workspace-derived scope key; persistent artifacts require a validated lesson
and double owner approval.

Generated Python tools run through a trusted host in the offline Bubblewrap
backend. Generated skills remain declarative workflows and are rendered below
the persona constitution only when their deterministic trigger tests match the
current request. Neither artifact type can modify its own validator, the
authorization model, persona, or emergency controls.

Autonomous runtime exceptions are captured as failure evidence. Completion is
captured as success evidence only when a verifier explicitly marks it verified.
Learning capture failure is journaled but never allowed to crash or falsely
complete the primary task.

---

# Local Model Loader

The interactive entry point resolves the model before constructing any LLM,
memory, or agent component. It discovers local files by extension and GGUF
header, ignores auxiliary multimodal projections and later split shards, then
loads a versioned per-path profile from private runtime state.

PALADYN launches `llama-server` as an argument array without a shell. The model
path, alias, loopback host, and port are controlled fields; remote model flags,
public binding, server-side tools, model presets, and API-key overrides cannot
be injected through profile extras. Inherited `LLAMA_ARG_*` variables are
removed. The loader requires successful `/health` and `/v1/models` responses
before applying the selected alias and endpoint to V's OpenAI-compatible client.
The child process owns a private log and is terminated as a process group during
normal shutdown, cancellation, failed startup, or initialization failure.

---

# EVM Capability Boundary

EVM support is divided by runtime profiles, not by persona promises:

| Capability | Client | Owner lab | Live chain |
|---|---:|---:|---:|
| ERC-20 interface analysis | yes | yes | read-only |
| Oracle validation | yes | yes | read-only |
| Security-wrapper lint | yes | yes | source only |
| Uniswap v4 hook simulation | no | yes | no |
| Flash-swap simulation | no | yes | no |
| State fork / arbitrary harness | no | pre-authorizable | local only |
| Wallet signing / broadcasting | no | no by default | separate grant |

Capabilities prefixed with `owner:` need to be present in both the task's normal
capability set and its owner-approved set. This prevents a worker model from
turning an ordinary capability request into an advanced EVM action.

Live-chain owner operations use two independent checks:

- the runtime envelope must contain an explicitly owner-approved live
  observation, simulation, signing, or broadcasting capability;
- the out-of-process bridge must receive a short-lived `LiveActionGrant` bound
  to chain ID, action type, targets, selectors, value, and confirmation.

Observation never implies signing. A grant contains policy only and never a
private key. The general agent process will not hold the signer.

The separate `paladyn-live` process currently implements only:

- pending-block and Geth txpool observation;
- transaction lookup;
- `eth_call` simulation and `eth_estimateGas`;
- exact chain-ID verification, response limits, and RPC error validation.

Its RPC method allowlist does not contain `eth_sendRawTransaction`, wallet,
personal, admin, or signing methods. A future signer will therefore be a new
process and capability, not an extension silently enabled inside this client.

Foundry compilation and testing use a dedicated `FoundrySandboxRunner`. `forge`
and `solc` are mounted read-only, network isolation remains enabled, compiler
autodetection/download is disabled, and an explicit solc binary is pinned.

# Sandbox Boundary

Generated and third-party code runs through an external Bubblewrap backend:

- new user, mount, PID, IPC, UTS, cgroup, and network namespaces;
- all Linux capabilities dropped and host environment cleared;
- read-only `/usr`, private `/proc`, `/dev`, `/tmp`, and `/home`;
- one writable task workspace and no visibility of PALADYN memory, `.env`, SSH,
  wallets, or the Docker socket;
- address-space, CPU, process-count, file-size, open-file, total-workspace,
  wall-clock, and output limits;
- process-group termination on timeout, output overflow, STOP, or PANIC.

The initial backend only accepts `OFFLINE`. Local-testnet, fetch-then-offline,
and allowlisted-proxy policies are declared but rejected until dedicated
enforcement backends are implemented. Bubblewrap is a process isolation layer,
not a VM boundary; untrusted native code should eventually use the planned
rootless-container or microVM backend.

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
