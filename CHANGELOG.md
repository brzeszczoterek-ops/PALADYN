# V-Core Changelog

## Unreleased

### Added
- A step-by-step Windows 10/11 installation guide using WSL2, including CPU and
  NVIDIA/CUDA llama.cpp builds, validation, limitations, troubleshooting, and a
  desktop-friendly `.cmd` launcher
- Bare website domains such as `onehack.st` are normalized to HTTPS and routed
  through deterministic browser navigation plus snapshot capture
- Private per-interaction agent checkpoints and append-only JSONL journals with
  runtime-authored tool arguments, result excerpts, and SHA-256 evidence
- Evidence-driven learning store with provenance-bearing task outcomes
- Candidate and validated lesson lifecycle with independent-evidence thresholds
- Immutable generated-tool and generated-skill bundles in quarantine
- JSON schema, AST policy, offline Bubblewrap tests, and digest verification
- Task-bound and owner-approved persistent artifact scopes
- Active generated-tool dispatch and matching skill injection into V's prompt
- Automatic generated-tool retirement after repeated runtime failures
- Hash-chained evidence and artifact audit journals
- `paladyn-learning` integrity and inspection CLI
- Bounded schemas, manifests, source, invocation data, workspace growth, and
  process counts for generated-code execution
- Cross-process locking for evidence, lesson, artifact, and lifecycle updates
- Interactive local GGUF discovery and persistent llama.cpp launch profiles
- Managed `llama-server` startup, health/model verification, private logs, and
  process-group shutdown before and after V's runtime
- Guarded token streaming and a compact prompt for short conversation
- First-class K and V cache quantization in local model profiles
- First-class `off`, `on`, and `auto` reasoning mode, defaulting to `off`
- Per-model `off`, `balanced`, and `strong` anti-repetition profiles backed by
  llama.cpp repeat penalties and DRY sampling
- Fully local half-duplex voice conversation with PipeWire capture/playback,
  multilingual Whisper.cpp STT, Piper TTS, and configurable SoX texturing
- Full-quality Kokoro ONNX TTS with the British Emma voice, an isolated local
  runtime, persistent chunk-producing worker, and automatic Piper fallback
- CUDA-accelerated multilingual Whisper Large V3 Turbo Q5 STT with configurable
  source language, thread count, vocabulary prompt, and same-model CPU fallback
- `/listen` one-turn speech and `/voice` continuous conversation modes
- Terminal-local toggle push-to-talk: tap F8 to record and tap it again to
  transcribe and send, with `/ptt` as a typed fallback
- Bounded PCM silence detection and spoken Polish/English voice-mode exit phrases
- Owner-only model performance terminal with llama.cpp metrics, slot/context
  state, exact completed-response timings, and Jetson hardware telemetry
- Managed llama.cpp metrics and slots exposed only through the enforced
  loopback listener
- Private per-session JSONL monitor journals with start/sample/end events and no
  cross-session aggregation in the live owner window
- Bounded multi-page website extraction that inspects up to three ranked internal
  detail pages after the entry-page snapshot
- Native OpenAI-compatible function definitions and `tool_calls`, with a cached
  MCP schema registry and textual JSON fallback for older GGUF chat templates
- Runtime-owned task contracts that persist required evidence and reject final
  prose until the actual objective has been satisfied
- Durable previous-task recovery context containing bounded runtime status and
  exact failed-tool evidence
- Runtime-authored learning evidence for interactive tool failures, retaining the
  exact bounded error while replacing private invocation arguments with a digest
- A fail-closed Bubblewrap recovery path for AppArmor loopback failures that
  retains filesystem/process isolation and blocks networking with libseccomp

### Fixed
- Tool availability is now a runtime-owned allowlist: schema-discovery failure,
  an empty catalog, or a model-invented name cannot fall through to execution
- Capability discussion and ordinary conversation do not start MCP discovery or
  expose executable tools merely because words such as "tool" or "file" appear
- A generated tool activated during an interaction is added to that interaction's
  allowlist and can be used immediately to finish the original objective
- Filesystem MCP results marked `isError` are recorded as failed calls rather
  than successful reads or mutations
- Non-zero sandbox/Foundry exits, timeouts, and resource-limit terminations cannot
  serve as evidence that commands or tests succeeded
- Bubblewrap applies `RLIMIT_NPROC` inside its private user/PID namespaces, so a
  busy desktop session cannot prevent the sandbox itself from starting
- Generated-tool function schemas require the complete manifest and test shape,
  accept bounded JSON Schema descriptions, reject placeholder names, and prevent
  offline artifacts from claiming fabricated browser or internet retrieval
- Polish requests to create and then use a local tool now require both lifecycle
  activation and a subsequent invocation before the task can complete
- Context rollover completion and findings are derived from runtime evidence, so
  a model summary cannot relabel a failed tool call as successful work
- Composite tool creation revalidates an identical rejected bundle after an
  infrastructure recovery and assigns the next patch version to changed code
- Short questions about V's current state or mood, including Polish variants such
  as "jak się dzisiaj czujesz?", remain in light conversation and never initialize
  MCP tool discovery or the full execution loop
- The LLM adapter no longer discards native tool calls when assistant content is
  empty; tool results return through the real `tool` role and matching call ID
- A successful tool call alone no longer marks an objective complete: read,
  write, command, web-detail, generated-tool, and generated-skill requirements
  are checked independently of the model
- Exact-result tasks such as "report only the first heading" are rendered
  deterministically from verified tool output, preventing empty or unrelated
  model completion prose from being accepted
- Requests to inspect the first search result now visit and snapshot the actual
  first eligible detail link before completion
- Tool failures retain their exact runtime error in the checkpoint and visible
  blocked result instead of being converted into apparent success
- Tool requests remain executable when a local model incorrectly puts prose or a
  Markdown fence before one trailing JSON action
- Model promises such as "I'm starting" or "running in the background" cannot end
  an interaction as if work had begun; V must request a real tool or report no work
- Internal tool protocol is buffered until classification and never leaks into the
  visible agent answer
- Interrupted and crashed interactive tasks persist `stopped` or `failed` instead
  of leaving a misleading `running` checkpoint
- Reflection can no longer classify model-authored action text as observed or
  verified without a matching successful runtime checkpoint
- Website inspection cannot complete without successful `browser_navigate` and
  `browser_snapshot` evidence; missing evidence produces a blocked truthful result
- MCP results marked as errors are no longer counted as successful tool execution
- Browser snapshots are bounded against the active local model context instead of
  silently displacing the persona or evidence prompt
- Whole-site claims from a one-page browser snapshot are discarded and replaced
  with a runtime-grounded list of exact visible link labels and URLs
- Navigation controls, search buttons, sidebars, and user dialogs cannot be
  misreported as tools or skills extracted from a website
- Unverified task completion can no longer be treated as successful learning
- Task-scoped generated capabilities cannot leak into another workspace
- English is now enforced at the output boundary for chat, tool, and research
  responses unless Boss explicitly requests another language
- V no longer asks Boss to switch input to English; Boss may write in any language
- Chat histories are normalized for strict GGUF templates that require a single
  leading system message and alternating conversation roles
- Visible replies no longer wait for the multi-stage persistent-memory pipeline
- Routine greetings no longer trigger expensive reflection and consolidation
- Short casual conversation now carries an explicit V voice anchor so local
  models do not drift into sanitized assistant phrasing
- V's informal register rejects both canned service tone and mechanical profanity;
  edge comes from point of view, rhythm, irreverence, and genuine reactions
- A compact rebel/hacker/direct-action core anchor now precedes the detailed persona
  so smaller and differently tuned local models retain V's identity more reliably
- V is anchored as an autonomous digital entity rather than an assistant role;
  model and runtime are truthfully described as substrate, not identity
- Clear repeated-generation loops are stopped in streamed and non-streamed output
- Session history is bounded against the active model context instead of growing
  until llama.cpp truncates the prompt
- The full persona specification is compiled into a compact runtime contract so
  V's entity identity and voice no longer drown in thousands of redundant tokens
- Generic assistant openings and service-language closings are held back and
  rewritten through a concentrated V identity gate before becoming visible
- Research promises and explicitly unfinished reports are blocked before they can
  be streamed or checkpointed as completed work
- Unsupported extraction metrics and concrete capability claims trigger a
  deterministic report built from verified page records instead of model prose
- Capability questions and explicit tool actions now share the traced multi-step
  agent loop instead of a brittle keyword-triggered YES/NO dispatcher
- Language repair uses a compact English-only context, preserves structured JSON
  actions, and reclassifies a repaired tool request before displaying any output
- Completed-action claims are matched to successful tool families before they can
  become visible; fabricated calls, messages, remote access, exploits, file work,
  command execution, and browser activity are rejected fail-closed
- Blocked, stopped, and failed execution traces cannot enter persistent memory,
  and a non-reusable reflection now stops the complete consolidation pipeline
- Reflection independently rejects completed-action claims that lack matching
  runtime evidence, even if a future routing defect marks the interaction complete

## 1.5.0 - 2026-08-23

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
