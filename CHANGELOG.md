# V-Core Changelog

## 3.0.0 - 2026-08-31

### Added
- Local GGUF qualification cards covering instruction following, structured
  output, tool use and abstention, coding, research routing, conversation, and
  V's structural voice contract, plus grounded unknowns and execution honesty
- Deterministic one-model-at-a-time routing across at most three qualified local
  models, with profile invalidation, verified fallbacks, a shared reconfigurable
  LLM client, and private prompt-digest journals
- The `paladyn-model` operator CLI for qualification, pool configuration,
  routing control, model inventory, and dry-run route inspection
- A source-owned generated-tool lifecycle: the LLM now emits Python only, while
  PALADYN derives the manifest, strict schemas, owner-oracle or deterministic
  smoke tests, activation, post-activation invocation, and verified report
- Model-profile chat-template selection with automatic Hermes 3 detection and a
  vendored, reviewed `tool_use` template, keeping startup fully offline while
  allowing llama.cpp to emit native function calls for that model family
- A domain-independent structured-fixture protocol for generated tools: PALADYN
  binds exact JSON inputs and expected results from the owner request, validates
  on the first fixture, and executes an activated tool on the last fixture
- High-level `web_search` and `web_read` tools: PALADYN now turns a focused
  DuckDuckGo query into grounded result URLs, opens an exact observed result,
  and rejects detail addresses invented by the model
- Model-level V identity primer loaded by the local llama-server at process
  startup through a private `--system-prompt-file`; the same short anchor is
  applied before PALADYN's detailed persona across compatible GGUF templates
- Live, per-task browser audit in the owner monitor, showing every requested
  URL, redirect, tool outcome, and actual HTTP status without mixing in visits
  from older tasks
- Read-only self-review of PALADYN's prior task logs through
  `runtime_review_task`, with bounded runtime-grounded findings that cite exact
  tool-call and context-rollover numbers

### Fixed
- Generated tools that ignore concrete owner fixture fields and hard-code the
  demonstrated answer are rejected before staging; explicit offline/no-web
  constraints also survive semantic routing and remove browser capabilities
- Source generation receives no callable tools, accepts a narrowly parsed legacy
  wrapper from models trained to emit function envelopes, and PALADYN—not the
  model—binds and executes the newly activated artifact
- Generated-artifact phases now force the required lifecycle builder at the
  provider boundary, reject calls to a not-yet-activated artifact, and recover
  schema-shaped bare builder payloads without adding task-, language-, or
  tool-name-specific patches
- Hermes-style textual function-call envelopes accept only the reviewed call
  metadata fields and cannot reinterpret an artifact manifest as an executable
  tool call
- Generated-tool repair rollovers now retain the failed candidate source, exact
  expected result, and validator error instead of dropping all useful evidence
  behind a duplicated large fixture
- Offline fixture URLs no longer create a false browser contract, explicit
  requests to execute the newly created tool require a separate runtime success,
  and malformed optional artifact versions fall back to `1.0.0`
- Dictated HTTP(S) addresses are reconstructed from spoken punctuation and
  treated as direct owner-supplied targets; a failed direct URL is attempted
  once and reported truthfully instead of being replaced with an unrelated
  search result (such as a Wikipedia page about “dwukropek”)
- After tool evidence is complete, two malformed grounded-answer drafts now
  trigger a deterministic evidence report instead of another rewrite/context
  rollover loop
- Light conversation no longer inherits the repeated “Still awake, still
  dangerous” catchphrase or closes with “What's the plan, Boss?”; deliberately
  mangled input now gets V's short sarcastic clarification instead of the
  sterile “You're speaking in code” response
- Push-to-talk now defaults to `F2` in both the runtime and desktop launcher;
  terminal bindings accept the common SS3 and CSI `F2` escape sequences
- Empty action acknowledgements such as “I know exactly what you want. Let's do
  it.” can no longer complete a task without execution; PALADYN rejects the
  empty enthusiasm and keeps the runtime loop open for a real tool call
- Runtime-owned failure, evidence, loop, and step-boundary messages keep their
  exact operational truth while speaking in V's direct voice instead of falling
  back to dry helpdesk prose
- Semantically mangled speech and word salad now stop at the intent boundary;
  V answers with a short sarcastic clarification instead of generating unrelated
  persona filler or pretending that the fragment was understood
- Grammatically parseable but absurd non sequiturs are distinguished from ordinary
  conversation and receive the same bounded V-style reaction; bare "Want me to
  do something?" closings are now rejected as helpdesk voice
- Repetitive chants and phonetic loops are rejected by a fast runtime-owned
  language-neutral detector before any LLM call, avoiding a slow full-persona
  generation for obvious banter
- Short non-action statements now use a compact current-message-only V prompt;
  previous task checkpoints, execution memory, agent instructions, and tool
  schemas can no longer hijack casual banter or mangled speech
- When a local intent parser copies an unrelated previous subject into a new
  utterance, PALADYN now asks Boss to repeat it instead of letting the main model
  fake understanding; routine context rollovers are also skipped when they save
  only a negligible number of tokens
- A non-continuation intent must ground its subject and search query in the
  current user message, preventing previous-task context from turning an
  unrelated utterance into stale browser work
- Public-business tasks now keep tool execution open until observed sources
  contain every requested address, contact, or opening-hours field; a namesake
  product page can no longer satisfy a location-information contract by itself
- Public-fact commands now use language-independent normalized intent fields;
  when a local classifier contradicts its own browser/report/query structure,
  PALADYN repairs the structural contradiction rather than matching one language
- Public-fact evidence is bound to the router's exact named subject, preventing
  an address or opening time for a similarly named business from completing the
  wrong task
- Failed persona rewrites no longer erase a substantive answer with a voice-gate
  status message
- Evidence-finalization replies use a compact 256-token budget, cutting the long
  wait before voice validation without reducing tool-generation budgets
- URLs copied imperfectly by a local model are now mapped back only when they
  match exactly one previously observed search result after separator cleanup;
  PALADYN opens that grounded URL instead of repeating the same web search
- Background learning no longer sends every historical experience and knowledge
  entry to the model: each memory stage now uses a bounded newest-first evidence
  slice, preventing accumulated memory from overflowing the model context
- Concrete public-fact lookups requesting multiple fields such as location
  count, opening hours, and addresses recover a browser execution contract when
  a local model incorrectly classifies the command as ordinary conversation
- Exiting PALADYN now terminates the complete local session, including an exact
  matching llama-server adopted from an interrupted earlier run, releasing VRAM
  and closing the owner monitor instead of leaving orphan processes behind
- Conditional requests such as “find an alternative; create a tool only if none
  exists” no longer make artifact creation an unconditional completion
  requirement, while the fallback builders remain available when needed
- Discovery redirects derive a bounded query from the primary request instead
  of sending the complete multi-stage workflow to DuckDuckGo
- Online discovery accepts a detail page only when its URL was present in an
  earlier inspected search result, so a model cannot substitute a remembered
  address for actual search evidence
- Tight-context rollover removes duplicated dynamic memory before raw evidence,
  preserves the newest verified tool result, and no longer resumes from an
  evidence-free capsule
- A rollover advances its evidence cursor only through calls actually embedded
  in the raw capsule; an omitted call is retried on the next compaction instead
  of being silently forgotten
- V's voice gate rejects service-desk endings such as “If you'd like, I can...”
  and “What would you prefer?”
- Interactive checkpoints abandoned by a dead PALADYN process are recovered as
  interrupted on the next start instead of remaining permanently `running`
- Once a task's runtime evidence contract is satisfied, PALADYN closes tool
  execution and reserves two tool-free turns for the final report; continued
  tool requests are rejected and fall back to a deterministic evidence summary
- Browser snapshots are rejected after failed navigation until a working URL is
  opened, and a third byte-identical result from the same tool is detected even
  when the model changes superficial arguments such as target or filename
- Startup can safely reuse an already-loaded local llama-server only after its
  health endpoint, model alias, owner PID, executable, GGUF path, port, private
  bind address, and offline flags all match the selected profile; unrelated HTTP
  services on the port remain protected from takeover
- Polish web commands using the direct form `sieć` now deterministically route
  to browser execution instead of being misclassified as ordinary conversation
- Browser research now exposes and forwards `browser_type`, allowing V to enter
  and submit search queries rather than misuse in-page text search on a homepage
- Browser responses with HTTP 4xx/5xx status are recorded as failed evidence,
  and named online recommendations absent from observed sources are rejected
  before they can reach Boss as fabricated research
- Repeated context rollovers summarize only evidence added since the previous
  capsule, preventing duplicated calls and findings from consuming the window
- V's voice gate now detects formulaic corporate-report structure and validates
  the rewritten answer again; a still-sanitized first rewrite receives a second,
  stricter V-voice pass instead of being emitted unchecked
- Alternating between two dead URLs can no longer evade loop detection: an exact
  action that failed twice is rejected before another external call, and browser
  recovery directs the model to real search results instead of domain guessing
- Online discovery without an owner-supplied address now starts from DuckDuckGo;
  when an initial direct navigation fails before any page opens, PALADYN also
  falls back to DuckDuckGo instead of letting the model mutate the dead hostname
- Language-independent browser intent now preserves the same discovery rule,
  Google detours during discovery are redirected back to DuckDuckGo even after a
  successful page load, and HTTP/CAPTCHA failures cannot seed bypass advice into
  a context-rollover capsule
- Browser retry identity normalizes URL case, root slashes, and fragments, while
  context rollover replaces DNS-inspired domain guesses with a verified-search
  recovery step
- Every `learning_*` operation now exposes its real required argument schema;
  empty or malformed model calls are rejected before tool execution and returned
  to the model for a bounded corrected attempt
- Research mentioning ordinary tools no longer exposes the complete learning
  lifecycle, preventing unrelated browsing tasks from drifting into empty
  evidence, lesson, staging, or creation calls
- Generated tool creation receives a context-aware output budget large enough for
  its manifest, schemas, deterministic tests, and Python source instead of
  truncating the call at the ordinary 512-token response limit
- V's voice gate now catches additional helpdesk scripts such as "I know what
  you're asking", "Let me break it down", and "Would you like me to..."; the
  compact persona also restores the explicitly unsanitized hacker register
- Browser accessibility scaffolding such as `generic [ref=...]` can no longer be
  accepted as a concrete research finding
- Generic action requests such as "find all available information about ..."
  now reach the multilingual semantic router when the lexical imperative alone
  does not identify an execution capability
- Agent prompts contain only the tools selected for the current interaction;
  unavailable learning and creation tools are no longer advertised from the
  global profile catalog and cannot distract a research task
- An explicitly named active tool overrides an erroneous semantic guess that the
  user requested creation of a new tool
- Conditional follow-ups about creating a missing tool are recognized as task
  continuations, including the common Polish Whisper mistranscription observed
  during push-to-talk testing

## 2.0.0 - 2026-08-26

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
- An owner-approved privileged generated-code policy: `owner_lab` may autonomously
  create and persist tools using arbitrary Python imports, file operations,
  subprocesses, and dynamic execution inside PALADYN's audited offline sandbox;
  `client` retains the restricted source policy and lesson promotion gate
- A fail-closed Bubblewrap recovery path for AppArmor loopback failures that
  retains filesystem/process isolation and blocks networking with libseccomp
- A PALADYN AppArmor launch profile for Ubuntu systems that restrict
  unprivileged user namespaces required by Bubblewrap

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
- Explicit references to active generated-tool names are preserved in the
  per-task allowlist and become mandatory completion evidence; a failed call can
  no longer end that requested action as completed
- Active generated tools publish their validated manifest description and input
  schema instead of an empty generic parameter object
- For an explicitly named tool with one required string field, quoted user text
  deterministically repairs an empty model-authored argument object
- Spoken multi-part tool names tolerate one unique, conservative transcription
  error, so STT output such as `Can't Words` still resolves to `count_words`
  without allowing ambiguous tool selection
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
