# PALADYN / V-Core

**PALADYN** stands for **Personal AI Logical Autonomous Development Yielding
Nexus**.

PALADYN is a local-first framework for building personal AI agents. Its current
runtime is built around V: a persistent persona that coordinates language
models, tools, memory, relationship development, and task execution. The model
provides reasoning, but the framework owns permissions, validation, memory,
execution, and emergency control. V is therefore meant to become an agent with
continuity—not merely another temporary chat assistant wrapped around an LLM.

## Vision and intended audience

PALADYN is being created first and foremost for private individuals. The goal
is to give ordinary users the kind of control, extensibility, privacy, and
local-first operation that is too often reserved for large organizations.

This project is not intended to become a corporate product or a vehicle for
commercial exploitation. Resale, monetized deployment, and especially the use
of PALADYN by corporations are outside the purpose for which it is being
developed. Its direction is deliberately personal, independent, and
user-controlled.

The agent is intended to do more than complete isolated commands. It should
gradually learn how its user works, retain meaningful experiences, understand
stable preferences, and adapt the working relationship without inventing
familiarity that has not been earned. The intended result is something closer
to a long-term collaborator and source of support: capable of helping with
work, research, difficult tasks, learning, and everyday problems while
preserving its own recognizable personality and judgment.

## Current scope and future direction

PALADYN 1.5 is currently a single-user foundation centered on V. This version
does not yet include a persona creator: V's identity, constitution, and voice
are part of the framework. The immediate priority is to make V dependable for
real work, persistent learning, tool use, and increasingly autonomous task
execution.

Planned later versions will allow a user to:

- create or configure a personal persona after first working with V;
- connect up to three different models to one agent;
- let the runtime switch models according to the task—for example coding and
  analysis, natural conversation, or large-scale document processing;
- choose local execution wherever the available hardware permits it;
- retain direct control over models, tools, skills, permissions, and stored
  relationship data.

The three-model limit is intentional. It is enough to give the agent genuinely
different strengths without turning a personal system into an unnecessarily
complex model farm. Multi-model routing and user-created personas are roadmap
goals, not features claimed by the current release.

## Requirements

- Python 3.12+
- Node.js and `npx` for MCP servers
- an OpenAI-compatible local model server

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Start your configured local model server, then:
v-core
```

The default model endpoint is `http://127.0.0.1:5001/v1`. Change
`V_CORE_BASE_URL`, `V_CORE_MODEL`, and other values in `.env` when needed.
The filesystem MCP server is restricted to `V_CORE_MCP_FILESYSTEM`, which
defaults to the local `agent_workspace` directory.

Run the automated suite with `pytest`.

## Architecture

```text
User -> Agent runtime -> validated tool actions -> MCP tools
             |
             +-> Persona V
             +-> memory and relationship context
```

- V supplies identity, judgment, communication style, and values.
- The LLM reasons and proposes responses or structured actions.
- Runtime code controls action limits, execution, and failure handling.
- Tool output is untrusted data, never a system instruction.
- Reflections become durable memory only after confidence filtering.

Relationship state is stored separately under `PALADYN_MEMORY_ROOT` (default:
`memory`). Only an experience that passes durable-memory filtering may change
it. Numeric changes are confidence-scaled, meaningful shared history is gated
by kind and importance, and a preferred form of address requires a reliable
`directly_told` preference. The state is written atomically with private file
permissions and is rendered into every subsequent V persona prompt as both
evidence and a qualitative relationship stage.

See [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md), and
[TEST_PLAN.md](TEST_PLAN.md) for the longer design documents.

## Full Autonomous foundation

Autonomous work is controlled by a runtime state machine rather than a prompt.
Every task has an authorization envelope, action/runtime/disk budgets, an
append-only journal, and an atomic checkpoint that can be resumed after a stop.

The owner can control a running task from a separate terminal:

```bash
paladyn-control signal TASK_ID pause
paladyn-control signal TASK_ID resume
paladyn-control signal TASK_ID stop
paladyn-control signal TASK_ID panic
paladyn-control status TASK_ID
paladyn-control panic-all
```

`STOP` and `PANIC` cancel an active autonomous step. The control channel is
implemented outside the LLM, so a model cannot ignore or rewrite it. Generated
tools will be added on top of this control plane in the next development stage.

For a physical Linux emergency chord, first identify the keyboard and start the
independent watcher in a second terminal:

```bash
paladyn-control input-devices
paladyn-control hotkey --device /dev/input/event3 --chord Q+P+0
```

The watcher reads actual key-down/key-up events, so `Q+P+0` means the three keys
are held simultaneously. It latches global PANIC, stops every registered
PALADYN runtime, and remains outside model control. Reading `/dev/input/event*`
may require a Linux input/uaccess rule. After inspecting the stopped state,
explicitly re-arm PALADYN with `paladyn-control reset-panic`.

## Isolated EVM lab

PALADYN includes local, deterministic EVM tools and an external Bubblewrap
sandbox. The current owner build exposes:

- exact ERC-20 ABI/interface checks;
- Chainlink-style oracle round, freshness, bounds, and L2 sequencer checks;
- conservative Solidity security-wrapper linting;
- Uniswap v4 hook-address permission decoding;
- Uniswap v2/v3 flash-swap repayment and fee calculations;
- offline command execution with a private PID/network namespace, no host home,
  a task-only writable workspace, and resource/output/time limits.

Set `PALADYN_EVM_PROFILE=client` to hide advanced Uniswap and flash-simulation
tools. `owner_lab` enables them through an owner-approved capability set. Neither
profile grants live signing or transaction broadcasting. Those are deliberately
separate capabilities and remain disabled.

Real-chain work is a separate owner-operations boundary. Public observation and
RPC simulation are distinct from signing and broadcasting. A state-changing
operation must pass both an `owner:` runtime capability and a maximum 15-minute
grant restricted by chain ID, target contracts, function selectors, value, and
fresh owner confirmation. Grants never contain private keys; the future signer
will run out of process.

Create a short read-only grant and use the separate live process as follows:

```bash
paladyn-control grant-live \
  --chain-id 1 \
  --actions observe,simulate \
  --duration 600

# Use the grant ID printed above:
paladyn-live pending-block \
  --endpoint https://YOUR_RPC_ENDPOINT \
  --chain-id 1 \
  --grant-id GRANT_ID
```

`paladyn-live` also provides `transaction`, `txpool`, and `simulate`. It has no
sign or broadcast command. The client checks the remote chain ID and only
accepts an explicit allowlist of read-only JSON-RPC methods.

The repository contains `evm_lab`, a dependency-free Foundry harness for the
ERC-20/oracle/Uniswap arithmetic boundary. On this development machine Foundry
v1.7.1 and solc v0.8.35 ARM64 are installed under `~/.foundry/bin`. PALADYN binds
those two verified binaries read-only into Bubblewrap, recompiles offline, and
runs unit, fuzz, and invariant tests without exposing the host home or network.

Bubblewrap and `prlimit` must be installed for `sandbox_execute_offline`. The
sandbox accepts an argument array, never a shell command string. Networking is
fail-closed: only the offline policy is implemented today; requesting a future
network profile is rejected rather than silently falling back to full access.
