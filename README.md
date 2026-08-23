# PALADYN / V-Core

PALADYN is a local-first agent framework built around V: a persistent persona
that coordinates an LLM, tools, memory, and task execution. The LLM proposes
actions; the runtime is responsible for executing and validating them.

The current target is a dependable single-user V runtime before building the
reduced client edition or additional personas.

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

See [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md), and
[TEST_PLAN.md](TEST_PLAN.md) for the longer design documents.
