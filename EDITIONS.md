# PALADYN editions

PALADYN's distribution boundary is based on operational capability, not on
conversation topics or artificial product degradation.

## Public

The public edition is a complete personal-agent foundation. It contains V,
local model discovery and qualification, deterministic routing across up to
three models, memory and relationship development, evidence-based learning,
bounded autonomous tasks, browser and workspace tools, and generated tools
executed in the restricted sandbox.

Public PALADYN contains only `src/v_core`. Requesting `PALADYN_EDITION=full` or
an `owner_lab` profile without the private package fails closed.

## Full

The private Full edition adds `src/v_full`. It owns privileged generated-code
authorization, the owner performance monitor, advanced EVM simulation and
Foundry integration, short-lived live-operation grants, and a bounded bridge to
the host Tor service. The bridge exposes fixed status/search/fetch operations,
not a shell or general package installer; generated tools remain offline. These
capabilities are registered through the edition-extension contract; `v_core`
does not import their implementations directly. The shared graphical shell asks
that extension for an optional UI contribution. Public PALADYN receives none;
Full supplies the private Owner Deck and its local operational status.

Both editions retain the execution-evidence contract, external emergency stop,
capability ownership, generated-code validation, audit trail, and protected
agent core.

## Public export

Run the exporter from the private repository:

```bash
python scripts/export_public.py /path/to/empty/PALADYN-public
```

The target must be new or an earlier directory created by this exporter. The
export follows `editions/public.toml`, rewrites Full-only command entry points,
and rejects private paths or static imports before it reports success. Publishing
the resulting directory remains a separate, explicit Git operation.
