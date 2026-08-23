# PALADYN Learning, Skills, and Generated Tools

PALADYN learning is an evidence-driven runtime mechanism. It is not model
fine-tuning, an unrestricted self-modifying loop, or an LLM writing claims into
its own system prompt.

## Lifecycle

```text
task result
    |
    v
recorded evidence -- rejected if unsupported or malformed
    |
    v
candidate lesson -- remains a hypothesis without independent support
    |
    v
validated lesson -- multiple tasks/fingerprints + verified evidence
    |
    v
tool or skill bundle in quarantine
    |
    +--> static policy and schema checks
    +--> offline sandbox tests (tools)
    +--> deterministic trigger tests (skills)
    |
    v
validated artifact
    |
    v
explicit capability gate
    |
    v
active, versioned artifact -- retired on replacement or repeated failure
```

Every evidence and artifact transition is appended to a private hash-chained
journal. Artifact source, manifest, scope, and status are checked against that
journal before use. Bundles are immutable: a change requires a new semantic
version.

## What counts as learning

PALADYN records the origin of each observation:

- `user_correction`: Boss explicitly corrected a result;
- `tool_result`: a tool produced relevant evidence;
- `test_result`: a deterministic test or verifier succeeded or failed;
- `task_runtime`: Full Autonomous encountered a real runtime outcome;
- `self_review`: V proposed an interpretation that is not independently proven.

An autonomous exception is verifiable evidence that the step failed. It is not
automatically proof of the root cause. Likewise, task completion is not stored
as a successful learning event unless the step supplies `metadata.verified=true`.
That flag is accepted only on the trusted runtime path: JSON proposed by an LLM
cannot declare itself verified. Model-proposed observations are downgraded to
unverified `self_review` evidence with confidence capped at `0.50`.

When V identifies an explicit correction from Boss, the framework binds the
evidence to its own interaction ID and stores Boss's raw message instead of a
model-authored replacement. Significant whitespace and line breaks are
preserved up to the evidence-size bound. The correction remains unverified
until supported by a real test or runtime verifier.

A lesson becomes `validated` only when it has:

- evidence from at least two distinct tasks;
- at least two distinct evidence fingerprints;
- at least one verified item;
- average confidence of at least `0.65`;
- at least one failure, regression, or correction.

Until then it remains a candidate and cannot justify a persistent artifact.

## Generated tools

The first supported generated-tool format is a small offline Python tool. A
tool must expose a synchronous function:

```python
def run(arguments):
    return {"result": arguments["value"] * 2}
```

Its manifest defines immutable versioning, JSON input/output schemas, scope,
linked lessons, limits, and exact test cases. PALADYN then:

1. validates the schema and reserved name boundary;
2. parses the Python AST and rejects dangerous imports/calls;
3. mounts the trusted host and generated source read-only into Bubblewrap;
4. runs every case with no network, no host home, no inherited environment,
   process/memory/CPU limits, output limits, a wall-clock timeout, and per-file
   and total-workspace disk limits;
5. validates the returned JSON against the declared output schema;
6. checks the bundle digest again immediately before activation.

Generated tools cannot replace built-in PALADYN, filesystem, browser, learning,
or EVM tool names. The initial backend intentionally supports only an allowlist
of non-I/O standard-library modules. Networked tools and third-party dependency
installation require a future enforcing backend and are rejected today.

An active tool is automatically retired after three execution failures. If it
replaced an earlier active version, PALADYN rolls back to that directly
superseded, already validated bundle. Invalid caller input is rejected by the
schema and does not count as a tool failure.
Generated source is capped at 200 KB, a manifest at 2 MB, each invocation at
64 KB of JSON arguments, and a tool at 100 exact test cases. Schemas are bounded
in depth and field count; non-finite and non-JSON values are rejected.

## Generated skills

A skill is a versioned operational playbook, not arbitrary code. Its manifest
contains:

- description and trigger phrases;
- ordered workflow steps;
- required registered tools;
- positive and negative trigger tests;
- task or persistent scope;
- optional links to learned lessons.

Validation rejects missing tools, failed trigger cases, and attempts to modify
V's persona, constitution, permissions, credentials, or emergency controls.
Only matching active skills are rendered into the next agent prompt. They are
delimited as workflow data and explicitly remain below the constitution and
runtime policy.

A skill is capped at 20 triggers, 20 steps, 64 required tools, 100 trigger test
cases, and a 500 KB manifest. It must include at least one matching and one
non-matching test.

## Scope and owner profiles

`task` artifacts are bound to a SHA-256-derived identity of the authorized task
workspace. They remain available after a restart of that same workspace but are
invisible to other workspaces.

`persistent` artifacts can be reused across tasks. They require:

- a validated linked lesson;
- `owner:create_persistent_artifacts` to stage;
- `owner:activate_persistent_artifacts` to activate;
- both capabilities in the normal and owner-approved capability sets.

`PALADYN_LEARNING_PROFILE=owner_lab` pre-authorizes those two owner capabilities
for the owner's build. This allows Full Autonomous to promote a fully validated
artifact without interrupting Boss for every individual creation. The `client`
profile does not pre-authorize persistent promotion.

## Operator audit

```bash
paladyn-learning verify
paladyn-learning artifacts
paladyn-learning evidence --limit 50
paladyn-learning lessons
```

`verify` fails closed if an evidence/audit journal, artifact identity, status,
manifest, or source digest has been modified. The learning root defaults to
`learning` and can be changed with `PALADYN_LEARNING_ROOT` or `--root`.

## Protected core

Generated artifacts live outside the PALADYN source tree. The persona,
constitution, relationship policy, learning policy, trusted generated-tool
host, authorization envelope, and kill-switch implementation are protected
paths. Learning may add evidence, lessons, tools, and skills; it may not redefine
the mechanism that decides whether those artifacts are trusted.
