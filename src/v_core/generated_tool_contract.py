"""Runtime-owned semantic test contracts for source-only generated tools.

The language model may interpret the owner's wording, but it cannot authorize
invented fixtures.  Every value in an accepted contract must be recoverable
from an exact quote in the current owner message.  The resulting contract is
then frozen before a different model is asked to write code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any

from .utils import parse_llm_json


class GeneratedToolContractError(ValueError):
    """Natural-language fixtures could not be frozen without guessing."""


@dataclass(frozen=True, slots=True)
class GeneratedToolTest:
    arguments: dict[str, Any]
    expected: dict[str, Any]
    evidence_quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arguments": deepcopy(self.arguments),
            "expected": deepcopy(self.expected),
            "evidence_quote": self.evidence_quote,
        }


@dataclass(frozen=True, slots=True)
class GeneratedToolContract:
    tests: tuple[GeneratedToolTest, ...]
    final_arguments: dict[str, Any]
    final_evidence_quote: str
    provenance: str = "owner_text_semantic_extraction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests": [case.to_dict() for case in self.tests],
            "final_arguments": deepcopy(self.final_arguments),
            "final_evidence_quote": self.final_evidence_quote,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GeneratedToolContract":
        return cls(
            tests=tuple(
                GeneratedToolTest(
                    arguments=deepcopy(item["arguments"]),
                    expected=deepcopy(item["expected"]),
                    evidence_quote=str(item["evidence_quote"]),
                )
                for item in value.get("tests", [])
            ),
            final_arguments=deepcopy(value.get("final_arguments", {})),
            final_evidence_quote=str(value.get("final_evidence_quote", "")),
            provenance=str(value.get("provenance", "owner_text_semantic_extraction")),
        )


_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "generated_tool_test_contract",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready", "ambiguous", "absent"]},
                "tests": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "arguments": {"type": "object"},
                            "expected": {"type": "object"},
                            "evidence_quote": {"type": "string", "maxLength": 500},
                        },
                        "required": ["arguments", "expected", "evidence_quote"],
                        "additionalProperties": False,
                    },
                },
                "final_arguments": {"type": "object"},
                "final_evidence_quote": {"type": "string", "maxLength": 500},
                "reason": {"type": "string", "maxLength": 300},
            },
            "required": [
                "status", "tests", "final_arguments", "final_evidence_quote", "reason"
            ],
            "additionalProperties": False,
        },
    },
}


_SYSTEM_PROMPT = """
You extract a test contract for a newly requested software tool. Understand the
owner's message in any language, including mixed-language speech transcription.
Do not write code and do not solve the examples. Extract only examples whose
input AND expected output are explicitly stated by the owner, plus the separate
input on which the owner wants the finished tool invoked.

Return JSON only:
{"status":"ready|ambiguous|absent","tests":[{"arguments":{},"expected":{},
"evidence_quote":"exact current-message quote"}],"final_arguments":{},
"final_evidence_quote":"exact current-message quote","reason":""}

Use short stable snake_case field names consistently. A test quote must contain
every value placed in that test. The final quote must contain every final input
value. Quotes must be exact substrings of current_user_message. Never infer a
missing expected value, invent an example, copy an output produced by candidate
code, or turn the final invocation into a test. If relationships conflict or a
value cannot be assigned unambiguously, return ambiguous. If no explicit
input/output example exists, return absent. JSON only, no markdown.
""".strip()


def has_natural_fixture_candidate(prompt: str) -> bool:
    """Cheap language-neutral gate before spending a model turn.

    Three scalar literals are the minimum useful shape: one input/output pair
    plus a separate final invocation.  The semantic extractor decides their
    roles; this gate does not interpret words or relationships.
    """

    numbers = re.findall(r"(?<![\w.])-?(?:0|[1-9]\d*)(?:\.\d+)?(?![\w.])", prompt)
    quoted = re.findall(r"(?<!\\)(?:\"(?:\\.|[^\"\\])+\"|'(?:\\.|[^'\\])+')", prompt)
    return len(numbers) + len(quoted) >= 3


def _leaf_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_values(child)
    else:
        yield value


def _value_is_grounded(value: Any, quote: str) -> bool:
    if isinstance(value, str):
        return value.casefold() in quote.casefold()
    if value is None:
        token = "null"
    elif isinstance(value, bool):
        token = "true" if value else "false"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        token = json.dumps(value, allow_nan=False)
    else:
        return False
    return re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", quote, re.IGNORECASE) is not None


def _grounded_object(value: Any, quote: str, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise GeneratedToolContractError(f"{label} must be a non-empty JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GeneratedToolContractError(f"{label} contains non-JSON data") from error
    if len(encoded.encode("utf-8")) > 50_000:
        raise GeneratedToolContractError(f"{label} is too large")
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", str(key)) for key in value):
        raise GeneratedToolContractError(f"{label} contains an invalid field name")
    if not all(_value_is_grounded(item, quote) for item in _leaf_values(value)):
        raise GeneratedToolContractError(f"{label} contains a value absent from its evidence quote")
    return deepcopy(value)


def parse_generated_tool_contract(response: str, prompt: str) -> GeneratedToolContract:
    payload = parse_llm_json(response, default={})
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        reason = str(payload.get("reason", "")) if isinstance(payload, dict) else ""
        raise GeneratedToolContractError(reason or "no unambiguous owner-specified test contract")
    raw_tests = payload.get("tests")
    if not isinstance(raw_tests, list) or not 1 <= len(raw_tests) <= 8:
        raise GeneratedToolContractError("the contract must contain between one and eight tests")

    tests: list[GeneratedToolTest] = []
    input_keys: tuple[str, ...] | None = None
    output_keys: tuple[str, ...] | None = None
    seen: dict[str, str] = {}
    for index, raw in enumerate(raw_tests, start=1):
        if not isinstance(raw, dict):
            raise GeneratedToolContractError(f"test {index} is not an object")
        quote = str(raw.get("evidence_quote", ""))
        if not quote or quote not in prompt:
            raise GeneratedToolContractError(f"test {index} quote is not exact owner text")
        arguments = _grounded_object(raw.get("arguments"), quote, label=f"test {index} arguments")
        expected = _grounded_object(raw.get("expected"), quote, label=f"test {index} expected")
        current_input_keys = tuple(sorted(arguments))
        current_output_keys = tuple(sorted(expected))
        if input_keys is None:
            input_keys, output_keys = current_input_keys, current_output_keys
        elif current_input_keys != input_keys or current_output_keys != output_keys:
            raise GeneratedToolContractError("test cases use inconsistent input or output fields")
        argument_digest = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        expected_digest = json.dumps(expected, sort_keys=True, ensure_ascii=False)
        previous = seen.setdefault(argument_digest, expected_digest)
        if previous != expected_digest:
            raise GeneratedToolContractError("the same input has conflicting expected outputs")
        tests.append(GeneratedToolTest(arguments, expected, quote))

    final_quote = str(payload.get("final_evidence_quote", ""))
    if not final_quote or final_quote not in prompt:
        raise GeneratedToolContractError("final invocation quote is not exact owner text")
    final_arguments = _grounded_object(
        payload.get("final_arguments"), final_quote, label="final arguments"
    )
    if (
        tuple(sorted(final_arguments)) != input_keys
        and input_keys is not None
        and len(input_keys) == 1
        and len(final_arguments) == 1
    ):
        # In ordinary speech the examples may say "a number gives ..." while
        # the final invocation names the same sole input ``n``.  With one field
        # there is no competing mapping to guess between, so canonicalize by
        # arity while retaining the separately grounded value.  Multi-field
        # aliases remain ambiguous and fail below.
        final_arguments = {input_keys[0]: next(iter(final_arguments.values()))}
    if tuple(sorted(final_arguments)) != input_keys:
        raise GeneratedToolContractError("final invocation fields differ from test input fields")
    return GeneratedToolContract(tuple(tests), final_arguments, final_quote)


async def extract_generated_tool_contract(llm: Any, prompt: str) -> GeneratedToolContract:
    response = await llm.ask(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"current_user_message": prompt}, ensure_ascii=False),
            },
        ],
        max_tokens=512,
        temperature=0.0,
        response_format=_RESPONSE_FORMAT,
    )
    return parse_generated_tool_contract(str(response), prompt)
