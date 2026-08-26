from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from ..capabilities.web_target import requests_web_access


_ONLINE_ACTION = re.compile(
    r"\b(?:browse|check|collect|extract|find|inspect|list|look\s+for|monitor|open|"
    r"research|scan|search|visit|crawl|scrape|"
    r"ekstrakc\w*|monitor\w*|przejr\w*|przeszuk\w*|sprawd\w*|szuk\w*|wejd\w*|"
    r"wyszuk\w*|wyciagn\w*|wyciągn\w*|znajd\w*)\b",
    re.IGNORECASE,
)
_ONLINE_RESOURCE = re.compile(
    r"\b(?:browser|darknet|internet|online|page|repository|repo|site|web|website|"
    r"github|osint|internet\w*|interne\w*|market\w*|sieci|stron\w*|witryn\w*)\b",
    re.IGNORECASE,
)
_DETAIL_PAGE = re.compile(
    r"\b(?:first|top)\s+(?:result|repository|repo|link)|"
    r"\b(?:inspect|open|visit)\s+(?:the\s+)?(?:first|top)\b|"
    r"\bpierwsz\w*\s+(?:wynik\w*|repozytori\w*|link\w*)|"
    r"\b(?:otworz|otwórz|sprawdz|sprawdź|wejdz|wejdź)\w*\s+"
    r"(?:w\s+)?pierwsz\w*\b",
    re.IGNORECASE,
)
_CREATE_TOOL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\s+(?:(?:a|an|custom|local|new|now\w*|"
    r"generated|lokal\w*|wlasn\w*|własn\w*)\s+){0,3}"
    r"(?:tool|narzedzi\w*|narzędzi\w*)\b",
    re.IGNORECASE,
)
_CREATE_SKILL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\s+(?:(?:a|an|new|now\w*|generated)\s+){0,3}"
    r"(?:skill|umiejetn\w*|umiejętn\w*)\b",
    re.IGNORECASE,
)
_USE_CREATED_TOOL = re.compile(
    r"\b(?:and\s+then\s+use|then\s+use|use\s+it|"
    r"a\s+nastepnie\s+uzyj|a\s+następnie\s+użyj|potem\s+uzyj|"
    r"potem\s+użyj|uzyj\s+go|użyj\s+go)\b",
    re.IGNORECASE,
)
_READ_FILE = re.compile(
    r"\b(?:cat|inspect|open|read|review|show|"
    r"odczyt\w*|otworz\w*|otwórz\w*|przeczyt\w*|przejr\w*|pokaz\w*|pokaż\w*)\b",
    re.IGNORECASE,
)
_MUTATE_FILE = re.compile(
    r"\b(?:append|create|delete|edit|move|rename|replace|save|write|"
    r"dodaj\w*|edytuj\w*|napisz\w*|przenies\w*|przenieś\w*|"
    r"usun\w*|usuń\w*|utworz\w*|utwórz\w*|zapisz\w*)\b",
    re.IGNORECASE,
)
_RUN_COMMAND = re.compile(
    r"\b(?:execute|run|test|uruchom\w*|wykonaj\w*|przetestuj\w*)\b",
    re.IGNORECASE,
)
_COMMAND_TARGET = re.compile(
    r"\b(?:command|script|shell|test|tests|pytest|foundry|forge|"
    r"komend\w*|skrypt\w*|test\w*)\b",
    re.IGNORECASE,
)
_FILE_TARGET = re.compile(
    r"(?:\b(?:file|plik\w*)\b|(?:^|[\s/])[\w.-]+\."
    r"(?:cfg|conf|csv|docx?|html?|ini|json|log|md|pdf|py|sh|sol|toml|ts|txt|"
    r"xml|ya?ml)\b)",
    re.IGNORECASE,
)
_REPORT_RESULT = re.compile(
    r"\b(?:answer|describe|explain|extract|find|give|identify|list|report|summari[sz]e|tell|what|which|"
    r"co|jakie|które|ktore|opisz\w*|podaj\w*|powiedz\w*|stre[śs]c\w*|wyciagn\w*|wyciągn\w*|"
    r"wymien\w*|wymień\w*|znajd\w*|znale[źz]\w*)\b",
    re.IGNORECASE,
)
_FIRST_HEADING = re.compile(
    r"\b(?:first\s+(?:heading|header|title)|"
    r"pierwsz\w*\s+(?:naglow\w*|nagłów\w*|tytul\w*|tytuł\w*))\b",
    re.IGNORECASE,
)
_GROUNDING_STOPWORDS = {
    "about", "after", "also", "and", "browser", "content", "data", "file",
    "first", "from", "have", "heading", "into", "page", "result", "snapshot",
    "that", "their", "there", "these", "this", "tool", "verified", "with",
}


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Runtime-owned, model-independent definition of completion evidence."""

    requires_browser_navigation: bool = False
    requires_browser_snapshot: bool = False
    requires_distinct_detail_page: bool = False
    requires_file_read: bool = False
    requires_file_mutation: bool = False
    requires_command_execution: bool = False
    requires_evidence_report: bool = False
    requires_first_heading: bool = False
    requires_created_tool: bool = False
    requires_created_tool_execution: bool = False
    requires_created_skill: bool = False

    @classmethod
    def from_prompt(cls, prompt: str) -> "TaskContract":
        online = requests_web_access(prompt) or bool(
            _ONLINE_ACTION.search(prompt) and _ONLINE_RESOURCE.search(prompt)
        )
        file_prompt = re.sub(r"https?://[^\s<>]+", "", prompt, flags=re.IGNORECASE)
        file_read = bool(
            _READ_FILE.search(file_prompt) and _FILE_TARGET.search(file_prompt)
        )
        file_mutation = bool(
            _MUTATE_FILE.search(file_prompt) and _FILE_TARGET.search(file_prompt)
        )
        command_execution = bool(
            not online and _RUN_COMMAND.search(prompt) and _COMMAND_TARGET.search(prompt)
        )
        creates_tool = bool(_CREATE_TOOL.search(prompt))
        return cls(
            requires_browser_navigation=online,
            requires_browser_snapshot=online,
            requires_distinct_detail_page=online and bool(_DETAIL_PAGE.search(prompt)),
            requires_file_read=file_read,
            requires_file_mutation=file_mutation,
            requires_command_execution=command_execution,
            requires_evidence_report=(online or file_read or command_execution) and bool(
                _REPORT_RESULT.search(prompt)
            ),
            requires_first_heading=file_read and bool(_FIRST_HEADING.search(prompt)),
            requires_created_tool=creates_tool,
            requires_created_tool_execution=(
                creates_tool and bool(_USE_CREATED_TOOL.search(prompt))
            ),
            requires_created_skill=bool(_CREATE_SKILL.search(prompt)),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "TaskContract":
        """Rebuild a contract from runtime-owned checkpoint data."""

        source = values if isinstance(values, dict) else {}
        return cls(
            **{
                name: bool(source.get(name, False))
                for name in cls.__dataclass_fields__
            }
        )

    def merged(self, other: "TaskContract") -> "TaskContract":
        """Return the union of two independently detected requirements."""

        return type(self)(
            **{
                name: bool(getattr(self, name) or getattr(other, name))
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def unmet(self, calls: list[dict[str, Any]]) -> list[str]:
        succeeded = [call for call in calls if call.get("status", "succeeded") == "succeeded"]
        names = [str(call.get("tool", "")) for call in succeeded]
        missing: list[str] = []

        if self.requires_browser_navigation and "browser_navigate" not in names:
            missing.append("browser_navigate")
        if self.requires_browser_snapshot and "browser_snapshot" not in names:
            missing.append("browser_snapshot")

        if self.requires_distinct_detail_page:
            navigations: list[tuple[int, str]] = []
            snapshots: list[int] = []
            for index, call in enumerate(succeeded):
                name = str(call.get("tool", ""))
                if name == "browser_navigate":
                    arguments = call.get("arguments", {})
                    url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                    if url and all(previous_url != url for _, previous_url in navigations):
                        navigations.append((index, url))
                elif name == "browser_snapshot":
                    snapshots.append(index)
            if len(navigations) < 2:
                missing.append("browser_navigate:distinct_detail_page")
            elif not any(index > navigations[1][0] for index in snapshots):
                missing.append("browser_snapshot:detail_page")

        if self.requires_file_read and not any(
            name in {"read_file", "cat"} for name in names
        ):
            missing.append("read_file")

        if self.requires_file_mutation and not any(
            name in {
                "write_file",
                "edit_file",
                "move_file",
                "create_directory",
            }
            for name in names
        ):
            missing.append("filesystem_mutation")

        if self.requires_command_execution and not any(
            name in {"sandbox_execute_offline", "evm_foundry_test_offline"}
            for name in names
        ):
            missing.append("command_execution")

        if self.requires_created_tool and "learning_create_tool" not in names:
            missing.append("learning_create_tool")
        if self.requires_created_tool_execution:
            created_name = ""
            created_index = -1
            for index, call in enumerate(succeeded):
                if call.get("tool") != "learning_create_tool":
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("name"):
                    created_name = str(payload["name"])
                    created_index = index
            if not created_name or not any(
                index > created_index and call.get("tool") == created_name
                for index, call in enumerate(succeeded)
            ):
                missing.append("generated_tool_execution")
        if self.requires_created_skill and "learning_create_skill" not in names:
            missing.append("learning_create_skill")
        return missing

    def answer_issues(self, answer: str, calls: list[dict[str, Any]]) -> list[str]:
        """Reject an empty completion claim that does not use observed evidence."""

        if not self.requires_evidence_report:
            return []
        observations = [
            str(call.get("result_excerpt", ""))
            for call in calls
            if call.get("status", "succeeded") == "succeeded"
            and call.get("tool") in {
                "browser_snapshot",
                "read_file",
                "cat",
                "sandbox_execute_offline",
                "evm_foundry_test_offline",
            }
            and call.get("result_excerpt")
        ]
        if not observations:
            return ["answer:evidence_observation_missing"]

        if self.requires_first_heading:
            first_line = next(
                (line.strip() for line in observations[0].splitlines() if line.strip()),
                "",
            )
            heading_tokens = {
                token
                for token in re.findall(r"[\w.-]{2,}", first_line.casefold())
                if token not in _GROUNDING_STOPWORDS
            }
            answer_heading_tokens = set(
                re.findall(r"[\w.-]{2,}", answer.casefold())
            )
            if not first_line or not heading_tokens.issubset(answer_heading_tokens):
                return ["answer:first_heading_missing"]

        answer_tokens = {
            token
            for token in re.findall(r"[\w.-]{4,}", answer.casefold())
            if token not in _GROUNDING_STOPWORDS
        }
        evidence_tokens = {
            token
            for token in re.findall(
                r"[\w.-]{4,}", "\n".join(observations).casefold()
            )
            if token not in _GROUNDING_STOPWORDS
        }
        if not answer_tokens.intersection(evidence_tokens):
            return ["answer:not_grounded_in_tool_evidence"]
        return []

    def deterministic_answer(self, calls: list[dict[str, Any]]) -> str | None:
        """Produce exact results for objectives that require no model judgment."""

        if not self.requires_first_heading:
            return None
        for call in reversed(calls):
            if (
                call.get("status", "succeeded") != "succeeded"
                or call.get("tool") not in {"read_file", "cat"}
            ):
                continue
            for line in str(call.get("result_excerpt", "")).splitlines():
                if line.strip():
                    return line.strip()
        return None
