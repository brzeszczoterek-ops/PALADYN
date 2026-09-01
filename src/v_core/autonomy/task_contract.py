from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any
from html import unescape
from urllib.parse import unquote, urlsplit

from ..capabilities.web_target import extract_web_target, requests_web_access


_ONLINE_ACTION = re.compile(
    r"\b(?:browse|check|collect|extract|find|inspect|list|look\s+for|monitor|open|"
    r"research|scan|search|visit|crawl|scrape|"
    r"ekstrakc\w*|monitor\w*|przejr\w*|przeszuk\w*|sprawd\w*|szuk\w*|wejd\w*|"
    r"wyszuk\w*|wyciagn\w*|wyciągn\w*|znajd\w*)\b",
    re.IGNORECASE,
)
_ONLINE_RESOURCE = re.compile(
    r"\b(?:browser|darknet|internet|online|page|repository|repo|site|web|website|"
    r"github|facebook\w*|instagram\w*|linkedin\w*|osint|profile|social\s+media|"
    r"internet\w*|interne\w*|market\w*|profil\w*|sie[cć]\w*|stron\w*|"
    r"witryn\w*)\b",
    re.IGNORECASE,
)
_TOR_ACCESS_REQUEST = re.compile(
    r"(?:\b(?:w|przez)\s+(?:darknet\w*|sieci\s+tor)\b|"
    r"\b(?:in|on|through|via)\s+(?:the\s+)?"
    r"(?:darknet|dark\s*web|tor\s+network)\b|"
    r"\b(?:przeszuk\w*|wejd\w*)\s+(?:do\s+|w\s+)?(?:darknet\w*|sie[cć]\s+tor)\b|"
    r"\b(?:search|browse|visit|open|inspect)\s+(?:the\s+)?"
    r"(?:darknet|dark\s*web|tor\s+network)\b|"
    r"\b(?:using|używ\w*|uzyw\w*)\s+(?:the\s+)?tor(?:\s+browser)?\b|"
    r"\.onion\b)",
    re.IGNORECASE,
)
_EXPLICIT_ONION_ADDRESS = re.compile(
    r"https?://(?:[a-z0-9-]+\.)*[a-z2-7]{56}\.onion(?:[^\s<>]*)?",
    re.IGNORECASE,
)
_EXPLICIT_WEB_ADDRESS = re.compile(
    r"(?:https?://|www\.|\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}(?:[/:?#]|\b))",
    re.IGNORECASE,
)
_DETAIL_PAGE = re.compile(
    r"\b(?:first|top)\s+(?:result|repository|repo|link)|"
    r"\b(?:inspect|open|visit)\s+(?:the\s+)?(?:first|top)\s+"
    r"(?:result|repository|repo|link)\b|"
    r"\bpierwsz\w*\s+(?:wynik\w*|repozytori\w*|link\w*)|"
    r"\b(?:otworz|otwórz|sprawdz|sprawdź|wejdz|wejdź)\w*\s+"
    r"(?:w\s+)?pierwsz\w*\s+(?:wynik\w*|repozytori\w*|link\w*)\b",
    re.IGNORECASE,
)
_CREATE_TOOL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|utworz|utwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\b(?:(?![.!?;\n]).){0,180}?\b"
    r"(?:tool\w*|narzedzi\w*|narzędzi\w*)\b",
    re.IGNORECASE,
)
_CREATE_SKILL = re.compile(
    r"\b(?:create|build|implement|write|generate|stworz|stwórz|utworz|utwórz|zbuduj|napisz|"
    r"wygeneruj|zaimplementuj)\w*\b(?:(?![.!?;\n]).){0,180}?\b"
    r"(?:skill\w*|umiejetn\w*|umiejętn\w*)\b",
    re.IGNORECASE,
)
_QUOTED_TEXT = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|„[^”\n]*(?:”|")|“[^”\n]*”',
    re.DOTALL,
)
_CONDITIONAL_FALLBACK = re.compile(
    r"\b(?:if|unless|otherwise|when\s+nothing|"
    r"je(?:sli|śli|zeli|żeli)|gdy|wtedy|w\s+przeciwnym\s+razie|"
    r"wenn|falls|sonst|si|sinon|se|altrimenti|"
    r"ha|amennyiben|különben|pokud|jinak|ak|inak|"
    r"если|иначе|якщо|інакше)\b",
    re.IGNORECASE,
)
_RUNTIME_REVIEW = re.compile(
    r"(?:\b(?:analy[sz]e|audit|inspect|review)\w*\b.{0,80}"
    r"\b(?:execution|logs?|run|runtime|session|trace)\w*\b|"
    r"\b(?:analizuj|przeanalizuj|przejrzyj|sprawd[źz])\w*\b.{0,80}"
    r"\b(?:b[łl][ęe]d\w*|log\w*|przebieg\w*|sesj\w*)\b|"
    r"\bruntime_review_task\b|^/review-last-run\b)",
    re.IGNORECASE,
)
_USE_CREATED_TOOL = re.compile(
    r"\b(?:and\s+then\s+(?:execute|invoke|run|use)|"
    r"then\s+(?:really\s+)?(?:execute|invoke|run|use)|"
    r"(?:really\s+)?(?:execute|invoke|run|use)\s+it|"
    r"(?:really\s+)?(?:execute|invoke|run|use)\s+(?:the\s+)?"
    r"(?:(?:new|newly\s+created|created)\s+)?tool|"
    r"a\s+nastepnie\s+uzyj|a\s+następnie\s+użyj|potem\s+uzyj|"
    r"potem\s+użyj|uzyj\s+go|użyj\s+go|"
    r"(?:show|give|present|demonstrate)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:result|results|output|demo)|"
    r"(?:pokaz|pokaż|przedstaw|podaj)\s+(?:mi\s+)?(?:jego\s+|jej\s+)?"
    r"(?:rezultat\w*|wynik\w*|dzialani\w*|działani\w*)|"
    r"a\s+nast[eę]pnie\s+(?:wykonaj|uruchom)|"
    r"potem\s+(?:wykonaj|uruchom)|(?:wykonaj|uruchom)\s+(?:go|je))\b",
    re.IGNORECASE,
)
_DISABLE_WEB = re.compile(
    r"\b(?:(?:do\s+not|don't|never|without)\s+"
    r"(?:live\s+)?(?:browse|browsing|contact|crawl|crawling|navigate|network|"
    r"scrape|scraping|search|visit)|"
    r"(?:bez|nigdy\s+nie|nie)\s+(?:kontakt\w*|laczeni\w*|łączeni\w*|"
    r"nawig\w*|przeglad\w*|przegląd\w*|sieci\w*|wyszuk\w*)|"
    r"offline[- ]only)\b",
    re.IGNORECASE,
)


def _search_outside_quoted_text(
    pattern: re.Pattern[str], prompt: str
) -> re.Match[str] | None:
    """Ignore command-like words that belong to a quoted input fixture."""

    quoted_spans = [match.span() for match in _QUOTED_TEXT.finditer(prompt)]
    for match in pattern.finditer(prompt):
        if not any(start <= match.start() < end for start, end in quoted_spans):
            return match
    return None
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
_EXPLICIT_LOCAL_PATH = re.compile(
    r"(?:^|[\s'\"`])(?:[a-z]:[\\/]|/|~/|\.\.?/)[^\s'\"`<>]+",
    re.IGNORECASE,
)
_REPORT_RESULT = re.compile(
    r"\b(?:answer|describe|explain|extract|find|give|identify|list|report|summari[sz]e|tell|what|which|"
    r"co|jakie|które|ktore|opisz\w*|podaj\w*|powiedz\w*|stre[śs]c\w*|wyciagn\w*|wyciągn\w*|"
    r"wymien\w*|wymień\w*|znajd\w*|znale[źz]\w*)\b",
    re.IGNORECASE,
)
_PUBLIC_FACT_FIELDS = (
    ("count", re.compile(r"\b(?:how\s+many|number\s+of|count|ile)\b", re.IGNORECASE)),
    ("opening_hours", re.compile(
        r"\b(?:opening\s+hours?|business\s+hours?|hours?|"
        r"godzin\w*|otwar\w*)\b",
        re.IGNORECASE,
    )),
    ("address", re.compile(
        r"\b(?:where|address\w*|location\w*|gdzie|adres\w*|lokaliz\w*)\b",
        re.IGNORECASE,
    )),
    ("contact", re.compile(
        r"\b(?:contact\w*|phone\w*|telephone\w*|kontakt\w*|telefon\w*)\b",
        re.IGNORECASE,
    )),
)
_ADDRESS_EVIDENCE = re.compile(
    r"(?:\b\d{2}-\d{3}\b|"
    r"\b(?:address|adres|aleja|avenue|boulevard|location|osiedle|plac|road|"
    r"square|street|ulica|ul\.)\b[^\n]{0,100}\b\d{1,5}[a-z]?(?:/\d+)?\b|"
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż.-]{3,}(?:ska|owa|cka|ego|skiej|owej)\s+"
    r"\d{1,5}[a-z]?(?:/\d+)?\b)",
    re.IGNORECASE,
)
_CONTACT_EVIDENCE = re.compile(
    r"(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"(?:\+?\d[\d\s().-]{6,}\d))",
    re.IGNORECASE,
)
_OPENING_HOURS_EVIDENCE = re.compile(
    r"(?:\b(?:mon|tue|wed|thu|fri|sat|sun|pon|wt|śr|sr|czw|pt|sob|niedz)"
    r"[\wąćęłńóśźż.-]*\b[^\n]{0,80}\b(?:[01]?\d|2[0-3])[:.]\d{2}\b|"
    r"\b(?:[01]?\d|2[0-3])[:.]\d{2}\s*[-–—]\s*"
    r"(?:[01]?\d|2[0-3])[:.]\d{2}\b)",
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
_RAW_BROWSER_SCAFFOLD = re.compile(
    r"(?:\bgeneric\s*\[(?:active\]\s*)?\[?ref=|\[ref=[^\]]+\]|"
    r"\bcursor=(?:pointer|text)\b)",
    re.IGNORECASE,
)
_GROUNDING_ENTITY_STOPWORDS = {
    "another", "based", "boss", "first", "finally", "here", "however",
    "english", "lastly", "next", "okay", "paladyn", "please", "response",
    "second", "the", "therefore", "this", "third", "would",
}
_HTTP_URL = re.compile(r"https?://[^\s<>\[\](){}\"']+", re.IGNORECASE)


def _grounding_url_key(value: str) -> str:
    """Normalize an observed HTTP URL without weakening its host/path identity."""

    parsed = urlsplit(unescape(value.rstrip(".,;:!?")))
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    path = unquote(parsed.path).rstrip("/").casefold()
    return hostname + path if hostname else ""


def _grounding_entity_is_present(entity: str, grounding_text: str) -> bool:
    """Accept a source-backed entity, including the safe ``X-based`` form."""

    normalized = entity.casefold()
    if normalized in grounding_text:
        return True
    for suffix in ("-based", "_based"):
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)].strip("-_")
            # This permits ``Python-based`` when the source says ``Python`` or
            # an inflected form such as Polish ``Pythonie``. It does not grant
            # semantic equivalence to unrelated product names.
            return len(stem) >= 3 and stem in grounding_text
    return False


def _is_search_listing_url(url: str) -> bool:
    """Return whether a navigation target is a search/listing page, not a source."""

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/").casefold()
    if hostname.endswith("duckduckgo.com"):
        return True
    if hostname in {"google.com", "www.google.com", "bing.com", "www.bing.com"}:
        return path in {"", "/search"}
    if hostname in {"search.brave.com", "search.yahoo.com"}:
        return True
    return path == "/search"


def _snapshot_mentions_url(snapshot: str, url: str) -> bool:
    """Verify that a later navigation target was exposed by earlier evidence.

    Browser snapshots may render a full URL, a decoded redirect, or only a
    hostname plus path. Requiring both host and a non-root path prevents a model
    from satisfying discovery by guessing an unrelated address from memory.
    """

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if not hostname:
        return False
    haystack = unquote(unescape(snapshot)).casefold().replace("www.", "")
    path = unquote(parsed.path).rstrip("/").casefold()
    if hostname not in haystack:
        return False
    if not path:
        return True
    # Keep the host and path structurally connected. The previous loose
    # "somewhere nearby" test treated a generic `/blog` as discovered when the
    # listing only exposed `/pl/blog/specific-article`, allowing a model to
    # downgrade a real result into an unrelated landing page.
    return f"{hostname}{path}" in haystack


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Runtime-owned, model-independent definition of completion evidence."""

    requires_browser_navigation: bool = False
    requires_browser_snapshot: bool = False
    requires_web_discovery: bool = False
    requires_distinct_detail_page: bool = False
    requires_file_read: bool = False
    requires_file_mutation: bool = False
    requires_command_execution: bool = False
    requires_evidence_report: bool = False
    requires_first_heading: bool = False
    requires_created_tool: bool = False
    requires_created_tool_execution: bool = False
    requires_created_skill: bool = False
    allows_artifact_fallback: bool = False
    requires_runtime_review: bool = False
    required_tools: tuple[str, ...] = ()
    required_public_fields: tuple[str, ...] = ()
    required_public_subject: str = ""

    @staticmethod
    def needs_web_discovery(prompt: str) -> bool:
        """Return whether online routing must discover a URL from search results."""

        return extract_web_target(prompt) is None

    @staticmethod
    def disables_web(prompt: str) -> bool:
        """Return a hard owner constraint that semantic routing cannot weaken."""

        return bool(_DISABLE_WEB.search(prompt))

    @staticmethod
    def has_explicit_local_file_target(prompt: str) -> bool:
        """Return whether the message names a file-like object or local path.

        Extensions and path syntax are structural across human languages. This
        lets semantic routing support multilingual file requests without letting
        a model turn an abstract word such as "plan" into an invented plan.txt.
        """

        without_urls = re.sub(
            r"https?://[^\s<>]+",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        return bool(
            _FILE_TARGET.search(without_urls)
            or _EXPLICIT_LOCAL_PATH.search(without_urls)
        )

    def without_web(self) -> "TaskContract":
        """Remove network requirements while preserving all local task evidence."""

        return replace(
            self,
            requires_browser_navigation=False,
            requires_browser_snapshot=False,
            requires_web_discovery=False,
            requires_distinct_detail_page=False,
            required_tools=tuple(
                name
                for name in self.required_tools
                if name not in {"full_tor_search", "full_tor_fetch"}
            ),
            required_public_fields=(),
            required_public_subject="",
        )

    @staticmethod
    def prefers_tor(prompt: str) -> bool:
        """Return whether the requested online surface is Tor rather than clearnet."""

        return bool(_TOR_ACCESS_REQUEST.search(prompt)) and not (
            TaskContract.disables_web(prompt)
        )

    @staticmethod
    def implies_artifact_discovery(prompt: str) -> bool:
        """Recognize a search attempt preceding conditional artifact creation.

        This is a model-independent recovery path for an intent classifier that
        returns contradictory empty capabilities. Explicit file and command
        targets remain local rather than being silently converted into web work.
        """

        without_urls = re.sub(
            r"https?://[^\s<>]+",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        local_file_work = bool(_FILE_TARGET.search(without_urls))
        local_command_work = bool(
            _RUN_COMMAND.search(prompt) and _COMMAND_TARGET.search(prompt)
        )
        return bool(_ONLINE_ACTION.search(prompt)) and not (
            local_file_work or local_command_work
        )

    @staticmethod
    def requested_public_fields(prompt: str) -> tuple[str, ...]:
        """Return concrete public-data fields expressed in any routed request."""

        return tuple(
            name
            for name, pattern in _PUBLIC_FACT_FIELDS
            if pattern.search(prompt) is not None
        )

    @staticmethod
    def implies_public_web_lookup(prompt: str) -> bool:
        """Recover a concrete public-fact lookup misread as ordinary chat.

        The multilingual semantic router remains the primary path. This narrow
        deterministic fallback requires an action plus at least two independent
        externally verifiable fields, such as count, opening hours, address, or
        contact details. It therefore does not turn a normal question or a
        creative writing request into browser work.
        """

        return bool(_ONLINE_ACTION.search(prompt)) and len(
            TaskContract.requested_public_fields(prompt)
        ) >= 2

    @classmethod
    def from_prompt(cls, prompt: str) -> "TaskContract":
        web_disabled = bool(_DISABLE_WEB.search(prompt))
        tor_requested = not web_disabled and bool(_TOR_ACCESS_REQUEST.search(prompt)) and (
            bool(_ONLINE_ACTION.search(prompt))
            or bool(_EXPLICIT_ONION_ADDRESS.search(prompt))
        )
        online = not tor_requested and not web_disabled and (
            requests_web_access(prompt)
            or bool(_ONLINE_ACTION.search(prompt) and _ONLINE_RESOURCE.search(prompt))
        )
        file_prompt = re.sub(r"https?://[^\s<>]+", "", prompt, flags=re.IGNORECASE)
        explicit_file_target = cls.has_explicit_local_file_target(file_prompt)
        file_read = bool(_READ_FILE.search(file_prompt) and explicit_file_target)
        file_mutation = bool(
            _MUTATE_FILE.search(file_prompt) and explicit_file_target
        )
        command_execution = bool(
            not online and _RUN_COMMAND.search(prompt) and _COMMAND_TARGET.search(prompt)
        )
        tool_creation_match = _search_outside_quoted_text(_CREATE_TOOL, prompt)
        skill_creation_match = _search_outside_quoted_text(_CREATE_SKILL, prompt)
        creation_matches = [
            match
            for match in (tool_creation_match, skill_creation_match)
            if match is not None
        ]
        primary_creation = min(
            creation_matches,
            key=lambda match: match.start(),
            default=None,
        )
        conditional_artifact = False
        if primary_creation is not None:
            before_creation = prompt[
                max(0, primary_creation.start() - 280) : primary_creation.start()
            ]
            after_creation = prompt[primary_creation.end() :]
            conditional_artifact = bool(
                _CONDITIONAL_FALLBACK.search(before_creation)
                or re.match(
                    r"[\s,;:-]*(?:only\s+)?(?:if|when|unless|"
                    r"je(?:sli|śli|zeli|żeli)|gdy|wenn|falls|si|se|"
                    r"ha|amennyiben|pokud|ak|если|якщо)\b",
                    after_creation,
                    re.IGNORECASE,
                )
            )
        creates_tool = bool(tool_creation_match) and not conditional_artifact
        runtime_review = bool(_RUNTIME_REVIEW.search(prompt))
        web_discovery = online and cls.needs_web_discovery(prompt)
        evidence_report = runtime_review or (
            (online or tor_requested or file_read or command_execution)
            and bool(_REPORT_RESULT.search(prompt))
        )
        public_fields = (
            cls.requested_public_fields(prompt)
            if online and evidence_report
            else ()
        )
        return cls(
            requires_browser_navigation=online,
            requires_browser_snapshot=online,
            # When Boss asks for online work without supplying an address, the
            # runtime must discover a real URL before visiting candidate sites.
            # This structural rule is language-independent and prevents a local
            # model from turning remembered names into invented domains.
            requires_web_discovery=web_discovery,
            # A research report based only on a search-results listing is not
            # research. Discovery must open and observe at least one real source.
            requires_distinct_detail_page=online and (
                bool(_DETAIL_PAGE.search(prompt))
                or (web_discovery and evidence_report)
            ),
            requires_file_read=file_read,
            requires_file_mutation=file_mutation,
            requires_command_execution=command_execution,
            requires_evidence_report=evidence_report,
            requires_first_heading=file_read and bool(_FIRST_HEADING.search(prompt)),
            requires_created_tool=creates_tool,
            requires_created_tool_execution=(
                creates_tool and bool(_USE_CREATED_TOOL.search(prompt))
            ),
            requires_created_skill=(
                bool(skill_creation_match) and not conditional_artifact
            ),
            allows_artifact_fallback=conditional_artifact,
            requires_runtime_review=runtime_review,
            required_tools=(
                (
                    "full_tor_fetch"
                    if _EXPLICIT_ONION_ADDRESS.search(prompt)
                    else "full_tor_search"
                ),
            )
            if tor_requested
            else (),
            required_public_fields=public_fields,
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "TaskContract":
        """Rebuild a contract from runtime-owned checkpoint data."""

        source = values if isinstance(values, dict) else {}
        tuple_fields = {"required_tools", "required_public_fields"}
        special_fields = {*tuple_fields, "required_public_subject"}
        flags = {
            name: bool(source.get(name, False))
            for name in cls.__dataclass_fields__
            if name not in special_fields
        }
        raw_tools = source.get("required_tools", [])
        required_tools = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_tools
                    if str(item).strip()
                )
            )
            if isinstance(raw_tools, (list, tuple))
            else ()
        )
        raw_public_fields = source.get("required_public_fields", [])
        required_public_fields = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_public_fields
                    if str(item).strip()
                    in {"count", "opening_hours", "address", "contact"}
                )
            )
            if isinstance(raw_public_fields, (list, tuple))
            else ()
        )
        return cls(
            **flags,
            required_tools=required_tools,
            required_public_fields=required_public_fields,
            required_public_subject=str(
                source.get("required_public_subject", "")
            ).strip()[:160],
        )

    def merged(self, other: "TaskContract") -> "TaskContract":
        """Return the union of two independently detected requirements."""

        tuple_fields = {"required_tools", "required_public_fields"}
        special_fields = {*tuple_fields, "required_public_subject"}
        flags = {
            name: bool(getattr(self, name) or getattr(other, name))
            for name in self.__dataclass_fields__
            if name not in special_fields
        }
        required_tools = tuple(
            dict.fromkeys((*self.required_tools, *other.required_tools))
        )
        required_public_fields = tuple(
            dict.fromkeys(
                (*self.required_public_fields, *other.required_public_fields)
            )
        )
        return type(self)(
            **flags,
            required_tools=required_tools,
            required_public_fields=required_public_fields,
            required_public_subject=(
                self.required_public_subject or other.required_public_subject
            ),
        )

    def with_required_tools(self, names: list[str] | tuple[str, ...]) -> "TaskContract":
        return type(self)(
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "required_tools",
                    "required_public_fields",
                    "required_public_subject",
                }
            },
            required_tools=tuple(
                dict.fromkeys(
                    (*self.required_tools, *(item for item in names if item))
                )
            ),
            required_public_fields=self.required_public_fields,
            required_public_subject=self.required_public_subject,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def unmet(self, calls: list[dict[str, Any]]) -> list[str]:
        succeeded = [call for call in calls if call.get("status", "succeeded") == "succeeded"]
        names = [str(call.get("tool", "")) for call in succeeded]
        missing: list[str] = []
        tool_builders = {
            "learning_create_tool",
            "learning_create_snapshot_extractor",
        }
        artifact_fallback_completed = self.allows_artifact_fallback and any(
            name in {*tool_builders, "learning_create_skill"}
            for name in names
        )

        if self.requires_browser_navigation and not any(
            name in {"browser_navigate", "web_search", "web_read"}
            for name in names
        ):
            missing.append("browser_navigate")
        if self.requires_browser_snapshot and not any(
            name in {"browser_snapshot", "web_search", "web_read"}
            for name in names
        ):
            missing.append("browser_snapshot")

        if self.requires_distinct_detail_page and not artifact_fallback_completed:
            verified_web_read = False
            for read_index, call in enumerate(succeeded):
                if call.get("tool") != "web_read":
                    continue
                arguments = call.get("arguments", {})
                url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                if not url:
                    continue
                if not self.requires_web_discovery:
                    verified_web_read = True
                    break
                if any(
                    prior.get("tool") == "web_search"
                    and search_index < read_index
                    and _snapshot_mentions_url(
                        str(prior.get("result_excerpt", "")),
                        url,
                    )
                    for search_index, prior in enumerate(succeeded)
                ):
                    verified_web_read = True
                    break

            if verified_web_read:
                detail_navigation = (0, "web_read")
            else:
                detail_navigation = None
            navigations: list[tuple[int, str]] = []
            snapshots: list[tuple[int, str]] = []
            for index, call in enumerate(succeeded):
                name = str(call.get("tool", ""))
                if name == "browser_navigate":
                    arguments = call.get("arguments", {})
                    url = str(arguments.get("url", "")) if isinstance(arguments, dict) else ""
                    if url and all(previous_url != url for _, previous_url in navigations):
                        navigations.append((index, url))
                elif name in {"browser_snapshot", "web_search"}:
                    snapshots.append((index, str(call.get("result_excerpt", ""))))
            if detail_navigation is not None:
                pass
            elif self.requires_web_discovery:
                # Freeze discovery evidence at the moment the model first leaves
                # the search/listing stage. A guessed or malformed detail URL can
                # expose more links on an unrelated landing page; those later
                # links must not retroactively become "discovered" evidence.
                # A subsequent corrected navigation is still accepted when its
                # exact URL was present in the original search-stage snapshots.
                first_detail_attempt = next(
                    (
                        index
                        for index, url in navigations
                        if not _is_search_listing_url(url)
                    ),
                    len(succeeded),
                )
                discovery_snapshots = [
                    (index, text)
                    for index, text in snapshots
                    if index < first_detail_attempt
                ]
                detail_navigation = next(
                    (
                        (index, url)
                        for index, url in navigations
                        if not _is_search_listing_url(url)
                        and any(
                            snapshot_index < index
                            and _snapshot_mentions_url(snapshot_text, url)
                            for snapshot_index, snapshot_text in discovery_snapshots
                        )
                    ),
                    None,
                )
            else:
                detail_navigation = navigations[1] if len(navigations) >= 2 else None
            if detail_navigation is None:
                if self.requires_web_discovery and any(
                    not _is_search_listing_url(url) for _, url in navigations
                ):
                    missing.append("browser_navigate:detail_not_discovered")
                else:
                    missing.append("browser_navigate:distinct_detail_page")
            elif not verified_web_read and not any(
                index > detail_navigation[0] for index, _ in snapshots
            ):
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

        if self.requires_created_tool and not any(
            name in tool_builders for name in names
        ):
            missing.append("learning_create_tool")
        if self.requires_created_tool_execution:
            created_name = ""
            created_index = -1
            for index, call in enumerate(succeeded):
                if call.get("tool") not in tool_builders:
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
        if self.requires_runtime_review and "runtime_review_task" not in names:
            missing.append("runtime_review_task")
        for required_tool in self.required_tools:
            if required_tool not in names:
                missing.append(required_tool)

        public_sources: list[str] = []
        for call in succeeded:
            tool = call.get("tool")
            excerpt = str(call.get("result_excerpt", ""))
            if tool == "web_read":
                public_sources.append(excerpt)
                continue
            if tool != "browser_snapshot":
                continue
            page_url = re.search(
                r"^- Page URL:\s*(\S+)",
                excerpt,
                re.MULTILINE,
            )
            # Search listings contain the requested words plus ranks and result
            # counts. Treating that plumbing as source evidence made a query such
            # as "address ..." followed by result_count=10 look like an address.
            # Only an opened non-search page may satisfy public fact fields.
            if page_url is not None and not _is_search_listing_url(page_url.group(1)):
                public_sources.append(excerpt)
        public_evidence = "\n".join(public_sources)
        if self.required_public_subject:
            subject_tokens = {
                token.casefold()
                for token in re.findall(
                    r"[^\W_]+(?:[-'][^\W_]+)*",
                    self.required_public_subject,
                    re.UNICODE,
                )
                if len(token) >= 3
            }
            evidence_folded = public_evidence.casefold()
            if subject_tokens and not all(
                token in evidence_folded for token in subject_tokens
            ):
                missing.append("public_fact:subject")
        address_found = bool(_ADDRESS_EVIDENCE.search(public_evidence))
        if "address" in self.required_public_fields and not address_found:
            missing.append("public_fact:address")
        if (
            "contact" in self.required_public_fields
            and not _CONTACT_EVIDENCE.search(public_evidence)
        ):
            missing.append("public_fact:contact")
        if (
            "opening_hours" in self.required_public_fields
            and not _OPENING_HOURS_EVIDENCE.search(public_evidence)
        ):
            missing.append("public_fact:opening_hours")
        if (
            "count" in self.required_public_fields
            and "address" not in self.required_public_fields
            and not re.search(
                r"\b\d+\s+(?:branches|locations|shops|stores|"
                r"cukierni\w*|lokal\w*|oddzia\w*|plac[oó]w\w*)\b",
                public_evidence,
                re.IGNORECASE,
            )
        ):
            missing.append("public_fact:count")
        return missing

    def answer_issues(
        self,
        answer: str,
        calls: list[dict[str, Any]],
        *,
        request: str = "",
    ) -> list[str]:
        """Reject an empty completion claim that does not use observed evidence."""

        if not self.requires_evidence_report:
            return []
        observations = [
            str(call.get("result_excerpt", ""))
            for call in calls
            if call.get("status", "succeeded") == "succeeded"
            and call.get("tool") in {
                "browser_snapshot",
                "web_search",
                "web_read",
                "read_file",
                "cat",
                "sandbox_execute_offline",
                "evm_foundry_test_offline",
                "runtime_review_task",
                "full_tor_search",
                "full_tor_fetch",
            }
            and call.get("result_excerpt")
        ]
        if not observations:
            return ["answer:evidence_observation_missing"]

        if _RAW_BROWSER_SCAFFOLD.search(answer):
            return ["answer:browser_scaffolding_is_not_a_finding"]

        if self.requires_browser_navigation or any(
            name in {"full_tor_search", "full_tor_fetch"}
            for name in self.required_tools
        ):
            grounding_text = (request + "\n" + "\n".join(observations)).casefold()
            observed_url_keys = {
                key
                for key in (
                    _grounding_url_key(url)
                    for observation in observations
                    for url in _HTTP_URL.findall(observation)
                )
                if key
            }
            ungrounded_urls = sorted(
                {
                    url.rstrip(".,;:!?")
                    for url in _HTTP_URL.findall(answer)
                    if _grounding_url_key(url) not in observed_url_keys
                }
            )
            if ungrounded_urls:
                return [
                    "answer:ungrounded_online_urls="
                    + "|".join(ungrounded_urls[:6])
                ]
            claimed_entities: set[str] = set()
            for highlighted in re.findall(
                r"\*\*([^*\n]{2,120})\*\*|`([^`\n]{2,120})`",
                answer,
            ):
                phrase = next((item for item in highlighted if item), "")
                claimed_entities.update(
                    token
                    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{2,}", phrase)
                    if token.casefold() not in _GROUNDING_ENTITY_STOPWORDS
                )
            for match in re.finditer(r"\b[A-Z][A-Za-z0-9.+_-]{2,}\b", answer):
                token = match.group(0)
                prefix = answer[: match.start()].rstrip()
                if not prefix or prefix[-1:] in {".", "!", "?", "\n"}:
                    continue
                if re.search(
                    r"\b(?:hey|hello|okay|ok|sorry|thanks)[,\s]+$",
                    prefix[-32:],
                    flags=re.IGNORECASE,
                ):
                    # A direct form of address is relationship prose, not an
                    # online product/entity claim that needs source grounding.
                    continue
                if token.casefold() not in _GROUNDING_ENTITY_STOPWORDS:
                    claimed_entities.add(token)
            ungrounded = sorted(
                entity
                for entity in claimed_entities
                if not _grounding_entity_is_present(entity, grounding_text)
            )
            if ungrounded:
                return [
                    "answer:ungrounded_online_claims="
                    + "|".join(ungrounded[:12])
                ]

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

        if self.requires_created_tool_execution:
            builders = {
                "learning_create_tool",
                "learning_create_snapshot_extractor",
            }
            created_name = ""
            creation_payload: dict[str, Any] = {}
            created_index = -1
            for index, call in enumerate(calls):
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool") not in builders
                ):
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("name"):
                    created_name = str(payload["name"])
                    creation_payload = payload
                    created_index = index
            if created_name:
                for index in range(len(calls) - 1, created_index, -1):
                    call = calls[index]
                    if (
                        call.get("status", "succeeded") != "succeeded"
                        or call.get("tool") != created_name
                    ):
                        continue
                    try:
                        result = json.loads(str(call.get("result_excerpt", "")))
                    except (TypeError, json.JSONDecodeError):
                        return None
                    if not isinstance(result, dict):
                        return None
                    validation = creation_payload.get("validation", {})
                    lifecycle_verified = (
                        str(creation_payload.get("status", "")).casefold()
                        == "active"
                        and isinstance(validation, dict)
                        and validation.get("passed") is True
                    )
                    lifecycle = (
                        "validated and activated"
                        if lifecycle_verified
                        else "created and activated"
                    )
                    records = result.get("records")
                    if isinstance(records, list) and records:
                        lines = [
                            f"Done. PALADYN {lifecycle} `{created_name}`, then "
                            "executed it on the runtime-observed page data.",
                            "",
                            "Verified first three records:",
                        ]
                        for number, record in enumerate(records[:3], start=1):
                            if not isinstance(record, dict):
                                return None
                            ordered_keys = [
                                key
                                for key in (
                                    "title",
                                    "price",
                                    "availability",
                                    "relative_product_url",
                                )
                                if key in record
                            ]
                            ordered_keys.extend(
                                key for key in record if key not in ordered_keys
                            )
                            fields = "; ".join(
                                f"{key}: {record[key]}"
                                for key in ordered_keys
                            )
                            lines.append(f"{number}. {fields}")
                        return "\n".join(lines)
                    return (
                        f"Done. PALADYN {lifecycle} `{created_name}`, then "
                        "executed it successfully. Verified result:\n\n"
                        + json.dumps(result, ensure_ascii=False, sort_keys=True)
                    )

        if self.requires_created_tool:
            for call in reversed(calls):
                if (
                    call.get("status", "succeeded") != "succeeded"
                    or call.get("tool")
                    not in {
                        "learning_create_tool",
                        "learning_create_snapshot_extractor",
                    }
                ):
                    continue
                try:
                    payload = json.loads(str(call.get("result_excerpt", "")))
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict) or not payload.get("name"):
                    continue
                validation = payload.get("validation", {})
                verified = (
                    str(payload.get("status", "")).casefold() == "active"
                    and isinstance(validation, dict)
                    and validation.get("passed") is True
                )
                if not verified:
                    continue
                tests = validation.get("tests", [])
                test_names = [
                    str(item.get("name", ""))
                    for item in tests
                    if isinstance(item, dict) and item.get("passed") is True
                ]
                test_report = (
                    f" Validation: {', '.join(test_names)}."
                    if test_names
                    else " Validation passed in the offline sandbox."
                )
                return (
                    f"Done. PALADYN built the generated tool, validated, and activated "
                    f"`{payload['name']}` from the generated source."
                    + test_report
                )

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
