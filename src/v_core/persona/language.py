from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect_langs


# langdetect uses random sampling internally. A fixed seed keeps PALADYN's
# language gate deterministic across runs and test environments.
DetectorFactory.seed = 0


_POLISH_CHARACTERS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

_POLISH_WORDS = frozenset(
    {
        "ale",
        "bardzo",
        "bedzie",
        "będzie",
        "chcesz",
        "ci",
        "co",
        "czesc",
        "cześć",
        "czy",
        "dla",
        "dobrze",
        "dzisiaj",
        "hej",
        "jak",
        "jest",
        "moge",
        "mogę",
        "mozemy",
        "możemy",
        "nie",
        "pomoc",
        "pomóc",
        "tak",
        "witaj",
        "zrobic",
        "zrobić",
    }
)

_NON_ENGLISH_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:answer|reply|respond|speak|write|continue)\s+"
        r"(?:to\s+me\s+)?(?:in|using)\s+(?!english\b)[a-z-]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:odpowiedz|odpowiadaj|pisz|napisz|mow|mów|rozmawiaj)\b"
        r".{0,32}\bpo\s+(?!angielsku\b)[a-ząćęłńóśźż-]+",
        re.IGNORECASE,
    ),
)

_ENGLISH_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:answer|reply|respond|speak|write|continue)\s+"
        r"(?:to\s+me\s+)?(?:in|using)\s+english\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:odpowiedz|odpowiadaj|pisz|napisz|mow|mów|rozmawiaj)\b"
        r".{0,32}\bpo\s+angielsku\b",
        re.IGNORECASE,
    ),
)

_USER_ENGLISH_DEMAND_PATTERNS = (
    re.compile(
        r"\bplease\s+(?:write|speak|reply|respond|ask|talk|communicate)\b"
        r".{0,48}\bin\s+english\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:can|could|would|will)\s+you\b.{0,64}"
        r"\b(?:write|speak|reply|respond|ask|talk|communicate|use)\b"
        r".{0,48}\benglish\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.!?]\s*)use\s+english\b",
        re.IGNORECASE,
    ),
)


def explicitly_requests_non_english(prompt: str) -> bool:
    """Return true only for an explicit instruction to change language.

    Merely writing in another language or mentioning a language is not enough.
    English wins if the prompt contains conflicting language instructions.
    """

    text = prompt.strip()
    if not text or any(pattern.search(text) for pattern in _ENGLISH_REQUEST_PATTERNS):
        return False
    return any(pattern.search(text) for pattern in _NON_ENGLISH_REQUEST_PATTERNS)


def asks_user_to_use_english(text: str) -> bool:
    """Detect an impermissible demand that Boss switch input language."""

    return any(pattern.search(text) for pattern in _USER_ENGLISH_DEMAND_PATTERNS)


def looks_non_english(text: str) -> bool:
    """Conservatively detect natural-language output that is not English.

    Code, paths, identifiers, and very short neutral answers are intentionally
    not rejected. Polish receives an additional deterministic check so common
    replies without diacritics cannot bypass the gate.
    """

    prose = _prose_for_detection(text)
    if not prose:
        return False

    if any(character in _POLISH_CHARACTERS for character in prose):
        return True

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿąćęłńóśźż]+", prose.casefold())
    polish_hits = sum(word in _POLISH_WORDS for word in words)
    if polish_hits >= 2 or (polish_hits == 1 and len(words) <= 4):
        return True

    letters = "".join(character for character in prose if character.isalpha())
    if len(letters) < 12:
        return False

    try:
        candidates = detect_langs(prose)
    except LangDetectException:
        return False

    if not candidates:
        return False

    best = candidates[0]
    return best.lang != "en" and best.prob >= 0.70


def _prose_for_detection(text: str) -> str:
    without_fences = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_inline_code = re.sub(r"`[^`]*`", " ", without_fences)
    without_urls = re.sub(r"https?://\S+", " ", without_inline_code)
    without_paths = re.sub(r"(?:^|\s)(?:[./~][^\s]+)", " ", without_urls)
    return " ".join(without_paths.split())
