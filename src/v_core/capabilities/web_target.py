from __future__ import annotations

import re
from urllib.parse import urlsplit


_FULL_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_BARE_DOMAIN = re.compile(
    r"(?<![@\w])"
    r"(?:www\.)?"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}"
    r"(?::[0-9]{1,5})?"
    r"(?:/[^\s<>]*)?",
    re.IGNORECASE,
)
_WEB_INTENT = re.compile(
    r"\b(?:"
    r"browse|check|explore|inspect|navigate|open|research|review|site|visit|website|"
    r"przejrzyj|sprawdz|sprawdź|strona|strone|stronę|wejdz|wejdź|witryna"
    r")\b",
    re.IGNORECASE,
)

# Speech-to-text engines commonly spell URL punctuation out instead of
# preserving it. Keep this table deliberately small and structural: it maps
# punctuation, never names or likely destinations. Additional languages can be
# added without changing the routing code.
_SPOKEN_URL_SEPARATORS = {
    "colon": ":",
    "dwukropek": ":",
    "doppelpunkt": ":",
    "slash": "/",
    "ukosnik": "/",
    "ukośnik": "/",
    "lamane": "/",
    "łamane": "/",
    "lamany": "/",
    "łamany": "/",
    "dot": ".",
    "period": ".",
    "punkt": ".",
    "kropka": ".",
    "dash": "-",
    "hyphen": "-",
    "minus": "-",
    "myslnik": "-",
    "myślnik": "-",
}
_SPOKEN_URL_TOKEN = re.compile(
    r"https?|[^\W_]+|[:/.-]",
    re.IGNORECASE | re.UNICODE,
)


def _extract_spoken_web_target(prompt: str) -> str | None:
    """Rebuild an HTTP(S) URL whose punctuation was spoken aloud.

    A syntactically valid host is required, so ordinary sentences mentioning
    ``HTTP`` do not become browser actions. Sentence-ending punctuation after
    the final host token terminates the reconstruction instead of leaking the
    following instruction into the URL.
    """

    matches = list(_SPOKEN_URL_TOKEN.finditer(prompt))
    start = next(
        (
            index
            for index, match in enumerate(matches)
            if match.group(0).casefold() in {"http", "https"}
        ),
        None,
    )
    if start is None:
        return None

    scheme = matches[start].group(0).casefold()
    rendered = scheme
    saw_spoken_separator = False
    for match in matches[start + 1 : start + 64]:
        token = match.group(0)
        folded = token.casefold()
        separator = _SPOKEN_URL_SEPARATORS.get(folded)
        if separator is not None:
            rendered += separator
            saw_spoken_separator = True
            continue
        if token in {":", "/", ".", "-"}:
            rendered += token
            continue

        # A URL dictated as a sentence normally ends with a literal period.
        # Stop there once the reconstructed value already has a real host.
        rendered += folded
        following = prompt[match.end() :]
        if re.match(r"\s*[.!?](?:\s|$)", following):
            try:
                parsed = urlsplit(rendered)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.hostname and "." in parsed.hostname:
                break

    if not saw_spoken_separator:
        return None
    try:
        parsed = urlsplit(rendered)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or "." not in parsed.hostname
        or any(character.isspace() for character in rendered)
    ):
        return None
    return rendered


def extract_web_target(prompt: str) -> str | None:
    full = _FULL_URL.search(prompt)
    if full is not None:
        return full.group(0).rstrip(").,;]}\"'")

    spoken = _extract_spoken_web_target(prompt)
    if spoken is not None:
        return spoken

    bare = _BARE_DOMAIN.search(prompt)
    if bare is None:
        return None
    target = bare.group(0).rstrip(").,;]}\"'")
    return f"https://{target}"


def requests_web_access(prompt: str) -> bool:
    if _FULL_URL.search(prompt) or _extract_spoken_web_target(prompt) is not None:
        return True
    return bool(_WEB_INTENT.search(prompt) and _BARE_DOMAIN.search(prompt))
