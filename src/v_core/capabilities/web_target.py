from __future__ import annotations

import re


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


def extract_web_target(prompt: str) -> str | None:
    full = _FULL_URL.search(prompt)
    if full is not None:
        return full.group(0).rstrip(").,;]}\"'")

    bare = _BARE_DOMAIN.search(prompt)
    if bare is None:
        return None
    target = bare.group(0).rstrip(").,;]}\"'")
    return f"https://{target}"


def requests_web_access(prompt: str) -> bool:
    if _FULL_URL.search(prompt):
        return True
    return bool(_WEB_INTENT.search(prompt) and _BARE_DOMAIN.search(prompt))
