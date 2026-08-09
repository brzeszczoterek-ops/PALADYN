from __future__ import annotations


class CapabilityDispatcher:

    RESEARCH = "research"
    TOOLS = "tools"
    CHAT = "chat"

    RESEARCH_KEYWORDS = (
        "research",
        "zbadaj",
        "przeanalizuj",
        "dowiedz",
        "znajdź informacje",
        "znajdz informacje",
        "wyszukaj informacje",
        "opowiedz o",
        "kim jest",
        "co to jest",
        "firma",
        "osoba",
        "miejscowość",
        "miejscowosc",
        "strona",
        "witryna",
        "www",
        "https://",
        "http://",
    )

    TOOL_KEYWORDS = (
        "plik",
        "folder",
        "katalog",
        "otwórz",
        "otworz",
        "przeczytaj",
        "zapisz",
        "edytuj",
        "usuń",
        "usun",
        "mkdir",
        "ls",
        "cat",
        "tree",
        "search",
        "move",
        "filesystem",
    )

    def dispatch(
        self,
        prompt: str,
    ) -> str:

        text = prompt.lower()

        if any(k in text for k in self.RESEARCH_KEYWORDS):
            return self.RESEARCH

        if any(k in text for k in self.TOOL_KEYWORDS):
            return self.TOOLS

        return self.CHAT
