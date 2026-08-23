from __future__ import annotations


class CapabilityDispatcher:

    RESEARCH = "research"
    TOOLS = "tools"
    CHAT = "chat"

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

        # The specialized research path currently requires a concrete URL.
        # General knowledge questions belong in the agent loop instead of
        # failing with an unrelated request for a link.
        if "https://" in text or "http://" in text:
            return self.RESEARCH

        if any(k in text for k in self.TOOL_KEYWORDS):
            return self.TOOLS

        return self.CHAT
