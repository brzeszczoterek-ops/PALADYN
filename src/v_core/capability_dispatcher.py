from __future__ import annotations

from .capabilities.web_target import requests_web_access


class CapabilityDispatcher:

    RESEARCH = "research"
    TOOLS = "tools"
    CHAT = "chat"

    def dispatch(
        self,
        prompt: str,
    ) -> str:

        # The specialized research path currently requires a concrete URL.
        # General knowledge questions belong in the agent loop instead of
        # failing with an unrelated request for a link.
        if requests_web_access(prompt):
            return self.RESEARCH

        # Every other request uses the multi-step agent loop. It can answer a
        # capability question without touching a tool, or request one or more
        # real tools through the JSON protocol. The former keyword router sent
        # questions such as "can you create tools?" into a brittle YES/NO
        # dispatcher and lost both the answer and execution trace.
        return self.CHAT
