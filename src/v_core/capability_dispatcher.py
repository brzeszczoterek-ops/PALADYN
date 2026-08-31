from __future__ import annotations

from .autonomy.task_contract import TaskContract
from .capabilities.web_target import requests_web_access


class CapabilityDispatcher:

    RESEARCH = "research"
    TOOLS = "tools"
    CHAT = "chat"

    def dispatch(
        self,
        prompt: str,
    ) -> str:

        contract = TaskContract.from_prompt(prompt)
        requires_orchestration = bool(
            contract.requires_file_read
            or contract.requires_file_mutation
            or contract.requires_command_execution
            or contract.requires_created_tool
            or contract.requires_created_tool_execution
            or contract.requires_created_skill
            or contract.allows_artifact_fallback
            or contract.requires_runtime_review
            or contract.required_tools
        )

        # A direct URL-only inspection can use the compact research path. Mixed
        # work must stay in the multi-step agent loop; otherwise browsing works
        # but tool/skill creation schemas are never exposed to V.
        if requests_web_access(prompt) and not requires_orchestration:
            return self.RESEARCH

        # Every other request uses the multi-step agent loop. It can answer a
        # capability question without touching a tool, or request one or more
        # real tools through the JSON protocol. The former keyword router sent
        # questions such as "can you create tools?" into a brittle YES/NO
        # dispatcher and lost both the answer and execution trace.
        return self.CHAT
