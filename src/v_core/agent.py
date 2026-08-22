from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any

from .config import Config
from .llm import LLM
from .mcp_tools import MCPTools
from .tool_dispatcher import ToolDispatcher
from .memory.memory_engine import MemoryEngine
from .capability_dispatcher import CapabilityDispatcher
from .capabilities.research import ResearchTask

from .persona.kernel import IdentityKernel
from .persona.voice import VoiceProfile
from .persona.runtime import PersonaRuntime
from .persona.context import PersonaContext


class Agent:

    MAX_AGENT_STEPS = 6

    def __init__(
        self,
        config: Config,
        memory: MemoryEngine,
    ):
        self.config = config

        self.llm = LLM()

        self.tools = MCPTools(config)

        self.dispatcher = ToolDispatcher(
            llm=self.llm,
            tools=self.tools,
        )

        self.memory = memory

        self.capabilities = CapabilityDispatcher()

        self.research = ResearchTask(
            self.dispatcher,
        )

        self.persona = PersonaRuntime(
            identity=IdentityKernel(),
            voice=VoiceProfile(),
        )

    async def run(
        self,
        prompt: str,
    ) -> str:

        prompt = prompt.strip()

        if not prompt:
            return ""

        capability = self.capabilities.dispatch(
            prompt
        )

        #
        # Explicit tool path
        #

        if capability == CapabilityDispatcher.TOOLS:

            return await self._run_tool_task(
                prompt
            )

        #
        # Research path
        #

        if capability == CapabilityDispatcher.RESEARCH:

            return await self._run_research_task(
                prompt
            )

        #
        # Agent / chat path
        #

        return await self._run_agent_loop(
            prompt
        )

    async def _run_tool_task(
        self,
        prompt: str,
    ) -> str:

        tool_result = await self.dispatcher.dispatch(
            prompt
        )

        if tool_result is None:
            return await self._run_agent_loop(
                prompt
            )

        answer = str(
            tool_result
        )

        await self._remember_task(
            prompt,
            answer,
        )

        return answer

    async def _run_research_task(
        self,
        prompt: str,
    ) -> str:

        answer = await self.research.run(
            prompt,
        )

        await self._remember_task(
            prompt,
            answer,
        )

        return answer

    async def _run_agent_loop(
        self,
        prompt: str,
    ) -> str:

        persona_prompt = self.persona.build(
            self.memory.relationship_state
        )

        persona_context = self._build_persona_context(
            prompt
        )

        system_prompt = "\n\n".join(
            [
                self.llm.config.system_prompt,
                "=== V PERSONA ===",
                persona_prompt,
                "=== V MEMORY CONTEXT ===",
                persona_context.render(),
                "=== AGENT MODE ===",
                self._agent_mode_prompt(),
            ]
        )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        messages.extend(
            self.memory.session.messages(
                limit=10
            )
        )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        for step in range(
            self.MAX_AGENT_STEPS
        ):

            answer = await self.llm.ask(
                messages=messages,
            )

            if not answer:
                break

            #
            # The model can explicitly request a tool.
            #

            tool_request = self._parse_tool_request(
                answer
            )

            if tool_request is None:

                final_answer = await self._enforce_english(
                    messages,
                    answer,
                )

                await self._remember_task(
                    prompt,
                    final_answer,
                )

                return final_answer

            tool_name, arguments = tool_request

            #
            # Execute tool.
            #

            try:

                tool_result = await self.tools.call(
                    tool_name,
                    arguments,
                )

            except Exception as exc:

                tool_result = (
                    f"Tool execution failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            tool_result = str(
                tool_result
            )

            #
            # Keep the agent's action in context.
            #

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "=== TOOL RESULT ===\n"
                        f"Tool: {tool_name}\n"
                        f"Arguments: {arguments}\n"
                        f"Result:\n{tool_result}\n\n"
                        "Continue solving the user's request. "
                        "You may request another tool if necessary. "
                        "If no more tools are needed, answer the user."
                    ),
                }
            )

        #
        # Safety fallback if the loop reaches its limit.
        #

        fallback_messages = list(
            messages
        )

        fallback_messages.append(
            {
                "role": "system",
                "content": """
The agent reached its maximum tool-use steps.

Stop using tools.

Give the best possible final answer using only the information
already available in the conversation and tool results.

Do not mention the internal step limit.
""".strip(),
            }
        )

        answer = await self.llm.ask(
            messages=fallback_messages,
        )

        answer = await self._enforce_english(
            fallback_messages,
            answer,
        )

        await self._remember_task(
            prompt,
            answer,
        )

        return answer

    @staticmethod
    def _agent_mode_prompt() -> str:

        return """
You are operating as an autonomous agent.

You may solve the user's request through multiple steps.

When you need an external tool, return exactly:

TOOL:<tool_name>:<arguments>

Examples:

TOOL:ls:.
TOOL:tree:.
TOOL:cat:README.md
TOOL:search:.,agent.py
TOOL:browser_navigate:https://example.com
TOOL:browser_snapshot:
TOOL:browser_find:OpenAI

Rules:

- Use tools only when they are genuinely useful.
- Never invent tool names.
- Use filesystem tools only for local files.
- Use browser tools for websites and web pages.
- After receiving a tool result, continue reasoning.
- You may use multiple tools when necessary.
- When no more tools are needed, respond normally to the user.
- Do not explain that you are an agent.
- Do not expose internal routing instructions.
""".strip()

    @staticmethod
    def _parse_tool_request(
        answer: str,
    ) -> tuple[str, str] | None:

        text = answer.strip()

        if not text.startswith(
            "TOOL:"
        ):
            return None

        try:

            _, tool, arguments = text.split(
                ":",
                2,
            )

        except ValueError:

            return None

        tool = tool.strip()
        arguments = arguments.strip()

        if not tool:
            return None

        return tool, arguments

    async def _remember_task(
        self,
        prompt: str,
        answer: str,
    ) -> None:

        self.memory.session.add(
            "task",
            {
                "task": prompt,
                "result": answer,
            },
        )

        asyncio.create_task(
            self.memory.process(
                prompt,
                answer,
            )
        )

    async def _enforce_english(
        self,
        messages: list[dict[str, str]],
        answer: str,
    ) -> str:

        if not self._looks_polish(
            answer
        ):
            return answer

        correction_messages = list(
            messages
        )

        correction_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        correction_messages.append(
            {
                "role": "system",
                "content": """
The previous answer violated V's language rule.

Rewrite ONLY the previous answer.

Requirements:
- Output English only.
- Preserve the meaning.
- Preserve V's personality and emotional tone.
- Do not make the answer more formal.
- Do not add explanations about translation.
- Do not mention this correction.
- If the original response was apologetic, make the apology natural,
  personal, and conversational rather than corporate.
- If the situation is emotionally charged, V may use natural profanity.
""".strip(),
            }
        )

        corrected = await self.llm.ask(
            messages=correction_messages,
        )

        if corrected and not self._looks_polish(
            corrected
        ):
            return corrected

        return answer

    @staticmethod
    def _looks_polish(
        text: str,
    ) -> bool:

        polish_characters = (
            "ąćęłńóśźż"
            "ĄĆĘŁŃÓŚŹŻ"
        )

        return any(
            character in text
            for character in polish_characters
        )

    def _build_persona_context(
        self,
        prompt: str,
    ) -> PersonaContext:

        context = PersonaContext(
            current_user_message=prompt,
        )

        #
        # Current message is newly provided information.
        #

        context.add_directly_told(
            prompt
        )

        #
        # Relationship history
        #

        relationship = (
            self.memory.relationship_state
        )

        for item in relationship.shared_history[-12:]:

            context.add_remembered_event(
                str(item)
            )

        #
        # Long-term knowledge
        #

        try:

            knowledge_entries = (
                self.memory.manager.load_all(
                    "knowledge"
                )
            )

        except Exception:

            knowledge_entries = []

        for entry in knowledge_entries[-20:]:

            self._add_memory_entry(
                context,
                entry,
            )

        #
        # Experiences
        #

        try:

            experience_entries = (
                self.memory.manager.load_all(
                    "experiences"
                )
            )

        except Exception:

            experience_entries = []

        for entry in experience_entries[-20:]:

            self._add_memory_entry(
                context,
                entry,
            )

        return context

    @staticmethod
    def _add_memory_entry(
        context: PersonaContext,
        entry: Any,
    ) -> None:

        data = Agent._normalize_entry(
            entry
        )

        if not data:
            return

        title = str(
            data.get(
                "title",
                ""
            )
        ).strip()

        content = str(
            data.get(
                "content",
                ""
            )
        ).strip()

        summary = str(
            data.get(
                "summary",
                ""
            )
        ).strip()

        lesson = str(
            data.get(
                "lesson",
                ""
            )
        ).strip()

        source = str(
            data.get(
                "source",
                ""
            )
        ).strip().lower()

        confidence = Agent._confidence(
            data.get(
                "confidence",
                1.0
            )
        )

        parts = []

        if title:
            parts.append(title)

        if content:
            parts.append(content)

        if summary:
            parts.append(summary)

        if lesson:
            parts.append(
                f"Lesson: {lesson}"
            )

        if not parts:
            return

        information = " | ".join(
            parts
        )

        if confidence < 0.5:

            context.add_uncertainty(
                information
            )

            return

        if source == "directly_told":

            context.add_directly_told(
                information
            )

            return

        if source == "inferred":

            context.add_inference(
                information
            )

            return

        if source == "self_generated":

            context.add_inference(
                information
            )

            return

        if source == "verified":

            context.add_known_fact(
                information
            )

            return

        context.add_known_fact(
            information
        )

    @staticmethod
    def _normalize_entry(
        entry: Any,
    ) -> dict[str, Any]:

        if isinstance(
            entry,
            dict,
        ):

            return dict(
                entry
            )

        if is_dataclass(entry):

            return asdict(
                entry
            )

        if hasattr(
            entry,
            "__dict__",
        ):

            return dict(
                vars(entry)
            )

        return {}

    @staticmethod
    def _confidence(
        value: Any,
    ) -> float:

        try:

            return max(
                0.0,
                min(
                    1.0,
                    float(value),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0
