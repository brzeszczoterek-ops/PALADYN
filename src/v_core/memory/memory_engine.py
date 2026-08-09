from __future__ import annotations

from .session import Session
from .reflection import Reflection
from .experience import Experience
from .summary import Summary
from .knowledge import Knowledge
from .manager import MemoryManager


class MemoryEngine:

    def __init__(
        self,
        session: Session,
        reflection: Reflection,
        experience: Experience,
        summary: Summary,
        knowledge: Knowledge,
        manager: MemoryManager,
    ):

        self.session = session
        self.reflection = reflection
        self.experience = experience
        self.summary = summary
        self.knowledge = knowledge
        self.manager = manager

    async def process(
        self,
        task: str,
        result: str,
    ):

        self.session.add(
            "task",
            {
                "task": task,
                "result": result,
            },
        )

        reflection = None

        try:

            reflection = await self.reflection.reflect(
                task,
                result,
            )

            self.manager.remember(
                "reflections",
                reflection,
            )

        except Exception as e:

            print(f"[Memory] Reflection failed: {e}")

            return None

        try:

            previous_experience = self.manager.load_all(
                "experiences",
            )

            current_knowledge = self.manager.load_all(
                "knowledge",
            )

            experience = await self.experience.learn(
                reflection,
                previous_experience,
                current_knowledge,
            )

            self.manager.remember(
                "experiences",
                experience,
            )

            return experience

        except Exception as e:

            print(f"[Memory] Experience failed: {e}")

            return None
