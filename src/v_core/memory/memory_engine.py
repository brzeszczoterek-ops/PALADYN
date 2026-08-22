from __future__ import annotations

from .session import Session
from .reflection import Reflection
from .experience import Experience
from .summary import Summary
from .knowledge import Knowledge
from .manager import MemoryManager

from ..relationship import (
    RelationshipStorage,
    RelationshipUpdater,
)


class MemoryEngine:

    def __init__(
        self,
        session: Session,
        reflection: Reflection,
        experience: Experience,
        summary: Summary,
        knowledge: Knowledge,
        manager: MemoryManager,
        relationship_updater: RelationshipUpdater,
        relationship_storage: RelationshipStorage,
    ):

        self.session = session
        self.reflection = reflection
        self.experience = experience
        self.summary = summary
        self.knowledge = knowledge
        self.manager = manager

        self.relationship_updater = relationship_updater
        self.relationship_storage = relationship_storage
        self.relationship_state = (
            relationship_storage.load()
        )

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

            print(
                f"[Memory] Reflection failed: {e}"
            )

            return None

        try:

            previous_experience = (
                self.manager.load_all(
                    "experiences",
                )
            )

            current_knowledge = (
                self.manager.load_all(
                    "knowledge",
                )
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

        except Exception as e:

            print(
                f"[Memory] Experience failed: {e}"
            )

            return None

        try:

            self.relationship_state = (
                await self.relationship_updater.update(
                    self.relationship_state,
                    experience,
                )
            )

            self.relationship_storage.save(
                self.relationship_state,
            )

        except Exception as e:

            print(
                f"[Memory] Relationship update failed: {e}"
            )

        try:

            experiences = (
                self.manager.load_all(
                    "experiences",
                )
            )

            knowledge = (
                self.manager.load_all(
                    "knowledge",
                )
            )

            summary = await self.summary.summarize(
                experiences,
                knowledge,
            )

            self.manager.remember(
                "summaries",
                summary,
            )

        except Exception as e:

            print(
                f"[Memory] Summary failed: {e}"
            )

            return experience

        try:

            knowledge_entry = (
                await self.knowledge.update(
                    summary,
                    knowledge,
                )
            )

            self.manager.remember(
                "knowledge",
                knowledge_entry,
            )

        except Exception as e:

            print(
                f"[Memory] Knowledge update failed: {e}"
            )

        return experience
