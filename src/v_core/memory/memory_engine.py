from __future__ import annotations

from typing import Any

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
        *,
        execution: dict[str, Any] | None = None,
    ):

        # Stopped, failed, or blocked work remains available in the current
        # session and execution journal, but must never become durable memory.
        # Otherwise a rejected model claim can return later as an alleged fact.
        if execution is not None and execution.get("status") != "completed":
            return None

        try:

            if execution is None:
                reflection = await self.reflection.reflect(task, result)
            else:
                reflection = await self.reflection.reflect(
                    task,
                    result,
                    execution=execution,
                )

            stored_reflection = self.manager.remember(
                "reflections",
                reflection,
            )

            # `remember=False` is a hard stop for the complete persistence
            # pipeline, not merely a request to omit the reflection file.
            if stored_reflection is None:
                return None

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

            stored_experience = self.manager.remember(
                "experiences",
                experience,
            )

        except Exception as e:

            print(
                f"[Memory] Experience failed: {e}"
            )

            return None

        # A rejected or unreliable experience must not affect the persistent
        # relationship, summaries, or long-term knowledge.
        if stored_experience is None:
            return experience

        try:

            candidate_relationship = await self.relationship_updater.update(
                self.relationship_state,
                experience,
            )

            if candidate_relationship is not self.relationship_state:
                self.relationship_storage.save(candidate_relationship)
                self.relationship_state = candidate_relationship

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
