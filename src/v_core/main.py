from __future__ import annotations

import asyncio

from .agent import Agent
from .autonomy import (
    AuthorizationEnvelope,
    AutonomousRunner,
    AutonomousTask,
    RuntimeRegistry,
)
from .autonomy.runner import StepDriver
from .config import load_config

from .memory.storage import MemoryStorage
from .memory.session import Session
from .memory.reflection import Reflection
from .memory.experience import Experience
from .memory.summary import Summary
from .memory.knowledge import Knowledge
from .memory.manager import MemoryManager
from .memory.memory_engine import MemoryEngine

from .relationship import (
    RelationshipStorage,
    RelationshipUpdater,
)

from .llm import LLM


class VCore:

    def __init__(self):

        self.config = load_config()

        llm = LLM()

        storage = MemoryStorage(
            self.config.memory_root
        )

        manager = MemoryManager(
            storage
        )

        session = Session()

        reflection = Reflection(
            llm
        )

        experience = Experience(
            llm
        )

        summary = Summary(
            llm
        )

        knowledge = Knowledge(
            llm
        )

        relationship_storage = RelationshipStorage(
            self.config.memory_root / "relationship"
        )

        relationship_updater = RelationshipUpdater(
            llm
        )

        memory = MemoryEngine(
            session=session,
            reflection=reflection,
            experience=experience,
            summary=summary,
            knowledge=knowledge,
            manager=manager,
            relationship_updater=relationship_updater,
            relationship_storage=relationship_storage,
        )

        self.agent = Agent(
            config=self.config,
            memory=memory,
        )

        self.autonomy = AutonomousRunner(
            self.config.autonomy_root
        )

        self.runtime_registry = RuntimeRegistry(
            self.config.autonomy_root / "runtime"
        )
        self.runtime_registry.register("v-core")

    async def ask(
        self,
        prompt: str,
    ) -> str:

        return await self.agent.run(
            prompt
        )

    async def close(self) -> None:
        try:
            await self.agent.close()
        finally:
            self.runtime_registry.unregister()

    async def run_autonomous(
        self,
        objective: str,
        driver: StepDriver,
        envelope: AuthorizationEnvelope | None = None,
        *,
        task_id: str | None = None,
    ) -> AutonomousTask:
        task = AutonomousTask(
            objective=objective,
            **({"task_id": task_id} if task_id else {}),
        )

        if envelope is None:
            envelope = AuthorizationEnvelope(
                workspace=str(
                    self.config.workspace
                    / "autonomous"
                    / task.task_id
                )
            )

        return await self.autonomy.run(
            task,
            envelope,
            driver,
        )


async def chat():

    core = VCore()

    print("PALADYN Framework powered by V")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            try:

                prompt = input("V > ").strip()

                if prompt.lower() in {
                    "exit",
                    "quit",
                }:
                    break

                print(
                    await core.ask(prompt)
                )

            except KeyboardInterrupt:
                break

            except Exception as exc:
                print(f"\n{exc}\n")
    finally:
        await core.close()


def main():

    asyncio.run(
        chat()
    )


if __name__ == "__main__":
    main()
