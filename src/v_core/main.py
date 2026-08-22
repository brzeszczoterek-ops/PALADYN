from __future__ import annotations

import asyncio
from pathlib import Path

from .agent import Agent
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
            Path("memory")
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
            "memory/relationship"
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

    async def ask(
        self,
        prompt: str,
    ) -> str:

        return await self.agent.run(
            prompt
        )


async def chat():

    core = VCore()

    print("PALADYN Framework powered by V")
    print("Type 'exit' to quit.\n")

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


def main():

    asyncio.run(
        chat()
    )


if __name__ == "__main__":
    main()
