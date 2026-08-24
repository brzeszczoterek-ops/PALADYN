from __future__ import annotations

from dataclasses import asdict

from .models import SessionEvent


class Session:

    def __init__(self):

        self.events: list[SessionEvent] = []


    def add(
        self,
        event_type: str,
        data: dict,
    ):

        self.events.append(
            SessionEvent(
                event_type=event_type,
                data=data,
            )
        )


    def clear(self):

        self.events.clear()


    def export(self):

        return [
            asdict(event)
            for event in self.events
        ]


    def messages(
        self,
        limit: int = 10,
        max_characters: int | None = None,
    ) -> list[dict]:

        selected: list[SessionEvent] = []
        used_characters = 0

        for event in reversed(self.events):
            if event.event_type != "task":
                continue
            task = str(event.data.get("task", ""))
            result = str(event.data.get("result", ""))
            event_characters = len(task) + len(result)
            if (
                max_characters is not None
                and selected
                and used_characters + event_characters > max_characters
            ):
                break
            selected.append(event)
            used_characters += event_characters
            if len(selected) >= limit:
                break

        history = []

        for event in reversed(selected):

            task = event.data.get("task", "")
            result = event.data.get("result", "")

            if task:
                history.append(
                    {
                        "role": "user",
                        "content": task,
                    }
                )

            if result:
                history.append(
                    {
                        "role": "assistant",
                        "content": result,
                    }
                )

        return history


    def __len__(self):

        return len(self.events)
