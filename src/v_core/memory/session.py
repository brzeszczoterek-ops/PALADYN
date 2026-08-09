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
    ) -> list[dict]:

        history = []

        for event in self.events[-limit:]:

            if event.event_type != "task":
                continue

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
