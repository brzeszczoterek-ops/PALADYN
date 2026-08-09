from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReflectionEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    task: str = ""

    result: str = ""

    summary: str = ""

    lesson: str = ""

    importance: str = "low"

    remember: bool = False


@dataclass
class ExperienceEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    summary: str = ""

    lesson: str = ""

    confidence: float = 0.0

    importance: str = "low"


@dataclass
class SummaryEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    summary: str = ""

    lessons: list[str] = field(default_factory=list)

    confidence: float = 0.0

    importance: str = "low"


@dataclass
class KnowledgeEntry:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    title: str = ""

    content: str = ""

    reason: str = ""

    confidence: float = 0.0


@dataclass
class SessionEvent:

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    event_type: str = ""

    data: dict = field(default_factory=dict)
