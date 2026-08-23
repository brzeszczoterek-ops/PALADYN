from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .models import ArtifactKind, ArtifactStatus
from .storage import LearningStore


@dataclass(frozen=True, slots=True)
class LearningAuditReport:
    evidence_records: int
    lessons: int
    artifacts: int
    active_tools: int
    active_skills: int
    quarantined: int
    rejected: int
    retired: int
    integrity_verified: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def audit_learning_store(root: Path) -> LearningAuditReport:
    store = LearningStore(root)
    evidence = store.list_evidence()
    store.evidence_journal.read_verified()
    store.audit_journal.read_verified()
    lessons = store.list_lessons()
    artifacts = store.list_records()
    for record in artifacts:
        if record.kind is ArtifactKind.TOOL:
            store.load_tool(record)
        else:
            store.load_skill(record)
    return LearningAuditReport(
        evidence_records=len(evidence),
        lessons=len(lessons),
        artifacts=len(artifacts),
        active_tools=sum(
            record.kind is ArtifactKind.TOOL
            and record.status is ArtifactStatus.ACTIVE
            for record in artifacts
        ),
        active_skills=sum(
            record.kind is ArtifactKind.SKILL
            and record.status is ArtifactStatus.ACTIVE
            for record in artifacts
        ),
        quarantined=sum(
            record.status is ArtifactStatus.QUARANTINED for record in artifacts
        ),
        rejected=sum(record.status is ArtifactStatus.REJECTED for record in artifacts),
        retired=sum(record.status is ArtifactStatus.RETIRED for record in artifacts),
    )
