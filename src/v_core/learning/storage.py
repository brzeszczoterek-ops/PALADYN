from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any

from .models import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactStatus,
    LearnedLesson,
    LearningEvidence,
    SkillManifest,
    ToolManifest,
    utc_now,
)


class LearningStorageError(RuntimeError):
    pass


class JournalIntegrityError(LearningStorageError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(manifest: dict[str, Any], source: str = "") -> str:
    checksum = hashlib.sha256()
    checksum.update(_canonical(manifest))
    checksum.update(b"\x00")
    checksum.update(source.encode("utf-8"))
    return checksum.hexdigest()


@dataclass(slots=True)
class HashChainJournal:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)

    def append(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        event = " ".join(str(event).split())[:120]
        if not event:
            raise ValueError("journal event is required")
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                records = self._read_handle(handle)
                previous = records[-1]["hash"] if records else "0" * 64
                body = {
                    "sequence": len(records) + 1,
                    "previous_hash": previous,
                    "timestamp": utc_now(),
                    "event": event,
                    "data": data,
                }
                record = dict(body)
                record["hash"] = hashlib.sha256(_canonical(body)).hexdigest()
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.chmod(self.path, 0o600)
                return record
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise LearningStorageError(f"could not append learning journal: {error}") from error

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                return self._read_handle(handle)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, JournalIntegrityError):
                raise
            raise LearningStorageError(f"could not read learning journal: {error}") from error

    @staticmethod
    def _read_handle(handle) -> list[dict[str, Any]]:
        handle.seek(0)
        result: list[dict[str, Any]] = []
        previous = "0" * 64
        for expected_sequence, line in enumerate(handle, start=1):
            if len(line) > 2_000_000:
                raise JournalIntegrityError("journal line exceeds size limit")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise JournalIntegrityError("journal record is not an object")
            body = {key: value for key, value in record.items() if key != "hash"}
            expected_hash = hashlib.sha256(_canonical(body)).hexdigest()
            if record.get("sequence") != expected_sequence:
                raise JournalIntegrityError("journal sequence is broken")
            if record.get("previous_hash") != previous:
                raise JournalIntegrityError("journal hash chain is broken")
            if record.get("hash") != expected_hash:
                raise JournalIntegrityError("journal record was modified")
            previous = expected_hash
            result.append(record)
        return result


class LearningStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.quarantine = self.root / "quarantine"
        self.records_root = self.root / "records"
        self.lessons_root = self.root / "lessons"
        for directory in (
            self.root,
            self.quarantine,
            self.records_root,
            self.lessons_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            self._protect(directory, 0o700)
        self.evidence_journal = HashChainJournal(self.root / "evidence.jsonl")
        self.audit_journal = HashChainJournal(self.root / "audit.jsonl")
        self.lock_path = self.root / ".store.lock"
        self._thread_lock = threading.RLock()
        self._lock_state = threading.local()

    def add_evidence(self, evidence: LearningEvidence) -> None:
        with self._locked():
            if any(
                existing.evidence_id == evidence.evidence_id
                for existing in self.list_evidence()
            ):
                raise LearningStorageError(
                    f"duplicate evidence ID: {evidence.evidence_id}"
                )
            self.evidence_journal.append("evidence_recorded", evidence.to_dict())

    def list_evidence(self) -> list[LearningEvidence]:
        with self._locked():
            return [
                LearningEvidence.from_dict(record["data"])
                for record in self.evidence_journal.read_verified()
                if record["event"] == "evidence_recorded"
            ]

    def get_evidence(self, evidence_id: str) -> LearningEvidence | None:
        with self._locked():
            for evidence in self.list_evidence():
                if evidence.evidence_id == evidence_id:
                    return evidence
            return None

    def save_lesson(self, lesson: LearnedLesson) -> Path:
        with self._locked():
            path = self.lessons_root / f"{lesson.lesson_id}.json"
            lesson_data = lesson.to_dict()
            lesson_digest = hashlib.sha256(_canonical(lesson_data)).hexdigest()
            self._atomic_json(path, {"schema_version": 1, "lesson": lesson_data})
            self.audit_journal.append(
                "lesson_saved",
                {
                    "lesson_id": lesson.lesson_id,
                    "status": lesson.status.value,
                    "revision": lesson.revision,
                    "digest": lesson_digest,
                },
            )
            return path

    def load_lesson(self, lesson_id: str) -> LearnedLesson | None:
        with self._locked():
            path = self._id_path(self.lessons_root, lesson_id)
            if not path.exists():
                return None
            payload = self._load_json(path)
            if payload.get("schema_version") != 1:
                raise LearningStorageError("unsupported lesson schema")
            lesson = LearnedLesson.from_dict(payload["lesson"])
            saved = [
                item["data"]
                for item in self.audit_journal.read_verified()
                if item["event"] == "lesson_saved"
                and item["data"].get("lesson_id") == lesson.lesson_id
            ]
            if not saved:
                raise JournalIntegrityError("lesson has no audit record")
            latest = saved[-1]
            expected = hashlib.sha256(_canonical(lesson.to_dict())).hexdigest()
            if (
                latest.get("digest") != expected
                or latest.get("status") != lesson.status.value
                or latest.get("revision") != lesson.revision
            ):
                raise JournalIntegrityError("lesson differs from audit journal")
            return lesson

    def list_lessons(self) -> list[LearnedLesson]:
        with self._locked():
            lessons = []
            for path in sorted(self.lessons_root.glob("*.json")):
                lesson = self.load_lesson(path.stem)
                if lesson is not None:
                    lessons.append(lesson)
            return lessons

    def stage_tool(
        self,
        manifest: ToolManifest,
        source: str,
        *,
        scope_key: str,
    ) -> ArtifactRecord:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("generated tool source is required")
        if len(source.encode("utf-8")) > 200_000 or "\x00" in source:
            raise ValueError("generated tool source is invalid or too large")
        return self._stage(
            ArtifactKind.TOOL,
            manifest.to_dict(),
            manifest.name,
            manifest.version,
            manifest.scope,
            scope_key,
            source,
        )

    def stage_skill(
        self,
        manifest: SkillManifest,
        *,
        scope_key: str,
    ) -> ArtifactRecord:
        return self._stage(
            ArtifactKind.SKILL,
            manifest.to_dict(),
            manifest.name,
            manifest.version,
            manifest.scope,
            scope_key,
        )

    def _stage(
        self,
        kind: ArtifactKind,
        manifest: dict[str, Any],
        name: str,
        version: str,
        scope,
        scope_key: str,
        source: str = "",
    ) -> ArtifactRecord:
        with self._locked():
            return self._stage_locked(
                kind,
                manifest,
                name,
                version,
                scope,
                scope_key,
                source,
            )

    def _stage_locked(
        self,
        kind: ArtifactKind,
        manifest: dict[str, Any],
        name: str,
        version: str,
        scope,
        scope_key: str,
        source: str = "",
    ) -> ArtifactRecord:
        for existing in self.list_records():
            if (
                existing.kind is kind
                and existing.name == name
                and existing.version == version
                and existing.scope is scope
                and existing.scope_key == scope_key
            ):
                raise LearningStorageError(
                    f"immutable artifact version already exists: {name} {version}"
                )

        artifact_id = os.urandom(16).hex()
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{artifact_id}-", dir=self.quarantine)
        )
        final = self.quarantine / artifact_id
        moved = False
        try:
            self._write_json_file(
                temporary / "manifest.json",
                {"schema_version": 1, "kind": kind.value, "manifest": manifest},
            )
            source_path = ""
            if kind is ArtifactKind.TOOL:
                path = temporary / "tool.py"
                path.write_text(source, encoding="utf-8")
                self._protect(path, 0o600)
                source_path = str((final / "tool.py").relative_to(self.root))
            os.replace(temporary, final)
            moved = True
            record = ArtifactRecord(
                artifact_id=artifact_id,
                kind=kind,
                name=name,
                version=version,
                scope=scope,
                scope_key=scope_key,
                digest=_digest(manifest, source),
                manifest_path=str((final / "manifest.json").relative_to(self.root)),
                source_path=source_path,
            )
            self.save_record(record)
            self.audit_journal.append(
                "artifact_staged",
                {
                    "artifact_id": artifact_id,
                    "kind": kind.value,
                    "name": name,
                    "version": version,
                    "scope": scope.value,
                    "scope_key": scope_key,
                    "digest": record.digest,
                },
            )
            return record
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if moved:
                shutil.rmtree(final, ignore_errors=True)
            raise

    def save_record(self, record: ArtifactRecord) -> None:
        with self._locked():
            path = self.records_root / f"{record.artifact_id}.json"
            self._atomic_json(path, {"schema_version": 1, "record": record.to_dict()})

    def load_record(self, artifact_id: str) -> ArtifactRecord | None:
        with self._locked():
            path = self._id_path(self.records_root, artifact_id)
            if not path.exists():
                return None
            payload = self._load_json(path)
            if payload.get("schema_version") != 1:
                raise LearningStorageError("unsupported artifact record schema")
            record = ArtifactRecord.from_dict(payload["record"])
            self._verify_record_audit(record)
            return record

    def list_records(
        self,
        *,
        status: ArtifactStatus | None = None,
        kind: ArtifactKind | None = None,
    ) -> list[ArtifactRecord]:
        with self._locked():
            records: list[ArtifactRecord] = []
            for path in sorted(self.records_root.glob("*.json")):
                record = self.load_record(path.stem)
                if record is None:
                    continue
                if status is not None and record.status is not status:
                    continue
                if kind is not None and record.kind is not kind:
                    continue
                records.append(record)
            return records

    def load_tool(self, record: ArtifactRecord) -> tuple[ToolManifest, Path]:
        if record.kind is not ArtifactKind.TOOL:
            raise LearningStorageError("artifact is not a tool")
        payload = self._load_json(self._relative(record.manifest_path))
        manifest = ToolManifest.from_dict(payload["manifest"])
        source_path = self._relative(record.source_path)
        source = source_path.read_text(encoding="utf-8")
        if _digest(manifest.to_dict(), source) != record.digest:
            raise JournalIntegrityError("tool bundle digest mismatch")
        return manifest, source_path

    def load_skill(self, record: ArtifactRecord) -> SkillManifest:
        if record.kind is not ArtifactKind.SKILL:
            raise LearningStorageError("artifact is not a skill")
        payload = self._load_json(self._relative(record.manifest_path))
        manifest = SkillManifest.from_dict(payload["manifest"])
        if _digest(manifest.to_dict()) != record.digest:
            raise JournalIntegrityError("skill bundle digest mismatch")
        return manifest

    def transition(
        self,
        record: ArtifactRecord,
        status: ArtifactStatus,
        *,
        validation: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        with self._locked():
            current = self.load_record(record.artifact_id)
            if current is None:
                raise LearningStorageError("artifact disappeared during transition")
            if current.status is not record.status:
                raise LearningStorageError("artifact status changed concurrently")
            current.validation = dict(record.validation)
            return self._transition_locked(current, status, validation=validation)

    def _transition_locked(
        self,
        record: ArtifactRecord,
        status: ArtifactStatus,
        *,
        validation: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record.transition(status)
        if validation is not None:
            record.validation = validation
        if status is ArtifactStatus.ACTIVE:
            record.activation_count += 1
        self.save_record(record)
        self.audit_journal.append(
            "artifact_transitioned",
            {
                "artifact_id": record.artifact_id,
                "from_digest": record.digest,
                "status": status.value,
                "validation": record.validation,
            },
        )
        return record

    def activate(self, record: ArtifactRecord) -> ArtifactRecord:
        with self._locked():
            current = self.load_record(record.artifact_id)
            if current is None or current.status is not ArtifactStatus.VALIDATED:
                raise LearningStorageError("artifact is no longer validated")
            if current.kind is ArtifactKind.TOOL:
                self.load_tool(current)
            else:
                self.load_skill(current)
            previous = [
                item
                for item in self.list_records(status=ArtifactStatus.ACTIVE)
                if item.kind is current.kind
                and item.name == current.name
                and item.scope is current.scope
                and item.scope_key == current.scope_key
            ]
            retired: list[ArtifactRecord] = []
            try:
                for item in previous:
                    item.validation["retired_reason"] = (
                        f"superseded_by:{current.artifact_id}"
                    )
                    self._transition_locked(item, ArtifactStatus.RETIRED)
                    retired.append(item)
                return self._transition_locked(current, ArtifactStatus.ACTIVE)
            except Exception:
                for item in retired:
                    self._transition_locked(item, ArtifactStatus.ACTIVE)
                raise

    def record_tool_run(
        self,
        artifact_id: str,
        *,
        succeeded: bool,
    ) -> ArtifactRecord:
        with self._locked():
            record = self.load_record(artifact_id)
            if record is None:
                raise LearningStorageError("generated tool record disappeared")
            if succeeded:
                record.successful_runs += 1
                self.save_record(record)
                self.audit_journal.append(
                    "generated_tool_succeeded",
                    {
                        "artifact_id": record.artifact_id,
                        "successful_runs": record.successful_runs,
                    },
                )
                return record

            record.failure_count += 1
            if record.failure_count >= 3 and record.status is ArtifactStatus.ACTIVE:
                rollback = next(
                    (
                        item
                        for item in reversed(self.list_records())
                        if item.kind is record.kind
                        and item.name == record.name
                        and item.scope is record.scope
                        and item.scope_key == record.scope_key
                        and item.status is ArtifactStatus.RETIRED
                        and item.validation.get("retired_reason")
                        == f"superseded_by:{record.artifact_id}"
                    ),
                    None,
                )
                record.validation["auto_retired_reason"] = (
                    "three runtime failures after activation"
                )
                if rollback is not None:
                    record.validation["rolled_back_to"] = rollback.artifact_id
                self._transition_locked(record, ArtifactStatus.RETIRED)
                if rollback is not None:
                    rollback.validation["reactivated_reason"] = (
                        f"rollback_from:{record.artifact_id}"
                    )
                    self._transition_locked(rollback, ArtifactStatus.ACTIVE)
            else:
                self.save_record(record)
            self.audit_journal.append(
                "generated_tool_failed",
                {
                    "artifact_id": record.artifact_id,
                    "failure_count": record.failure_count,
                    "status": record.status.value,
                },
            )
            return record

    def _relative(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise LearningStorageError("stored artifact path escapes learning root")
        return path

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            depth = getattr(self._lock_state, "depth", 0)
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock_state.depth -= 1
                return

            descriptor = os.open(
                self.lock_path,
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                os.chmod(self.lock_path, 0o600)
                self._lock_state.depth = 1
                try:
                    yield
                finally:
                    self._lock_state.depth = 0

    def _verify_record_audit(self, record: ArtifactRecord) -> None:
        records = self.audit_journal.read_verified()
        staged = next(
            (
                item["data"]
                for item in records
                if item["event"] == "artifact_staged"
                and item["data"].get("artifact_id") == record.artifact_id
            ),
            None,
        )
        if staged is None:
            raise JournalIntegrityError("artifact record has no staging audit event")
        identity = (
            staged.get("kind"),
            staged.get("name"),
            staged.get("version"),
            staged.get("scope"),
            staged.get("scope_key"),
            staged.get("digest"),
        )
        expected_identity = (
            record.kind.value,
            record.name,
            record.version,
            record.scope.value,
            record.scope_key,
            record.digest,
        )
        if identity != expected_identity:
            raise JournalIntegrityError("artifact identity differs from audit journal")

        expected_status = ArtifactStatus.QUARANTINED.value
        expected_validation: dict[str, Any] = {}
        activation_count = 0
        successful_runs = 0
        failure_count = 0
        for item in records:
            if (
                item["event"] == "artifact_transitioned"
                and item["data"].get("artifact_id") == record.artifact_id
            ):
                expected_status = item["data"].get("status")
                expected_validation = item["data"].get("validation", {})
                if expected_status == ArtifactStatus.ACTIVE.value:
                    activation_count += 1
            if (
                item["event"] == "generated_tool_succeeded"
                and item["data"].get("artifact_id") == record.artifact_id
            ):
                successful_runs += 1
            if (
                item["event"] == "generated_tool_failed"
                and item["data"].get("artifact_id") == record.artifact_id
            ):
                failure_count += 1
        if record.status.value != expected_status:
            raise JournalIntegrityError("artifact status differs from audit journal")
        if record.validation != expected_validation:
            raise JournalIntegrityError("artifact validation differs from audit journal")
        if record.activation_count != activation_count:
            raise JournalIntegrityError("artifact activation count differs from audit journal")
        if record.successful_runs != successful_runs:
            raise JournalIntegrityError("artifact success count differs from audit journal")
        if record.failure_count != failure_count:
            raise JournalIntegrityError("artifact failure count differs from audit journal")

    @staticmethod
    def _id_path(root: Path, identifier: str) -> Path:
        if not identifier or any(character not in "0123456789abcdef" for character in identifier):
            raise ValueError("identifier must be lowercase hexadecimal")
        return root / f"{identifier}.json"

    def _atomic_json(self, path: Path, payload: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._protect(path, 0o600)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise

    def _write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        self._protect(path, 0o600)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise LearningStorageError(f"could not load {path.name}: {error}") from error
        if not isinstance(payload, dict):
            raise LearningStorageError(f"{path.name} is not a JSON object")
        return payload

    @staticmethod
    def _protect(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError:
            pass
