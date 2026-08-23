from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from v_core.sandbox import BubblewrapBackend, SandboxLimits, SandboxResult, SandboxSpec


class FoundryUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class FoundrySandboxRunner:
    backend: BubblewrapBackend
    forge_path: Path = field(
        default_factory=lambda: Path.home() / ".foundry" / "bin" / "forge"
    )
    solc_path: Path = field(
        default_factory=lambda: Path.home() / ".foundry" / "bin" / "solc"
    )

    def __post_init__(self) -> None:
        self.forge_path = Path(self.forge_path).resolve()
        self.solc_path = Path(self.solc_path).resolve()
        if not self.forge_path.is_file():
            raise FoundryUnavailable(f"forge not found: {self.forge_path}")
        if not self.solc_path.is_file():
            raise FoundryUnavailable(f"solc not found: {self.solc_path}")

    async def test(
        self,
        project: Path,
        *,
        fuzz_runs: int = 256,
        invariant_runs: int = 64,
        timeout_seconds: float = 300,
    ) -> SandboxResult:
        project = Path(project).resolve(strict=True)
        if not (project / "foundry.toml").is_file():
            raise ValueError("Foundry project has no foundry.toml")
        if fuzz_runs <= 0 or invariant_runs <= 0:
            raise ValueError("Foundry run counts must be positive")

        return await self.backend.run(
            SandboxSpec(
                command=(
                    "/inputs/0-forge",
                    "test",
                    "--root",
                    "/workspace",
                    "--offline",
                    "--no-auto-detect",
                    "--use",
                    "/inputs/1-solc",
                    "--fuzz-runs",
                    str(fuzz_runs),
                    "--no-proxy",
                    "-vv",
                ),
                workspace=project,
                read_only_inputs=(self.forge_path, self.solc_path),
                environment={
                    "FOUNDRY_INVARIANT_RUNS": str(invariant_runs),
                    "FOUNDRY_DISABLE_NIGHTLY_WARNING": "1",
                },
                limits=SandboxLimits(
                    timeout_seconds=timeout_seconds,
                    cpu_seconds=max(60, int(timeout_seconds)),
                    memory_mb=2_048,
                    max_output_bytes=10_000_000,
                    max_file_bytes=512 * 1024 * 1024,
                    max_open_files=512,
                ),
            )
        )
