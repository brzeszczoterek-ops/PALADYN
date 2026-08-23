from __future__ import annotations

import os
from pathlib import Path
import time

import typer
from rich.console import Console

from .control import ControlChannel, GlobalControlChannel
from .hotkey import LinuxHotkeyWatcher, input_devices, parse_chord
from .journal import CheckpointStore
from .models import ControlSignal
from .runtime import RuntimeRegistry
from v_core.evm.grants import LiveGrantStore
from v_core.evm.live import LiveAction, LiveActionGrant, MAX_LIVE_GRANT_SECONDS


app = typer.Typer(add_completion=False)
console = Console()


def default_root() -> Path:
    return Path(os.getenv("PALADYN_AUTONOMY_ROOT", "autonomy")).expanduser()


@app.command("signal")
def send_signal(
    task_id: str,
    action: ControlSignal,
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    channel = ControlChannel(root / "control", task_id)
    channel.request(action)
    console.print(f"{action.value.upper()} requested for {task_id}")


@app.command()
def status(
    task_id: str,
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    loaded = CheckpointStore(root / "checkpoints").load(task_id)
    if loaded is None:
        raise typer.Exit(code=1)

    task, envelope = loaded
    console.print(f"Task      : {task.task_id}")
    console.print(f"Status    : {task.status.value}")
    console.print(f"Mode      : {envelope.mode.value}")
    console.print(f"Actions   : {task.action_count}/{envelope.budget.max_actions}")
    console.print(f"Failures  : {task.failure_count}/{envelope.budget.max_failures}")
    console.print(f"Objective : {task.objective}")
    if task.blocked_reason:
        console.print(f"Reason    : {task.blocked_reason}")


@app.command("panic-all")
def panic_all(
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    GlobalControlChannel(root / "control").request_panic()
    terminated = RuntimeRegistry(root / "runtime").terminate_all()
    console.print("GLOBAL PANIC latched")
    if terminated:
        console.print(f"Termination requested for PID(s): {', '.join(map(str, terminated))}")


@app.command("reset-panic")
def reset_panic(
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    GlobalControlChannel(root / "control").clear_panic()
    console.print("Global panic latch cleared")


@app.command("input-devices")
def list_input_devices() -> None:
    devices = input_devices()
    if not devices:
        console.print("No Linux input event devices found.")
        raise typer.Exit(code=1)
    for path, name in devices:
        if not path.exists():
            access = "device node not visible"
        elif os.access(path, os.R_OK):
            access = "readable"
        else:
            access = "permission denied"
        console.print(f"{path}  {name}  [{access}]")


@app.command("hotkey")
def hotkey(
    device: Path = typer.Option(
        ...,
        help="Keyboard event device, for example /dev/input/event3.",
    ),
    chord: str = typer.Option("Q+P+0", help="Simultaneous emergency chord."),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    try:
        parsed_chord = parse_chord(chord)
    except ValueError as exc:
        console.print(f"Invalid chord: {exc}")
        raise typer.Exit(code=2) from exc

    global_control = GlobalControlChannel(root / "control")
    runtimes = RuntimeRegistry(root / "runtime")

    def trigger() -> None:
        global_control.request_panic()
        terminated = runtimes.terminate_all()
        console.print(
            f"EMERGENCY CHORD {chord.upper()} — GLOBAL PANIC; "
            f"termination requested for {len(terminated)} runtime(s)"
        )

    console.print(f"Watching {device} for {chord.upper()} (Ctrl+C exits watcher)")
    try:
        LinuxHotkeyWatcher(device, parsed_chord).watch(trigger)
    except PermissionError as exc:
        console.print(
            f"Cannot read {device}. Grant this user read access to the keyboard "
            "event device (normally through an input/uaccess rule)."
        )
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        console.print(f"Input device does not exist: {device}")
        raise typer.Exit(code=1) from exc


@app.command("grant-live")
def grant_live(
    chain_id: int = typer.Option(..., help="Exact EVM chain ID."),
    actions: str = typer.Option(
        "observe,simulate",
        help="Comma-separated: observe, simulate, sign, broadcast.",
    ),
    duration: int = typer.Option(600, help="Grant lifetime in seconds (max 900)."),
    targets: str = typer.Option("", help="Comma-separated target addresses."),
    selectors: str = typer.Option("", help="Comma-separated 4-byte selectors."),
    max_value_wei: int = typer.Option(0, help="Maximum simulated/sent value."),
    allow_state_change: bool = typer.Option(
        False,
        help="Required when sign or broadcast is requested.",
    ),
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    if duration <= 0 or duration > MAX_LIVE_GRANT_SECONDS:
        console.print(f"Duration must be 1..{MAX_LIVE_GRANT_SECONDS} seconds.")
        raise typer.Exit(code=2)
    try:
        parsed_actions = frozenset(
            LiveAction(item.strip().lower())
            for item in actions.split(",")
            if item.strip()
        )
    except ValueError as exc:
        console.print(f"Invalid live action: {exc}")
        raise typer.Exit(code=2) from exc
    state_changing = {LiveAction.SIGN, LiveAction.BROADCAST} & parsed_actions
    if state_changing and not allow_state_change:
        console.print("sign/broadcast requires --allow-state-change")
        raise typer.Exit(code=2)
    parsed_targets = frozenset(
        item.strip() for item in targets.split(",") if item.strip()
    )
    parsed_selectors = frozenset(
        item.strip() for item in selectors.split(",") if item.strip()
    )
    if state_changing and (not parsed_targets or not parsed_selectors):
        console.print("sign/broadcast requires explicit targets and selectors")
        raise typer.Exit(code=2)

    now = int(time.time())
    try:
        grant = LiveActionGrant(
            chain_id=chain_id,
            issued_at=now,
            expires_at=now + duration,
            actions=parsed_actions,
            allowed_targets=parsed_targets,
            allowed_selectors=parsed_selectors,
            max_value_wei=max_value_wei,
        )
    except ValueError as exc:
        console.print(f"Invalid grant: {exc}")
        raise typer.Exit(code=2) from exc
    LiveGrantStore(root / "live_grants").save(grant)
    console.print(f"Live grant: {grant.grant_id}")
    console.print(f"Expires   : {grant.expires_at}")
    console.print(f"Actions   : {', '.join(sorted(item.value for item in grant.actions))}")


@app.command("revoke-live")
def revoke_live(
    grant_id: str,
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    try:
        LiveGrantStore(root / "live_grants").revoke(grant_id)
    except ValueError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    console.print(f"Live grant revoked: {grant_id}")


@app.command("live-grants")
def live_grants(
    root: Path = typer.Option(default_root(), help="Autonomy state directory."),
) -> None:
    grants = LiveGrantStore(root / "live_grants").active(now=int(time.time()))
    if not grants:
        console.print("No active live grants.")
        return
    for grant in grants:
        actions = ",".join(sorted(item.value for item in grant.actions))
        console.print(
            f"{grant.grant_id} chain={grant.chain_id} "
            f"expires={grant.expires_at} actions={actions}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
