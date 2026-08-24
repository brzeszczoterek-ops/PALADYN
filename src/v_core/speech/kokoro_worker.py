#!/usr/bin/env python3
"""Persistent, local Kokoro renderer using a small JSON-lines protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import soundfile as sf
from kokoro_onnx import Kokoro


REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--language", default="en-gb")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


async def _render(kokoro: Kokoro, request: dict[str, Any], args: argparse.Namespace) -> None:
    request_id = request.get("id")
    text = request.get("text")
    output_value = request.get("output_dir")
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("invalid request id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(output_value, str):
        raise ValueError("output_dir must be a path")
    output_dir = Path(output_value).resolve(strict=True)
    if not output_dir.is_dir():
        raise ValueError("output_dir is not a directory")

    chunk_count = 0
    async for samples, sample_rate in kokoro.create_stream(
        text,
        voice=args.voice,
        speed=args.speed,
        lang=args.language,
    ):
        chunk_count += 1
        final_path = output_dir / f"{request_id}-{chunk_count:04d}.wav"
        partial_path = final_path.with_suffix(".wav.part")
        sf.write(partial_path, samples, sample_rate, format="WAV")
        os.chmod(partial_path, 0o600)
        partial_path.replace(final_path)
        _emit(
            {
                "event": "chunk",
                "id": request_id,
                "path": str(final_path),
                "sample_rate": sample_rate,
            }
        )

    if chunk_count == 0:
        raise RuntimeError("Kokoro returned no audio")
    _emit({"event": "done", "id": request_id, "chunks": chunk_count})


async def _main() -> int:
    os.umask(0o077)
    args = _arguments()
    kokoro = Kokoro(args.model, args.voices)
    if args.voice not in kokoro.get_voices():
        raise ValueError(f"unknown Kokoro voice: {args.voice}")
    _emit({"event": "ready", "voice": args.voice})

    while line := await asyncio.to_thread(sys.stdin.readline):
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            if request.get("command") == "shutdown":
                _emit({"event": "stopped"})
                return 0
            if request.get("command") != "synthesize":
                raise ValueError("unknown command")
            await _render(kokoro, request, args)
        except Exception as error:
            request_id = request.get("id") if isinstance(request, dict) else None
            _emit(
                {
                    "event": "error",
                    "id": request_id,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
