from __future__ import annotations

import asyncio
from types import SimpleNamespace

from starlette.testclient import TestClient

from v_core.edition import load_edition_extension, resolve_edition
from v_core.ui import UIRuntime, create_app


class _Process:
    def poll(self) -> None:
        return None


class _Tools:
    def __init__(self) -> None:
        self.edition_extension = load_edition_extension(resolve_edition("public"))

    def local_tool_names(self) -> list[str]:
        return ["web_search", "web_read", "learning_create_tool"]


class _Core:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(tools=_Tools())
        self.model_runtime = None
        self.closed = False

    async def ask(self, prompt: str, on_token=None) -> str:
        answer = f"V received: {prompt}"
        if on_token is not None:
            on_token("V received: ")
            on_token(prompt)
        return answer

    async def close(self) -> None:
        self.closed = True


def _runtime() -> UIRuntime:
    edition = resolve_edition("public")
    config = SimpleNamespace(edition=edition, voice_root="voice")
    session = SimpleNamespace(
        profile=SimpleNamespace(
            alias="test-model",
            model_path="/models/test.gguf",
            context_size=12_032,
            reasoning="off",
            cache_type_k="q8_0",
            cache_type_v="q4_0",
        ),
        process=_Process(),
    )
    return UIRuntime(core=_Core(), config=config, model_session=session)


def test_ui_is_local_session_token_protected() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    index = client.get("/")
    denied = client.get("/api/status")
    accepted = client.get(
        "/api/status",
        headers={"X-PALADYN-Session": runtime.session_token},
    )

    assert index.status_code == 200
    assert runtime.session_token in index.text
    assert "frame-ancestors 'none'" in index.headers["content-security-policy"]
    assert denied.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["edition"] == "public"
    assert accepted.json()["model"]["alias"] == "test-model"
    assert accepted.json()["owner"] is None


def test_ui_uses_independent_scroll_regions_and_fixed_composer() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    index = client.get("/")
    css = client.get("/assets/app.css")
    javascript = client.get("/assets/app.js")

    assert index.status_code == 200
    assert 'id="messages" class="messages" tabindex="0"' in index.text
    assert css.status_code == 200
    assert ".conversation { display: flex; min-height: 0; flex-direction: column; overflow: hidden; }" in css.text
    assert "flex: 0 0 auto;" in css.text
    assert "scrollbar-gutter: stable;" in css.text
    assert javascript.status_code == 200
    assert "function chatIsPinnedToBottom()" in javascript.text
    assert "scrollChatToBottom(followOutput);" in javascript.text


def test_ui_streams_v_response_as_ndjson() -> None:
    runtime = _runtime()
    client = TestClient(create_app(runtime))

    with client.stream(
        "POST",
        "/api/chat",
        headers={"X-PALADYN-Session": runtime.session_token},
        json={"message": "hello", "speak": False},
    ) as response:
        events = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert '"type": "started"' in events[0]
    assert any('"text": "V received: "' in line for line in events)
    assert any('"text": "hello"' in line for line in events)
    assert '"type": "done"' in events[-1]


def test_ui_shutdown_uses_server_callback() -> None:
    runtime = _runtime()
    called: list[bool] = []
    runtime.shutdown_callback = lambda: called.append(True)
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/shutdown",
        headers={"X-PALADYN-Session": runtime.session_token},
        json={},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "shutting_down"}


def test_ui_runtime_close_stops_core_and_managed_model() -> None:
    runtime = _runtime()

    asyncio.run(runtime.close())

    assert runtime.core.closed is True
