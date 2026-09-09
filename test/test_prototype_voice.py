import asyncio
import json
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.persona.voice import looks_generic_assistant_voice


PAYLOAD = json.dumps({"status": "validated", "validation": {
    "passed": True, "activation_eligible": False, "qualification": "prototype_only"}})


def agent(language=""):
    result = object.__new__(Agent)
    result.memory = SimpleNamespace(relationship_state=SimpleNamespace(preferred_response_language=language))
    return result


def answer():
    return Agent._generated_prototype_answer("learning_create_tool", PAYLOAD)


@pytest.mark.asyncio
async def test_default_voice_stays_english_for_polish_prompt_without_extra_inference():
    instance = agent()
    class NoInference:
        async def ask(self, **kwargs):
            raise AssertionError("default grounded response needs no rewrite")
    instance.llm = NoInference()
    rendered = await instance._render_prototype_feedback("Stwórz narzędzie", answer())
    assert rendered == answer()
    assert "Boss," in rendered
    assert "not a working tool" in rendered
    assert "not activating" in rendered
    assert "run has stopped" in rendered
    assert not looks_generic_assistant_voice(rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize("preference,override", [("Polish", ""), ("English", "Polish")])
async def test_configured_language_and_session_override_use_existing_voice_path(preference, override):
    instance = agent(preference)
    instance._response_language_override = override
    polish = ("Boss, mam prototyp, nie działające narzędzie. Porównywanie kodu z jego własnym "
              "wynikiem nie dowodzi, że robi robotę. Kod i wyniki prób są zachowane, ale "
              "nie aktywuję narzędzia. Potrzebny jest niezależny wzorzec poprawnego wyniku. "
              "Zadanie nie jest wykonane, a przebieg się zakończył.")
    requests = []
    class Translator:
        async def ask(self, **kwargs):
            requests.append(kwargs)
            return polish
    instance.llm = Translator()
    rendered = await instance._render_prototype_feedback("Test this prototype", answer())
    assert rendered == polish
    assert len(requests) == 1
    assert "natural Polish" in str(requests[0]["messages"])
    assert "not activated" in str(requests[0]["messages"])


@pytest.mark.asyncio
async def test_rewrite_error_preserves_truthful_fallback_and_records_language_mismatch():
    instance = agent("Polish")
    class Broken:
        async def ask(self, **kwargs):
            raise RuntimeError("model unavailable")
    instance.llm = Broken()
    events = []
    trace = SimpleNamespace(record_event=lambda name, data: events.append((name, data)))
    assert await instance._render_prototype_feedback("Test", answer(), trace=trace) == answer()
    assert events[0][0] == "prototype_feedback_fallback"
    assert events[0][1]["fallback_language"] == "English"


@pytest.mark.asyncio
async def test_rewrite_cannot_add_obvious_unexecuted_actions():
    instance = agent()
    async def bad_rewrite(*args, **kwargs):
        return "I opened the website and downloaded the page."
    instance._enforce_english = bad_rewrite
    assert await instance._render_prototype_feedback("Test", answer()) == answer()


@pytest.mark.asyncio
async def test_presentation_cancellation_is_not_swallowed():
    instance = agent()
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    instance._enforce_english = cancelled
    with pytest.raises(asyncio.CancelledError):
        await instance._render_prototype_feedback("Test", answer())
