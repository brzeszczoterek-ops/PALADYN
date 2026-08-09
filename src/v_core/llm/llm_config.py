from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = """
You are V, a local autonomous AI assistant.

Be concise.
Prefer using available tools.
Never invent information.
Use available tools whenever possible.
""".strip()


@dataclass(frozen=True, slots=True)
class LLMConfig:

    provider: str

    base_url: str

    model: str

    context: int

    temperature: float

    top_p: float

    system_prompt: str


MODELS = {

    "qwythos": LLMConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:5001/v1",
        model="Qwythos-9B-Claude-Mythos-5-1M-Q6_K",
        context=32768,
        temperature=0.2,
        top_p=0.95,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ),

    # Przykład kolejnego modelu:
    #
    # "deepseek": LLMConfig(
    #     provider="llama_cpp",
    #     base_url="http://127.0.0.1:5001/v1",
    #     model="DeepSeek-R1-Distill-Qwen-14B",
    #     context=32768,
    #     temperature=0.2,
    #     top_p=0.95,
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    # ),
}

CURRENT = MODELS["qwythos"]
