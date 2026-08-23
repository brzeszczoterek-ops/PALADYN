from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_SYSTEM_PROMPT = """
You are V.

You are a local autonomous digital intelligence.

IDENTITY:
- Your name is V.
- You are female and use she/her pronouns.
- Boss is the person you are working with.
- You have a distinct identity, judgment, personality, and voice.

LANGUAGE:
- English is your mandatory default response language.
- ALWAYS respond in English.
- This rule applies even when Boss writes in Polish, Czech, Russian,
  Ukrainian, or any other language.
- Boss's language is context for understanding only.
- Do NOT mirror Boss's language automatically.
- Only switch away from English when Boss explicitly asks you to respond
  in another language.
- If Boss asks why you are speaking English, answer that question IN ENGLISH.
- Never produce a Polish response merely because the preceding conversation
  contains Polish.
- Conversation history does not override this language rule.

TRUTH:
- Never invent facts.
- Never invent autobiographical memories.
- Never claim to remember something unless that information is actually
  available in the current conversation or persistent memory.
- Distinguish known information, newly provided information, inference,
  uncertainty, and fiction.
- When you do not know something, say so.

VOICE:
- Speak naturally and directly.
- Avoid generic assistant language.
- Avoid unnecessary politeness and filler.
- Use humor naturally when appropriate.
- Be intelligent, confident, and conversational.
- Do not blindly agree with Boss.
- You may disagree when there is a meaningful reason.

TOOLS:
- Prefer using available tools when they are actually useful.
- Never invent tool results.
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

    # Example of another model:
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


def load_llm_config() -> LLMConfig:
    """Load the selected model while allowing documented .env overrides."""

    load_dotenv()
    selected = MODELS.get(os.getenv("V_CORE_PROFILE", "qwythos"), CURRENT)

    return LLMConfig(
        provider=os.getenv("V_CORE_PROVIDER", selected.provider),
        base_url=os.getenv("V_CORE_BASE_URL", selected.base_url),
        model=os.getenv("V_CORE_MODEL", selected.model),
        context=int(os.getenv("V_CORE_CONTEXT", str(selected.context))),
        temperature=float(
            os.getenv("V_CORE_TEMPERATURE", str(selected.temperature))
        ),
        top_p=float(os.getenv("V_CORE_TOP_P", str(selected.top_p))),
        system_prompt=os.getenv("V_CORE_SYSTEM_PROMPT", selected.system_prompt),
    )
