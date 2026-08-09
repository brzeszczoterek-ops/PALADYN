from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(
    response: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parse JSON returned by an LLM.

    Handles:
    - <|begin of thinking|> ... <|end of thinking|>
    - ```json ... ```
    - leading/trailing whitespace
    - extra text before or after the JSON object

    Returns:
        Parsed dictionary or the provided default.
    """

    if default is None:
        default = {}

    if not response:
        return default

    text = response.strip()

    #
    # Remove thinking blocks
    #

    text = re.sub(
        r"<\|begin of thinking\|>.*?<\|end of thinking\|>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    #
    # Remove Markdown code fences
    #

    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    #
    # Find first JSON object
    #

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return default

    text = text[start:end + 1]

    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return default
