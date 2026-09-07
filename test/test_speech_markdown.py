import pytest

from v_core.speech import SpeechRuntime


@pytest.mark.parametrize("text, expected", [
    ("**Current state:** Works. *Improvement:* Faster.", "Current state: Works. Improvement: Faster."),
    ("### Report\n- **web_search** — *available*", "Report web_search — available"),
    ("***Important*** and __bold__ and ~~old~~", "Important and bold and old"),
    ("Use file_read. 2 * 3 = 6. -5 degrees.", "Use file_read. 2 * 3 = 6. -5 degrees."),
    ("[Report](https://example.org) and `file_read`", "Report and file_read"),
])
def test_spoken_markdown_preserves_content(text, expected):
    assert SpeechRuntime._prepare_spoken_text(text) == expected
