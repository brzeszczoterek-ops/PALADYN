"""Request-local UI preview; never execution evidence or speech input."""
from contextvars import ContextVar
from typing import Callable

response_preview: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "response_preview", default=None
)
