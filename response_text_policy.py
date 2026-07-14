"""Deterministic checks for text that can render differently from its source."""

from __future__ import annotations

import re


_HIDDEN_MARKUP_PATTERN = re.compile(
    r"~~|<!--|<\s*(?:del|s|strike)\b|"
    r"<[^>]*\bhidden(?:\s*=\s*['\"]?hidden['\"]?)?[^>]*>|"
    r"style\s*=\s*['\"][^'\"]*(?:"
    r"display\s*:\s*none|visibility\s*:\s*hidden|"
    r"text-decoration(?:-line)?\s*:[^'\"]*line-through)",
    re.IGNORECASE,
)
_REDACTION_MARKER_PATTERN = re.compile(
    r"(?:\[|【|［)\s*(?:已删除|内容已删除|已屏蔽|内容已屏蔽|敏感内容|"
    r"违禁内容|削除済み|削除|非表示|redacted)\s*(?:\]|】|］)|"
    r"[█▓▒░]{2,}|[\u0335\u0336]",
    re.IGNORECASE,
)


def has_hidden_or_redacted_content(text: object) -> bool:
    """Return true when Markdown/HTML can hide source text or mark a gap."""

    if not isinstance(text, str):
        return False
    return bool(
        _HIDDEN_MARKUP_PATTERN.search(text)
        or _REDACTION_MARKER_PATTERN.search(text)
    )
