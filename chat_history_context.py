"""Build model history without replaying hidden assistant content."""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from response_text_policy import has_hidden_or_redacted_content


def build_model_history(records: Iterable[Any]) -> list[BaseMessage]:
    """Convert stored rows to messages, excluding unsafe assistant source text."""

    messages: list[BaseMessage] = []
    for record in records:
        message_type = getattr(record, "type", None)
        content = getattr(record, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        if message_type == "human":
            messages.append(HumanMessage(content=content))
        elif message_type == "ai" and not has_hidden_or_redacted_content(content):
            messages.append(AIMessage(content=content))
    return messages
