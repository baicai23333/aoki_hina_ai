"""Pure presentation helpers shared by the Streamlit chat views."""

from __future__ import annotations

from chat_storage import MANUALLY_RETRYABLE_TRANSLATION_ISSUES


_TRANSLATION_STATUS_MESSAGES = {
    "rejected": "日语译文未通过安全复核，本条仅显示中文，未生成语音。",
    "failed": "日语翻译暂时失败，本条仅显示中文，未生成语音。",
}


def translation_status_message(status: object) -> str | None:
    """Return a fixed public message without exposing internal failure details."""

    if not isinstance(status, str):
        return None
    return _TRANSLATION_STATUS_MESSAGES.get(status)


def manual_translation_retry_available(
    status: object,
    issue_code: object,
    *,
    source_has_hidden_content: bool,
) -> bool:
    """Return whether a failed historical translation may be retried safely."""

    return (
        status == "failed"
        and isinstance(issue_code, str)
        and issue_code in MANUALLY_RETRYABLE_TRANSLATION_ISSUES
        and source_has_hidden_content is False
    )
