"""Pure presentation helpers shared by the Streamlit chat views."""

from __future__ import annotations


_TRANSLATION_STATUS_MESSAGES = {
    "rejected": "日语译文未通过安全复核，本条仅显示中文，未生成语音。",
    "failed": "日语翻译暂时失败，本条仅显示中文，未生成语音。",
}


def translation_status_message(status: object) -> str | None:
    """Return a fixed public message without exposing internal failure details."""

    if not isinstance(status, str):
        return None
    return _TRANSLATION_STATUS_MESSAGES.get(status)
