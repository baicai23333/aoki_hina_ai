"""Build a small, non-sensitive trace for optional local pipeline debugging."""

from __future__ import annotations

import re
from typing import Any, Mapping


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_BOUNDARY_ACTIONS = {
    "none",
    "clarify_identity",
    "refuse_private",
    "insufficient_public_evidence",
}
_VALIDATION_CODES = {
    "clarify_identity",
    "refuse_private",
    "insufficient_public_evidence",
    "validator_invalid_schema",
    "validator_inconsistent_result",
    "validator_rejected_without_issue",
    "validator_invalid_json",
    "review_rejected_draft",
    "empty_draft",
    "claims_real_person_identity",
    "claims_private_activity",
    "unsupported_real_person_fact",
    "excessive_exclamation_marks",
    "unexpected_japanese_output",
    "hidden_or_redacted_content",
    "empty_or_too_short",
    "realtime_unavailable",
}
_TRANSLATION_STATUSES = {
    "not_requested",
    "fixed",
    "validated",
    "rejected",
    "failed",
}
_TTS_STATUSES = {"not_requested", "succeeded", "failed", "disabled"}
_DURATION_STAGES = ("pipeline", "translation", "tts")


def _safe_string_ids(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        value
        for value in values
        if isinstance(value, str) and _SAFE_ID.fullmatch(value)
    ][:20]


def _safe_memory_ids(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ][:20]


def _validation_codes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    codes: list[str] = []
    for value in values:
        code = (
            value
            if isinstance(value, str) and value in _VALIDATION_CODES
            else "model_review_rejected"
        )
        if code not in codes:
            codes.append(code)
    return codes


def build_debug_trace(
    result: Any,
    *,
    translation_status: str = "not_requested",
    tts_status: str = "not_requested",
    stage_duration_ms: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Return an allowlisted trace without text, prompts, plans, paths, or errors."""

    raw_intent = getattr(result, "intent", "unknown")
    intent = getattr(raw_intent, "value", raw_intent)
    if not isinstance(intent, str) or not _SAFE_ID.fullmatch(intent):
        intent = "unknown"
    plan = getattr(result, "plan", {})
    is_grounded = isinstance(plan, dict) and plan.get("grounded") is True
    is_runtime_time = (
        isinstance(plan, dict) and plan.get("runtime_time_answer") is True
    )
    if is_runtime_time:
        route = "deterministic_runtime"
    elif intent == "public_fact" and is_grounded:
        route = "grounded_public_fact"
    elif intent == "public_fact":
        route = "deterministic_public_fact"
    elif intent in {"identity_attack", "private_probe"}:
        route = "deterministic_boundary"
    else:
        route = "generated"

    raw_boundary = plan.get("boundary_action") if isinstance(plan, dict) else None
    boundary_action = raw_boundary if raw_boundary in _BOUNDARY_ACTIONS else "none"

    durations: dict[str, int] = {}
    for stage in _DURATION_STAGES:
        raw_value = (stage_duration_ms or {}).get(stage)
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            durations[stage] = max(0, min(int(round(raw_value)), 3_600_000))

    return {
        "intent": intent,
        "route": route,
        "evidence_ids": _safe_string_ids(getattr(result, "evidence_ids", [])),
        "fact_ids": _safe_string_ids(getattr(result, "fact_ids", [])),
        "memory_ids": _safe_memory_ids(getattr(result, "memory_ids", [])),
        "boundary_action": boundary_action,
        "validation_codes": _validation_codes(
            getattr(result, "validation_issues", [])
        ),
        "translation_status": (
            translation_status
            if translation_status in _TRANSLATION_STATUSES
            else "failed"
        ),
        "tts_status": tts_status if tts_status in _TTS_STATUSES else "failed",
        "stage_duration_ms": durations,
    }
