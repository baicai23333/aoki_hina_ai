"""Best-effort failed-login cooldown keyed to one Streamlit browser session."""

from __future__ import annotations

import threading
import time


_LOCK = threading.Lock()
_STATES: dict[str, dict[str, float]] = {}
_STATE_TTL_SECONDS = 3_600
_MAX_STATES = 10_000


def _prune_states(now: float) -> None:
    stale = [
        key
        for key, state in _STATES.items()
        if now - float(state.get("last_seen", now)) > _STATE_TTL_SECONDS
    ]
    for key in stale:
        _STATES.pop(key, None)
    if len(_STATES) >= _MAX_STATES:
        oldest = min(
            _STATES,
            key=lambda key: float(_STATES[key].get("last_seen", now)),
        )
        _STATES.pop(oldest, None)


def login_wait_seconds(fingerprint: str, now: float | None = None) -> int:
    current = time.monotonic() if now is None else now
    with _LOCK:
        _prune_states(current)
        state = _STATES.get(fingerprint, {})
        locked_until = float(state.get("locked_until", 0.0))
        return max(0, int(locked_until - current) + 1)


def record_login_failure(
    fingerprint: str,
    *,
    max_attempts: int = 5,
    lock_seconds: int = 30,
    now: float | None = None,
) -> bool:
    current = time.monotonic() if now is None else now
    with _LOCK:
        _prune_states(current)
        state = _STATES.setdefault(
            fingerprint,
            {"failures": 0.0, "locked_until": 0.0, "last_seen": current},
        )
        state["last_seen"] = current
        failures = int(state["failures"]) + 1
        if failures >= max_attempts:
            state["failures"] = 0.0
            state["locked_until"] = current + lock_seconds
            return True
        state["failures"] = float(failures)
        return False


def clear_login_failures(fingerprint: str) -> None:
    with _LOCK:
        _STATES.pop(fingerprint, None)


def _reset_login_guard_for_tests() -> None:
    with _LOCK:
        _STATES.clear()
