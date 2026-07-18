"""Privacy-safe audit events for the response translation pipeline.

The public helpers in this module deliberately accept only fixed metadata.
Source text, translated text, prompts, raw exceptions, and credentials must
never be passed to the audit sink.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal, Protocol, TypeGuard
from uuid import uuid4


AuditEventName = Literal["provider_exception", "terminal_failure", "stored_outcome"]
AuditStage = Literal[
    "translator",
    "translator_repair",
    "reviewer",
    "orchestration",
    "storage",
]

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SAFE_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_SAFE_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
_SENSITIVE_TOKEN_PREFIXES = ("sk-", "sk_", "bearer", "api-key", "apikey")
_AUDIT_EVENTS = frozenset({"provider_exception", "terminal_failure", "stored_outcome"})
_AUDIT_STAGES = frozenset(
    {"translator", "translator_repair", "reviewer", "orchestration", "storage"}
)
_TRANSLATION_STATUSES = frozenset({"validated", "fixed", "rejected", "failed"})


@dataclass(frozen=True)
class TranslationAuditEvent:
    timestamp_utc: str
    operation_id: str
    event: AuditEventName
    stage: AuditStage
    application_attempt: int
    retry_scheduled: bool
    exception_type: str | None = None
    http_status: int | None = None
    provider_request_id: str | None = None
    translation_status: str | None = None
    issue_code: str | None = None
    message_id: int | None = None
    schema_version: int = 1


class TranslationAuditSink(Protocol):
    def emit(self, event: TranslationAuditEvent) -> None: ...


class JsonlTranslationAuditSink:
    """Thread-safe rotating JSONL sink with one fixed event schema."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handler = RotatingFileHandler(
            self.path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._lock = threading.RLock()

    def emit(self, event: TranslationAuditEvent) -> None:
        if not isinstance(event, TranslationAuditEvent):
            raise TypeError("event must be a TranslationAuditEvent")
        _validate_event(event)
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        record = logging.LogRecord(
            name="aoki.translation_audit",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=line,
            args=(),
            exc_info=None,
        )
        with self._lock:
            self._handler.emit(record)
            self._handler.flush()
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._handler.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def new_translation_operation_id() -> str:
    return uuid4().hex


def _safe_token(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and _SAFE_TOKEN.fullmatch(value) is not None
        and not value.lower().startswith(_SENSITIVE_TOKEN_PREFIXES)
    )


def normalize_translation_operation_id(value: object) -> str:
    if _safe_token(value):
        return value
    return new_translation_operation_id()


def _safe_exception_type(exception: BaseException) -> str | None:
    name = type(exception).__name__
    return name if _SAFE_EXCEPTION_TYPE.fullmatch(name) else None


def _safe_http_status(exception: BaseException) -> int | None:
    value = getattr(exception, "status_code", None)
    if value is None:
        response = getattr(exception, "response", None)
        value = getattr(response, "status_code", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 100 <= value <= 599 else None


def _safe_request_id(exception: BaseException) -> str | None:
    value = getattr(exception, "request_id", None)
    if value is None:
        response = getattr(exception, "response", None)
        headers = getattr(response, "headers", None)
        getter = getattr(headers, "get", None)
        if callable(getter):
            value = getter("x-request-id") or getter("X-Request-ID")
    if not _safe_token(value):
        return None
    return value


def _safe_issue_code(value: object) -> str | None:
    if _safe_token(value):
        return value
    return None


def _safe_attempt(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return min(max(value, 0), 10_000)


def _safe_message_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _validate_event(event: TranslationAuditEvent) -> None:
    valid = (
        event.schema_version == 1
        and isinstance(event.timestamp_utc, str)
        and _SAFE_TIMESTAMP.fullmatch(event.timestamp_utc) is not None
        and _safe_token(event.operation_id)
        and event.event in _AUDIT_EVENTS
        and event.stage in _AUDIT_STAGES
        and isinstance(event.application_attempt, int)
        and not isinstance(event.application_attempt, bool)
        and 0 <= event.application_attempt <= 10_000
        and type(event.retry_scheduled) is bool
        and (
            event.exception_type is None
            or (
                isinstance(event.exception_type, str)
                and _SAFE_EXCEPTION_TYPE.fullmatch(event.exception_type) is not None
            )
        )
        and (
            event.http_status is None
            or (
                isinstance(event.http_status, int)
                and not isinstance(event.http_status, bool)
                and 100 <= event.http_status <= 599
            )
        )
        and (
            event.provider_request_id is None
            or _safe_token(event.provider_request_id)
        )
        and (
            event.translation_status is None
            or event.translation_status in _TRANSLATION_STATUSES
        )
        and (event.issue_code is None or _safe_token(event.issue_code))
        and (
            event.message_id is None
            or (
                isinstance(event.message_id, int)
                and not isinstance(event.message_id, bool)
                and event.message_id > 0
            )
        )
    )
    if not valid:
        raise ValueError("invalid translation audit event")


def safe_emit(
    sink: TranslationAuditSink | None,
    event: TranslationAuditEvent,
) -> None:
    """Audit failures must never change the fail-closed translation result."""

    if sink is None:
        return
    try:
        sink.emit(event)
    except Exception:
        pass


def record_provider_exception(
    sink: TranslationAuditSink | None,
    *,
    operation_id: str,
    stage: AuditStage,
    application_attempt: int,
    retry_scheduled: bool,
    exception: BaseException,
    issue_code: str,
) -> None:
    try:
        event = TranslationAuditEvent(
            timestamp_utc=_utc_now(),
            operation_id=normalize_translation_operation_id(operation_id),
            event="provider_exception",
            stage=stage,
            application_attempt=_safe_attempt(application_attempt),
            retry_scheduled=bool(retry_scheduled),
            exception_type=_safe_exception_type(exception),
            http_status=_safe_http_status(exception),
            provider_request_id=_safe_request_id(exception),
            issue_code=_safe_issue_code(issue_code),
        )
    except Exception:
        return
    safe_emit(
        sink,
        event,
    )


def record_terminal_failure(
    sink: TranslationAuditSink | None,
    *,
    operation_id: str,
    stage: AuditStage,
    application_attempt: int,
    issue_code: str,
) -> None:
    try:
        event = TranslationAuditEvent(
            timestamp_utc=_utc_now(),
            operation_id=normalize_translation_operation_id(operation_id),
            event="terminal_failure",
            stage=stage,
            application_attempt=_safe_attempt(application_attempt),
            retry_scheduled=False,
            translation_status="failed",
            issue_code=_safe_issue_code(issue_code),
        )
    except Exception:
        return
    safe_emit(
        sink,
        event,
    )


def record_stored_outcome(
    sink: TranslationAuditSink | None,
    *,
    operation_id: str,
    translation_status: str,
    issue_code: str | None,
    message_id: int,
) -> None:
    try:
        status = (
            translation_status
            if isinstance(translation_status, str)
            and translation_status in _TRANSLATION_STATUSES
            else None
        )
        event = TranslationAuditEvent(
            timestamp_utc=_utc_now(),
            operation_id=normalize_translation_operation_id(operation_id),
            event="stored_outcome",
            stage="storage",
            application_attempt=0,
            retry_scheduled=False,
            translation_status=status,
            issue_code=_safe_issue_code(issue_code),
            message_id=_safe_message_id(message_id),
        )
    except Exception:
        return
    safe_emit(
        sink,
        event,
    )


DEFAULT_TRANSLATION_AUDIT_PATH = (
    Path(__file__).resolve().parent / "logs" / "translation_audit.log"
)


def _build_default_sink() -> TranslationAuditSink | None:
    try:
        return JsonlTranslationAuditSink(DEFAULT_TRANSLATION_AUDIT_PATH)
    except Exception:
        return None


DEFAULT_TRANSLATION_AUDIT_SINK = _build_default_sink()
