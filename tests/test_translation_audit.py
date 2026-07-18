import json
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from translation_audit import (
    JsonlTranslationAuditSink,
    TranslationAuditEvent,
    _build_default_sink,
    record_provider_exception,
    record_stored_outcome,
    safe_emit,
)


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self):
        self.status_code = 429
        self.headers = {
            "x-request-id": "req-safe_123",
            "Authorization": "Bearer secret-api-key",
            "Cookie": "secret-cookie",
        }


class _ProviderError(RuntimeError):
    def __init__(self):
        super().__init__(
            "secret中文源文 secret日本語 prompt=secret API_KEY=secret response-body"
        )
        self.response = _Response()


class _BrokenSink:
    def emit(self, event):
        raise RuntimeError("audit storage is unavailable")


class _HostileException(RuntimeError):
    @property
    def status_code(self):
        raise RuntimeError("unsafe status property")


class TranslationAuditTests(unittest.TestCase):
    def setUp(self):
        self.log_path = ROOT / f".test_translation_audit_{uuid4().hex}.log"

    def tearDown(self):
        for path in self.log_path.parent.glob(f"{self.log_path.name}*"):
            path.unlink(missing_ok=True)

    def test_provider_metadata_is_logged_without_sensitive_content(self):
        sink = JsonlTranslationAuditSink(self.log_path)
        try:
            record_provider_exception(
                sink,
                operation_id="op-safe-123",
                stage="translator",
                application_attempt=1,
                retry_scheduled=True,
                exception=_ProviderError(),
                issue_code="translator_exception",
            )
        finally:
            sink.close()

        raw = self.log_path.read_text(encoding="utf-8")
        event = json.loads(raw)

        self.assertEqual(event["schema_version"], 1)
        self.assertEqual(event["operation_id"], "op-safe-123")
        self.assertEqual(event["exception_type"], "_ProviderError")
        self.assertEqual(event["http_status"], 429)
        self.assertEqual(event["provider_request_id"], "req-safe_123")
        self.assertEqual(event["application_attempt"], 1)
        self.assertTrue(event["retry_scheduled"])
        for secret in (
            "secret中文源文",
            "secret日本語",
            "prompt=secret",
            "secret-api-key",
            "secret-cookie",
            "response-body",
        ):
            self.assertNotIn(secret, raw)

    def test_untrusted_identifiers_are_dropped_or_replaced(self):
        error = RuntimeError("do not serialize me")
        error.request_id = "sk-secret-request-id"
        error.status_code = 999
        sink = JsonlTranslationAuditSink(self.log_path)
        try:
            record_provider_exception(
                sink,
                operation_id="sk-secret-operation-id",
                stage="reviewer",
                application_attempt=-100,
                retry_scheduled=False,
                exception=error,
                issue_code="bad issue with spaces",
            )
        finally:
            sink.close()

        event = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertRegex(event["operation_id"], r"^[a-f0-9]{32}$")
        self.assertEqual(event["application_attempt"], 0)
        self.assertIsNone(event["http_status"])
        self.assertIsNone(event["provider_request_id"])
        self.assertIsNone(event["issue_code"])
        self.assertNotIn("serialize me", json.dumps(event))

    def test_sink_rejects_direct_events_that_bypass_safe_helpers(self):
        sink = JsonlTranslationAuditSink(self.log_path)
        unsafe_event = TranslationAuditEvent(
            timestamp_utc="2026-07-17T00:00:00.000Z",
            operation_id="这里是不能写入日志的原文",
            event="terminal_failure",
            stage="translator",
            application_attempt=1,
            retry_scheduled=False,
            issue_code="translator_exception",
        )
        try:
            with self.assertRaisesRegex(ValueError, "invalid translation audit event"):
                sink.emit(unsafe_event)
        finally:
            sink.close()

        self.assertFalse(self.log_path.exists())

    def test_rotation_keeps_every_line_valid_json(self):
        sink = JsonlTranslationAuditSink(
            self.log_path,
            max_bytes=350,
            backup_count=2,
        )
        try:
            for message_id in range(1, 13):
                record_stored_outcome(
                    sink,
                    operation_id=f"operation-{message_id}",
                    translation_status="validated",
                    issue_code=None,
                    message_id=message_id,
                )
        finally:
            sink.close()

        paths = sorted(self.log_path.parent.glob(f"{self.log_path.name}*"))
        self.assertGreaterEqual(len(paths), 2)
        self.assertLessEqual(len(paths), 3)
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                parsed = json.loads(line)
                self.assertEqual(parsed["event"], "stored_outcome")

    def test_audit_sink_failure_is_always_swallowed(self):
        event = TranslationAuditEvent(
            timestamp_utc="2026-07-17T00:00:00.000Z",
            operation_id="safe-op",
            event="terminal_failure",
            stage="translator",
            application_attempt=2,
            retry_scheduled=False,
            issue_code="translator_exception",
        )

        safe_emit(_BrokenSink(), event)

        record_provider_exception(
            _BrokenSink(),
            operation_id="safe-op",
            stage="translator",
            application_attempt=1,
            retry_scheduled=False,
            exception=_HostileException("must not escape"),
            issue_code="translator_exception",
        )

    def test_default_sink_initialization_failure_does_not_break_importers(self):
        with patch(
            "translation_audit.JsonlTranslationAuditSink",
            side_effect=PermissionError("read-only log directory"),
        ):
            self.assertIsNone(_build_default_sink())


if __name__ == "__main__":
    unittest.main()
