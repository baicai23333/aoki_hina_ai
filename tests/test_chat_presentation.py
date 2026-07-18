import unittest

from chat_presentation import (
    manual_translation_retry_available,
    translation_status_message,
)


class TranslationStatusMessageTests(unittest.TestCase):
    def test_rejected_and_failed_use_fixed_public_messages(self):
        self.assertEqual(
            translation_status_message("rejected"),
            "日语译文未通过安全复核，本条仅显示中文，未生成语音。",
        )
        self.assertEqual(
            translation_status_message("failed"),
            "日语翻译暂时失败，本条仅显示中文，未生成语音。",
        )

    def test_other_or_untrusted_statuses_are_hidden(self):
        statuses = (
            "validated",
            "fixed",
            "none",
            "legacy_unverified",
            "raw exception: secret path",
            None,
            123,
        )
        for status in statuses:
            with self.subTest(status=status):
                self.assertIsNone(translation_status_message(status))

    def test_manual_retry_is_only_available_for_transient_failed_states(self):
        retryable_issues = (
            "translator_exception",
            "reviewer_exception",
            "reviewer_invalid_json",
            "reviewer_invalid_schema",
        )
        for issue_code in retryable_issues:
            with self.subTest(issue_code=issue_code):
                self.assertTrue(
                    manual_translation_retry_available(
                        "failed",
                        issue_code,
                        source_has_hidden_content=False,
                    )
                )

    def test_manual_retry_hides_fixed_rejected_and_sensitive_failures(self):
        cases = (
            ("validated", "translator_exception", False),
            ("fixed", "translator_exception", False),
            ("rejected", "reviewer_rejected", False),
            ("failed", "fixed_source_mismatch", False),
            ("failed", "source_hidden_or_redacted", False),
            ("failed", "source_empty", False),
            ("failed", "translator_exception", True),
            ("failed", None, False),
        )
        for status, issue_code, hidden in cases:
            with self.subTest(status=status, issue_code=issue_code, hidden=hidden):
                self.assertFalse(
                    manual_translation_retry_available(
                        status,
                        issue_code,
                        source_has_hidden_content=hidden,
                    )
                )


if __name__ == "__main__":
    unittest.main()
