import unittest
from pathlib import Path

from admin_login_guard import (
    _reset_login_guard_for_tests,
    clear_login_failures,
    login_wait_seconds,
    record_login_failure,
)
from admin_page import (
    _configuration_issue,
    _credential_fingerprint,
    _session_matches_credentials,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$YW5lbmNvZGVkaGFzaA"


class AdminPageSecurityTests(unittest.TestCase):
    def test_configuration_rejects_missing_oversized_and_malformed_values(self):
        self.assertIsNotNone(_configuration_issue(None, None))
        self.assertIsNotNone(_configuration_issue(" admin", VALID_HASH))
        self.assertIsNotNone(_configuration_issue("a" * 257, VALID_HASH))
        self.assertIsNotNone(_configuration_issue("admin", "plaintext"))
        self.assertIsNone(_configuration_issue("admin", VALID_HASH))

    def test_rotating_the_hash_invalidates_the_previous_session_fingerprint(self):
        old_fingerprint = _credential_fingerprint("admin", VALID_HASH)
        self.assertTrue(
            _session_matches_credentials(
                "admin", old_fingerprint, "admin", VALID_HASH
            )
        )
        rotated_hash = VALID_HASH + "x"
        self.assertFalse(
            _session_matches_credentials(
                "admin", old_fingerprint, "admin", rotated_hash
            )
        )
        self.assertFalse(
            _session_matches_credentials(
                "other", old_fingerprint, "admin", VALID_HASH
            )
        )

    def test_sensitive_content_is_not_rendered_through_markdown_capable_calls(self):
        source = (ROOT / "admin_page.py").read_text(encoding="utf-8")
        self.assertNotIn("st.write(item.content", source)
        self.assertNotIn("st.write(item.japanese_content", source)
        self.assertNotIn("st.tabs(", source)
        self.assertNotIn("AOKI_ADMIN_DATABASE_PATH", source)
        self.assertNotIn("st.context.ip_address", source)

    def test_failed_login_guard_tracks_one_opaque_guard_key(self):
        _reset_login_guard_for_tests()
        fingerprint = "test-fingerprint"
        for _ in range(4):
            self.assertFalse(record_login_failure(fingerprint, now=100.0))
        self.assertTrue(record_login_failure(fingerprint, now=100.0))
        self.assertGreater(login_wait_seconds(fingerprint, now=100.0), 0)
        clear_login_failures(fingerprint)
        self.assertEqual(login_wait_seconds(fingerprint, now=100.0), 0)


if __name__ == "__main__":
    unittest.main()
