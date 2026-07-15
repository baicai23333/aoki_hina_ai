import hmac
import unittest
from unittest.mock import patch

from argon2 import PasswordHasher

from admin_auth import verify_admin_credentials


class AdminAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Low test-only work factors keep the suite fast. The encoded hash carries
        # its own parameters, so the production verifier can validate it normally.
        cls.password = "correct horse battery staple"
        cls.password_hash = PasswordHasher(
            time_cost=1,
            memory_cost=1024,
            parallelism=1,
        ).hash(cls.password)

    def test_matching_credentials_are_accepted(self):
        self.assertTrue(
            verify_admin_credentials(
                "site-admin",
                self.password,
                "site-admin",
                self.password_hash,
            )
        )

    def test_wrong_username_is_rejected(self):
        self.assertFalse(
            verify_admin_credentials(
                "another-user",
                self.password,
                "site-admin",
                self.password_hash,
            )
        )

    def test_wrong_password_is_rejected(self):
        self.assertFalse(
            verify_admin_credentials(
                "site-admin",
                "wrong password",
                "site-admin",
                self.password_hash,
            )
        )

    def test_unicode_username_uses_constant_time_byte_comparison(self):
        with patch("admin_auth.hmac.compare_digest", wraps=hmac.compare_digest) as compare:
            accepted = verify_admin_credentials(
                "管理员",
                self.password,
                "管理员",
                self.password_hash,
            )

        self.assertTrue(accepted)
        compare.assert_called_once_with("管理员".encode("utf-8"), "管理员".encode("utf-8"))

    def test_valid_hash_is_checked_even_when_username_is_wrong(self):
        with patch.object(
            PasswordHasher,
            "verify",
            autospec=True,
            return_value=True,
        ) as verify:
            accepted = verify_admin_credentials(
                "another-user",
                self.password,
                "site-admin",
                self.password_hash,
            )

        self.assertFalse(accepted)
        verify.assert_called_once()

    def test_missing_or_empty_values_fail_closed(self):
        valid_values = [
            "site-admin",
            self.password,
            "site-admin",
            self.password_hash,
        ]
        for index in range(len(valid_values)):
            for missing in (None, ""):
                values = valid_values.copy()
                values[index] = missing
                with self.subTest(index=index, missing=missing):
                    self.assertFalse(verify_admin_credentials(*values))

    def test_non_string_values_fail_closed(self):
        valid_values = [
            "site-admin",
            self.password,
            "site-admin",
            self.password_hash,
        ]
        for index in range(len(valid_values)):
            values = valid_values.copy()
            values[index] = b"not-a-string"
            with self.subTest(index=index):
                self.assertFalse(verify_admin_credentials(*values))

    def test_malformed_argon2_hash_fails_closed(self):
        for bad_hash in (
            "not-an-argon2-hash",
            "$argon2id$broken",
            self.password_hash[:-8],
        ):
            with self.subTest(bad_hash=bad_hash):
                self.assertFalse(
                    verify_admin_credentials(
                        "site-admin",
                        self.password,
                        "site-admin",
                        bad_hash,
                    )
                )


if __name__ == "__main__":
    unittest.main()
