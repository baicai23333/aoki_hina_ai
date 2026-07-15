"""Side-effect-free credential verification for the admin UI.

The caller owns configuration loading.  This module deliberately does not read
environment variables, open the application database, or import Streamlit.
"""

from __future__ import annotations

import hmac

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


_PASSWORD_HASHER = PasswordHasher()


def verify_admin_credentials(
    username: str | None,
    password: str | None,
    configured_username: str | None,
    configured_password_hash: str | None,
) -> bool:
    """Return whether supplied credentials match the configured administrator.

    All values are supplied explicitly so verification stays independent of UI
    and configuration storage.  Invalid input and malformed Argon2 hashes fail
    closed instead of surfacing authentication-library errors.
    """

    values = (
        username,
        password,
        configured_username,
        configured_password_hash,
    )
    if any(not isinstance(value, str) or not value for value in values):
        return False

    # compare_digest only accepts ASCII str values, while application usernames
    # may contain any Unicode text.  Comparing their UTF-8 bytes supports both.
    username_matches = hmac.compare_digest(
        username.encode("utf-8"),
        configured_username.encode("utf-8"),
    )

    try:
        password_matches = _PASSWORD_HASHER.verify(
            configured_password_hash,
            password,
        )
    except (InvalidHashError, VerificationError):
        return False

    # Do not short-circuit before Argon2 verification when only the username is
    # wrong; that would create a much larger username-enumeration timing signal.
    return username_matches and password_matches
