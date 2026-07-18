"""Opt-in browser geolocation component for the Streamlit chat page.

The component never requests permission on page load.  A browser prompt is
opened only after the signed-in user clicks the component button.  Coordinates
are rounded to roughly city-neighbourhood precision before they are returned to
Python, and the server validates and rounds them again before persistence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "aoki_browser_geolocation",
    path=str(Path(__file__).resolve().parent),
)


class BrowserGeolocationError(ValueError):
    """Raised when a component payload does not match the public contract."""


def normalize_geolocation_result(value: object) -> dict[str, Any] | None:
    """Return a bounded, coarse location payload or ``None``.

    The browser is untrusted input.  Unknown fields are ignored, exact
    coordinates are never retained, and error messages are mapped to a small
    allowlist so arbitrary browser text cannot reach the UI.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise BrowserGeolocationError("geolocation result must be an object")

    status = value.get("status")
    if status == "error":
        code = value.get("code")
        allowed_codes = {
            "permission_denied",
            "position_unavailable",
            "timeout",
            "unsupported",
            "insecure_context",
            "unknown",
        }
        return {
            "status": "error",
            "code": code if code in allowed_codes else "unknown",
        }

    if status != "success":
        raise BrowserGeolocationError("unknown geolocation status")

    latitude = value.get("latitude")
    longitude = value.get("longitude")
    if isinstance(latitude, bool) or not isinstance(latitude, (int, float)):
        raise BrowserGeolocationError("latitude must be numeric")
    if isinstance(longitude, bool) or not isinstance(longitude, (int, float)):
        raise BrowserGeolocationError("longitude must be numeric")
    latitude = float(latitude)
    longitude = float(longitude)
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise BrowserGeolocationError("latitude is out of range")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise BrowserGeolocationError("longitude is out of range")

    accuracy = value.get("accuracy_m")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        accuracy_m = None
    else:
        numeric_accuracy = float(accuracy)
        accuracy_m = (
            int(min(max(round(numeric_accuracy / 100) * 100, 100), 100_000))
            if math.isfinite(numeric_accuracy) and numeric_accuracy >= 0
            else None
        )

    return {
        "status": "success",
        # Two decimals is sufficient for weather while avoiding storage of a
        # device's precise position (roughly one kilometre of resolution).
        "latitude": round(latitude, 2),
        "longitude": round(longitude, 2),
        "accuracy_m": accuracy_m,
    }


def browser_geolocation(*, key: str) -> dict[str, Any] | None:
    """Render the opt-in locator and return a validated component payload."""

    raw_value = _COMPONENT(key=key, default=None)
    return normalize_geolocation_result(raw_value)


__all__ = [
    "BrowserGeolocationError",
    "browser_geolocation",
    "normalize_geolocation_result",
]
