from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from runtime_context import RuntimeContextValidationError, build_runtime_context
from runtime_profile import RuntimeProfile


NOW = datetime(2026, 7, 16, 15, 30, tzinfo=timezone.utc)


def profile(**overrides) -> RuntimeProfile:
    values = {
        "username": "alice",
        "browser_timezone": "Asia/Shanghai",
        "browser_locale": "zh-CN",
        "home_city": "广州",
        "temporary_city": None,
        "temporary_city_expires_at": None,
        "coarse_latitude": None,
        "coarse_longitude": None,
        "coarse_coordinates_expires_at": None,
        "created_at": NOW - timedelta(days=1),
        "updated_at": NOW,
    }
    values.update(overrides)
    return RuntimeProfile(**values)


class RuntimeContextTests(unittest.TestCase):
    def test_server_utc_is_converted_with_browser_zoneinfo(self) -> None:
        context = build_runtime_context(profile(), now_utc=NOW)
        self.assertEqual(context.timezone_name, "Asia/Shanghai")
        self.assertEqual(context.timezone_source, "stored_browser")
        self.assertEqual(context.local_datetime.isoformat(), "2026-07-16T23:30:00+08:00")
        self.assertEqual(context.weekday_zh, "星期四")
        self.assertEqual(context.day_period_zh, "晚上")
        self.assertEqual(context.location.city, "广州")

    def test_current_browser_values_override_stored_values(self) -> None:
        context = build_runtime_context(
            profile(),
            browser_timezone="Asia/Tokyo",
            browser_locale="ja_JP",
            now_utc=NOW,
        )
        self.assertEqual(context.timezone_source, "current_browser")
        self.assertEqual(context.timezone_name, "Asia/Tokyo")
        self.assertEqual(context.locale, "ja-JP")
        self.assertEqual(context.local_datetime.isoformat(), "2026-07-17T00:30:00+09:00")

    def test_expired_transient_location_is_ignored_without_guessing(self) -> None:
        context = build_runtime_context(
            profile(
                temporary_city="杭州",
                temporary_city_expires_at=NOW - timedelta(seconds=1),
                coarse_latitude=23.13,
                coarse_longitude=113.26,
                coarse_coordinates_expires_at=NOW - timedelta(seconds=1),
            ),
            now_utc=NOW,
        )
        self.assertEqual(context.location.kind, "home_city")
        self.assertEqual(context.location.city, "广州")

        context = build_runtime_context(
            profile(
                home_city=None,
                temporary_city=None,
                temporary_city_expires_at=None,
            ),
            now_utc=NOW,
        )
        self.assertIsNone(context.location)

    def test_prompt_hides_authorized_coordinates_but_weather_payload_can_use_them(self) -> None:
        context = build_runtime_context(
            profile(
                home_city=None,
                coarse_latitude=23.13,
                coarse_longitude=113.26,
                coarse_coordinates_expires_at=NOW + timedelta(hours=1),
            ),
            now_utc=NOW,
        )
        prompt = context.to_prompt_text()
        self.assertNotIn("23.13", prompt)
        self.assertNotIn("113.26", prompt)
        self.assertIn("available_for_weather_tool", prompt)
        self.assertEqual(context.weather_location()["latitude"], 23.13)
        json.dumps(context.to_dict(), ensure_ascii=False)
        json.dumps(context.to_dict(include_coordinates=True), ensure_ascii=False)

    def test_missing_timezone_uses_labeled_utc_fallback_not_a_location_guess(self) -> None:
        context = build_runtime_context(None, now_utc=NOW)
        self.assertEqual(context.timezone_name, "UTC")
        self.assertEqual(context.timezone_source, "fallback")
        self.assertIsNone(context.location)

    def test_invalid_time_inputs_fail_closed(self) -> None:
        with self.assertRaises(RuntimeContextValidationError):
            build_runtime_context(profile(), browser_timezone="Mars/Olympus", now_utc=NOW)
        with self.assertRaises(RuntimeContextValidationError):
            build_runtime_context(profile(), browser_locale="bad locale!", now_utc=NOW)
        with self.assertRaises(RuntimeContextValidationError):
            build_runtime_context(profile(), now_utc=datetime(2026, 7, 16, 12, 0))


if __name__ == "__main__":
    unittest.main()
