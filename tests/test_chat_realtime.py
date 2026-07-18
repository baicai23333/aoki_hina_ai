import unittest
from datetime import datetime, timedelta, timezone

from chat_realtime import (
    build_realtime_unavailable_bundle,
    build_weather_lookup,
    realtime_grounding_is_unavailable,
    serialize_approved_updates,
)
from grounding import GroundedFact, GroundingBundle
from information_store import OfficialUpdate
from runtime_context import RuntimeContext
from runtime_profile import RuntimeLocation
from weather_service import WeatherService


class FakeWeatherService(WeatherService):
    def __init__(self):
        self.calls = []

    def get_weather(self, **kwargs):
        self.calls.append(kwargs)
        return {"location": {"name": "ok"}, "current": {}, "forecast": []}


def context(location=None):
    now = datetime(2026, 7, 16, 4, 0, tzinfo=timezone.utc)
    return RuntimeContext(
        utc_datetime=now,
        local_datetime=now.astimezone(timezone(timedelta(hours=8))),
        timezone_name="Asia/Shanghai",
        timezone_source="current_browser",
        locale="zh-CN",
        location=location,
    )


class ChatRealtimeTests(unittest.TestCase):
    def test_orchestrator_failure_bundle_is_safe_and_actionable(self):
        bundle = build_realtime_unavailable_bundle("weather")

        self.assertFalse(bundle.facts[0].fact_eligible)
        self.assertEqual(bundle.ui_artifacts[0].artifact_type, "search_status")
        self.assertEqual(
            bundle.ui_artifacts[0].payload["error_code"],
            "tool_execution_failed",
        )
        self.assertTrue(realtime_grounding_is_unavailable(bundle, "weather"))

    def test_general_search_results_can_remain_unconfirmed_without_failing(self):
        bundle = GroundingBundle(
            facts=(
                GroundedFact(
                    "普通搜索结果摘要",
                    untrusted=True,
                    fact_eligible=False,
                ),
            )
        )

        self.assertFalse(
            realtime_grounding_is_unavailable(bundle, "web_search")
        )

        self.assertTrue(
            realtime_grounding_is_unavailable(bundle, "official_search")
        )

    def test_message_city_wins_without_using_saved_coordinates(self):
        service = FakeWeatherService()
        lookup = build_weather_lookup(
            service,
            context(RuntimeLocation("authorized_coarse_coordinates", latitude=23.1, longitude=113.3)),
        )

        lookup(location="杭州", days=2)

        self.assertEqual(service.calls, [{"city": "杭州", "days": 2}])

    def test_saved_city_is_used_only_after_tool_gate(self):
        service = FakeWeatherService()
        lookup = build_weather_lookup(service, context(RuntimeLocation("home_city", city="广州")))

        lookup(location=None, days=3)

        self.assertEqual(service.calls, [{"city": "广州", "days": 3}])

    def test_authorized_coarse_coordinates_keep_browser_timezone(self):
        service = FakeWeatherService()
        lookup = build_weather_lookup(
            service,
            context(RuntimeLocation("authorized_coarse_coordinates", latitude=23.13, longitude=113.26)),
        )

        lookup(location=None, days=1)

        self.assertEqual(
            service.calls,
            [{
                "latitude": 23.13,
                "longitude": 113.26,
                "timezone_name": "Asia/Shanghai",
                "days": 1,
            }],
        )

    def test_only_approved_update_fields_are_serialized(self):
        base = dict(
            id=1,
            document_id=2,
            source_id=3,
            source_name="MyGO official",
            canonical_url="https://bang-dream.com/news/1",
            published_at="2026-07-16T00:00:00Z",
            category="live",
            title="Live notice",
            summary="Official summary",
            event_start_at="2026-08-01T10:00:00Z",
            event_end_at=None,
            venue="Tokyo",
            status="scheduled",
            confidence=0.99,
            replaces_update_id=None,
            reviewed_by="admin-secret",
            reviewed_at="2026-07-16T01:00:00Z",
            created_at="2026-07-16T00:00:00Z",
            updated_at="2026-07-16T01:00:00Z",
        )
        approved = OfficialUpdate(verification_status="approved", **base)
        pending = OfficialUpdate(verification_status="pending", **{**base, "id": 2})

        payload = serialize_approved_updates([approved, pending])

        self.assertEqual(len(payload["updates"]), 1)
        self.assertTrue(payload["updates"][0]["approved"])
        self.assertIn("replaces_update_id", payload["updates"][0])
        self.assertNotIn("reviewed_by", payload["updates"][0])
        self.assertNotIn("confidence", payload["updates"][0])


if __name__ == "__main__":
    unittest.main()
