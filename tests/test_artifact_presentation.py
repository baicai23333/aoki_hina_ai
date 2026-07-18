import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from artifact_presentation import (
    OPEN_METEO_URL,
    SearchStatusView,
    SourceCardsView,
    WeatherCardView,
    normalize_artifact,
    normalize_message_artifact,
    normalize_message_artifacts,
    normalize_search_status,
    normalize_source_cards,
    normalize_weather_card,
    safe_http_url,
)


class ArtifactPresentationTests(unittest.TestCase):
    def test_weather_normalization_is_chinese_friendly_and_privacy_allowlisted(self):
        payload = {
            "latitude": 23.13,
            "longitude": 113.26,
            "accuracy": 25,
            "raw_content": "SECRET RAW PAGE",
            "snippet": "SECRET SNIPPET",
            "internal_error": "SECRET TRACEBACK",
            "location": {
                "name": "广州",
                "timezone": "Asia/Shanghai",
                "latitude": 23.13,
                "coordinates": [23.13, 113.26],
            },
            "current": {
                "weather": "多云",
                "temperature_c": 31.2,
                "apparent_temperature_c": 36.1,
                "precipitation_mm": 0,
                "wind_speed_kmh": 8.4,
                "longitude": 113.26,
            },
            "forecast": [
                {
                    "date": "2026-07-18",
                    "weather": "阵雨",
                    "min_temperature_c": 27,
                    "max_temperature_c": 34,
                    "precipitation_probability_percent": 65,
                    "coordinates": [23.13, 113.26],
                }
            ],
            "fetched_at": "2026-07-17T00:20:00Z",
        }

        view = normalize_weather_card(payload)

        self.assertIsInstance(view, WeatherCardView)
        assert view is not None
        self.assertEqual(view.location, "广州")
        self.assertEqual(view.weather, "多云")
        self.assertEqual(view.temperature_c, 31.2)
        self.assertEqual(view.updated_label, "2026年07月17日 08:20")
        self.assertEqual(view.forecast[0].date_label, "07月18日")
        self.assertEqual(view.attribution_url, OPEN_METEO_URL)
        serialized = json.dumps(asdict(view), ensure_ascii=False).casefold()
        for forbidden in (
            "latitude",
            "longitude",
            "coordinates",
            "accuracy",
            "raw_content",
            "secret raw page",
            "snippet",
            "secret snippet",
            "internal_error",
            "secret traceback",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_weather_malformed_fields_are_ignored_without_crashing(self):
        view = normalize_weather_card(
            {
                "location": {"name": "杭州", "timezone": "Invalid/Timezone"},
                "current": {
                    "temperature_c": float("nan"),
                    "precipitation_mm": -2,
                },
                "forecast": [
                    {
                        "date": "not-a-date",
                        "min_temperature_c": 40,
                        "max_temperature_c": 20,
                    },
                    "bad",
                ],
                "fetched_at": "not-a-datetime",
            }
        )

        self.assertIsInstance(view, WeatherCardView)
        assert view is not None
        self.assertEqual(view.location, "杭州")
        self.assertIsNone(view.temperature_c)
        self.assertEqual(view.forecast, ())
        self.assertIsNone(normalize_weather_card({"raw_content": "only raw data"}))
        self.assertIsNone(normalize_weather_card("not an object"))

    def test_source_cards_only_keep_safe_links_and_display_metadata(self):
        payload = {
            "items": [
                {
                    "title": "官方活动公告",
                    "url": "https://bm-echoes.com/news/1",
                    "published_at": "2026-07-16T12:30:00+09:00",
                    "approved": True,
                    "snippet": "DO NOT DISPLAY THIS SNIPPET",
                    "raw_content": "DO NOT DISPLAY RAW CONTENT",
                },
                {
                    "title": "官方企划新闻",
                    "url": "http://bang-dream.com/news/1",
                    "published_at": "2026-07-15",
                    "official_source": "bang_dream",
                },
                {"title": "脚本链接", "url": "javascript:alert(1)"},
                {"title": "本地文件", "url": "file:///etc/passwd"},
                {"title": "带凭据", "url": "https://user:secret@example.com/"},
                {"title": "缺少链接", "snippet": "https://example.com/"},
            ]
        }

        view = normalize_source_cards(payload)

        self.assertIsInstance(view, SourceCardsView)
        assert view is not None
        self.assertEqual(len(view.items), 2)
        self.assertEqual(view.items[0].confirmation_label, "官方确认")
        self.assertEqual(view.items[0].published_label, "2026年07月16日 12:30")
        self.assertEqual(view.items[1].confirmation_label, "官方来源")
        serialized = json.dumps(asdict(view), ensure_ascii=False)
        self.assertNotIn("DO NOT DISPLAY THIS SNIPPET", serialized)
        self.assertNotIn("DO NOT DISPLAY RAW CONTENT", serialized)
        self.assertNotIn("javascript:", serialized)
        self.assertNotIn("user:secret", serialized)

    def test_safe_http_url_rejects_non_web_and_malformed_links(self):
        self.assertEqual(safe_http_url("https://example.com/a"), "https://example.com/a")
        self.assertEqual(safe_http_url("http://example.com"), "http://example.com")
        for unsafe in (
            "javascript:alert(1)",
            "data:text/html,x",
            "file:///tmp/x",
            "//example.com/path",
            "https://user:password@example.com/",
            "https://example.com\\@evil.test/",
            "https://example.com/\nnext",
            "https://example.com:bad/",
            "",
            None,
        ):
            self.assertIsNone(safe_http_url(unsafe))

    def test_search_status_never_displays_internal_diagnostics(self):
        view = normalize_search_status(
            {
                "name": "get_weather",
                "ok": False,
                "error_code": "tool_execution_failed",
                "exception": "requests failed with API_KEY=secret",
                "raw_content": "traceback",
            }
        )

        self.assertEqual(
            view,
            SearchStatusView("天气查询暂时失败，请稍后再试。", level="warning"),
        )
        serialized = json.dumps(asdict(view), ensure_ascii=False)
        self.assertNotIn("API_KEY", serialized)
        self.assertNotIn("tool_execution_failed", serialized)
        self.assertEqual(
            normalize_search_status(
                {"kind": "recent_updates", "count": 3, "approved_count": 2}
            ),
            SearchStatusView("近期信息共找到 3 条，其中 2 条已审核。"),
        )

    def test_unknown_or_malformed_artifacts_are_safely_skipped(self):
        self.assertIsNone(normalize_artifact("raw_html", {"html": "<b>x</b>"}))
        self.assertIsNone(normalize_artifact("source_cards", {"items": "bad"}))
        self.assertIsNone(normalize_artifact(None, {}))
        self.assertEqual(normalize_message_artifacts(None), ())

        artifacts = [
            SimpleNamespace(
                artifact_type="search_status",
                payload={"ok": True, "count": 2},
            ),
            {"artifact_type": "unknown", "payload": {}},
            {"artifact_type": "weather_card", "payload": []},
            object(),
        ]
        normalized = normalize_message_artifacts(artifacts)
        self.assertEqual(
            normalized,
            (SearchStatusView("即时信息查询完成，共找到 2 条结果。"),),
        )
        self.assertEqual(normalize_message_artifact(artifacts[0]), normalized[0])


if __name__ == "__main__":
    unittest.main()
