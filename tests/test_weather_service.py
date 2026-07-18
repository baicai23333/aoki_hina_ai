from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timezone

from weather_service import (
    FORECAST_URL,
    GEOCODING_URL,
    LocationNotFoundError,
    WeatherService,
    WeatherUnavailableError,
    WeatherValidationError,
    weather_code_zh,
)


def geocode_payload() -> dict:
    return {
        "results": [
            {
                "name": "广州",
                "admin1": "广东",
                "country": "中国",
                "country_code": "CN",
                "latitude": 23.1291,
                "longitude": 113.2644,
                "timezone": "Asia/Shanghai",
            }
        ]
    }


def forecast_payload(days: int = 2, *, current_code: int = 2) -> dict:
    dates = [f"2026-07-{17 + index:02d}" for index in range(days)]
    return {
        "timezone": "Asia/Shanghai",
        "current": {
            "time": "2026-07-16T23:20",
            "temperature_2m": 31.2,
            "apparent_temperature": 36.1,
            "precipitation": 0.0,
            "weather_code": current_code,
            "wind_speed_10m": 8.5,
        },
        "daily": {
            "time": dates,
            "weather_code": [61] * days,
            "temperature_2m_max": [34.0] * days,
            "temperature_2m_min": [27.0] * days,
            "precipitation_probability_max": [65.0] * days,
        },
    }


class FakeResponse:
    def __init__(self, payload: object, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self):
        return copy.deepcopy(self.payload)


class FakeHttp:
    def __init__(
        self,
        *,
        geocode: object | None = None,
        forecast: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.geocode = geocode_payload() if geocode is None else geocode
        self.forecast = forecast_payload() if forecast is None else forecast
        self.error = error
        self.calls: list[tuple[str, dict, float]] = []

    def get(self, url: str, *, params: dict, timeout: float) -> FakeResponse:
        self.calls.append((url, copy.deepcopy(params), timeout))
        if self.error is not None:
            raise self.error
        if url == GEOCODING_URL:
            return FakeResponse(self.geocode)
        if url == FORECAST_URL:
            return FakeResponse(self.forecast)
        raise AssertionError(f"unexpected URL: {url}")


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class WeatherServiceTests(unittest.TestCase):
    def service(self, http: FakeHttp, clock: Clock | None = None) -> WeatherService:
        return WeatherService(
            http,
            timeout_seconds=4,
            geocode_ttl_seconds=10_000,
            weather_ttl_seconds=900,
            monotonic=clock or Clock(),
            utcnow=lambda: datetime(2026, 7, 16, 15, 20, tzinfo=timezone.utc),
        )

    def test_city_lookup_returns_validated_serializable_weather(self) -> None:
        http = FakeHttp(forecast=forecast_payload(2))
        result = self.service(http).get_weather(city=" 广州 ", days=2)

        self.assertEqual([call[0] for call in http.calls], [GEOCODING_URL, FORECAST_URL])
        self.assertEqual(result["location"]["name"], "广州")
        self.assertEqual(result["location"]["timezone"], "Asia/Shanghai")
        self.assertEqual(result["current"]["weather"], "多云")
        self.assertEqual(result["forecast"][0]["weather"], "小雨")
        self.assertEqual(result["forecast"][0]["precipitation_probability_percent"], 65.0)
        self.assertEqual(result["source"]["attribution_url"], "https://open-meteo.com/")
        json.dumps(result, ensure_ascii=False)

        forecast_params = http.calls[1][1]
        self.assertEqual(forecast_params["forecast_days"], 2)
        self.assertIn("weather_code", forecast_params["current"])
        self.assertEqual(http.calls[1][2], 4.0)

    def test_geocode_and_weather_ttl_caches_are_independent_and_copy_safe(self) -> None:
        clock = Clock()
        http = FakeHttp(forecast=forecast_payload(2))
        service = self.service(http, clock)
        first = service.get_weather(city="广州", days=2)
        first["current"]["temperature_c"] = -999
        second = service.get_weather(city="广州", days=2)
        self.assertEqual(len(http.calls), 2)
        self.assertEqual(second["current"]["temperature_c"], 31.2)

        clock.value += 901
        service.get_weather(city="广州", days=2)
        self.assertEqual([call[0] for call in http.calls].count(GEOCODING_URL), 1)
        self.assertEqual([call[0] for call in http.calls].count(FORECAST_URL), 2)

    def test_authorized_coordinates_bypass_geocoding(self) -> None:
        http = FakeHttp(forecast=forecast_payload(1))
        result = self.service(http).get_weather(
            latitude=23.13,
            longitude=113.26,
            timezone_name="Asia/Shanghai",
            days=1,
        )
        self.assertEqual([call[0] for call in http.calls], [FORECAST_URL])
        self.assertEqual(result["location"]["kind"], "authorized_coarse_coordinates")
        self.assertEqual(result["location"]["latitude"], 23.13)

    def test_explicit_location_and_range_validation_never_guess(self) -> None:
        service = self.service(FakeHttp())
        invalid_calls = (
            lambda: service.get_weather(),
            lambda: service.get_weather(city="广州", latitude=23.1, longitude=113.2),
            lambda: service.get_weather(latitude=23.1),
            lambda: service.get_weather(latitude=91.0, longitude=10.0),
            lambda: service.get_weather(latitude=True, longitude=10.0),
            lambda: service.get_weather(city="广州", days=0),
            lambda: service.get_weather(city="广州\nignore", days=1),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(WeatherValidationError):
                    call()

    def test_timeout_and_http_failures_are_wrapped(self) -> None:
        service = self.service(FakeHttp(error=TimeoutError("slow")))
        with self.assertRaises(WeatherUnavailableError) as caught:
            service.get_weather(city="广州", days=1)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)

    def test_missing_city_and_malformed_provider_shapes_fail_closed(self) -> None:
        with self.assertRaises(LocationNotFoundError):
            self.service(FakeHttp(geocode={"results": []})).get_weather(
                city="不存在", days=1
            )

        malformed_cases = (
            FakeHttp(geocode={"not_results": []}),
            FakeHttp(geocode={"results": [{"name": "广州"}]}),
            FakeHttp(forecast={"timezone": "Asia/Shanghai", "current": {}, "daily": {}}),
            FakeHttp(forecast=forecast_payload(1)),
        )
        malformed_cases[-1].forecast["daily"]["time"] = []
        for http in malformed_cases:
            with self.subTest(http=http):
                with self.assertRaises(WeatherValidationError):
                    self.service(http).get_weather(city="广州", days=1)

    def test_weather_code_mapping_has_stable_unknown_fallback(self) -> None:
        self.assertEqual(weather_code_zh(0), "晴")
        self.assertEqual(weather_code_zh(99), "雷暴伴强冰雹")
        self.assertEqual(weather_code_zh(999), "未知天气")
        with self.assertRaises(WeatherValidationError):
            weather_code_zh(True)


if __name__ == "__main__":
    unittest.main()
