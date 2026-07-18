"""Validated Open-Meteo geocoding and weather forecasts with bounded caches."""

from __future__ import annotations

import copy
import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION_URL = "https://open-meteo.com/"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_GEOCODE_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_WEATHER_TTL_SECONDS = 15 * 60
MAX_CITY_LENGTH = 120

WEATHER_CODE_ZH = {
    0: "晴",
    1: "晴间多云",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "轻微冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴轻微冰雹",
    99: "雷暴伴强冰雹",
}

_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,8}$")


class WeatherServiceError(RuntimeError):
    """Base exception for weather lookups."""


class WeatherValidationError(ValueError):
    """Raised when caller input or provider output is invalid."""


class WeatherUnavailableError(WeatherServiceError):
    """Raised when Open-Meteo cannot be reached or returns an HTTP failure."""


class LocationNotFoundError(WeatherServiceError):
    """Raised when a manually supplied city cannot be geocoded."""


@dataclass(frozen=True)
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


def weather_code_zh(code: int) -> str:
    """Map an Open-Meteo WMO weather code to a stable Chinese label."""

    if isinstance(code, bool) or not isinstance(code, int):
        raise WeatherValidationError("weather code must be an integer")
    return WEATHER_CODE_ZH.get(code, "未知天气")


class WeatherService:
    """Small synchronous client suitable for Streamlit and deterministic tests."""

    def __init__(
        self,
        http_client: Any = requests,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        geocode_ttl_seconds: float = DEFAULT_GEOCODE_TTL_SECONDS,
        weather_ttl_seconds: float = DEFAULT_WEATHER_TTL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = http_client
        self.timeout_seconds = self._positive_number(timeout_seconds, "timeout_seconds")
        self.geocode_ttl_seconds = self._positive_number(
            geocode_ttl_seconds, "geocode_ttl_seconds"
        )
        self.weather_ttl_seconds = self._positive_number(
            weather_ttl_seconds, "weather_ttl_seconds"
        )
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        if utcnow is not None and not callable(utcnow):
            raise TypeError("utcnow must be callable")
        self._monotonic = monotonic
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._geocode_cache: dict[tuple[str, str], _CacheEntry] = {}
        self._weather_cache: dict[tuple[object, ...], _CacheEntry] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _positive_number(value: float, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WeatherValidationError(f"{field_name} must be a positive number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise WeatherValidationError(f"{field_name} must be a positive number")
        return numeric

    @staticmethod
    def _city(value: str) -> str:
        if not isinstance(value, str):
            raise WeatherValidationError("city must be a string")
        cleaned = value.strip()
        if not 1 <= len(cleaned) <= MAX_CITY_LENGTH:
            raise WeatherValidationError(f"city length must be 1..{MAX_CITY_LENGTH}")
        if _CONTROL_CHARACTER_PATTERN.search(cleaned):
            raise WeatherValidationError("city cannot contain control characters")
        return cleaned

    @staticmethod
    def _language(value: str) -> str:
        if not isinstance(value, str):
            raise WeatherValidationError("language must be a string")
        cleaned = value.strip().lower()
        if not _LANGUAGE_PATTERN.fullmatch(cleaned):
            raise WeatherValidationError("language must be a simple language code")
        return cleaned

    @staticmethod
    def _coordinate(value: float, field_name: str, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WeatherValidationError(f"{field_name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            raise WeatherValidationError(
                f"{field_name} must be between {minimum:g} and {maximum:g}"
            )
        return numeric

    @staticmethod
    def _days(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 7:
            raise WeatherValidationError("days must be an integer between 1 and 7")
        return value

    @staticmethod
    def _timezone_name(value: str | None, *, allow_auto: bool = True) -> str:
        if value is None and allow_auto:
            return "auto"
        if not isinstance(value, str):
            raise WeatherValidationError("timezone_name must be a string")
        cleaned = value.strip()
        if allow_auto and cleaned == "auto":
            return cleaned
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise WeatherValidationError("timezone_name is not a valid IANA timezone") from exc
        return cleaned

    @staticmethod
    def _normalized_city_key(value: str) -> str:
        return unicodedata.normalize("NFKC", value).casefold()

    def _cache_get(
        self,
        cache: dict[tuple[object, ...], _CacheEntry],
        key: tuple[object, ...],
    ) -> dict[str, Any] | None:
        now = float(self._monotonic())
        with self._cache_lock:
            entry = cache.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                cache.pop(key, None)
                return None
            return copy.deepcopy(entry.value)

    def _cache_put(
        self,
        cache: dict[tuple[object, ...], _CacheEntry],
        key: tuple[object, ...],
        value: dict[str, Any],
        ttl_seconds: float,
    ) -> None:
        expires_at = float(self._monotonic()) + ttl_seconds
        with self._cache_lock:
            cache[key] = _CacheEntry(copy.deepcopy(value), expires_at)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._geocode_cache.clear()
            self._weather_cache.clear()

    def _request_json(self, url: str, params: dict[str, object]) -> dict[str, Any]:
        try:
            response = self._http.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if not callable(raise_for_status):
                raise TypeError("HTTP response has no raise_for_status method")
            raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise WeatherUnavailableError("weather provider request failed") from exc
        if not isinstance(payload, dict):
            raise WeatherValidationError("weather provider returned a non-object JSON payload")
        return payload

    def geocode_city(self, city: str, *, language: str = "zh") -> dict[str, Any]:
        """Resolve an explicitly supplied city, using a long-lived memory cache."""

        clean_city = self._city(city)
        clean_language = self._language(language)
        cache_key = (self._normalized_city_key(clean_city), clean_language)
        cached = self._cache_get(self._geocode_cache, cache_key)
        if cached is not None:
            return cached

        payload = self._request_json(
            GEOCODING_URL,
            {
                "name": clean_city,
                "count": 1,
                "language": clean_language,
                "format": "json",
            },
        )
        results = payload.get("results")
        if results is None:
            raise WeatherValidationError("geocoding response is missing results")
        if not isinstance(results, list):
            raise WeatherValidationError("geocoding results must be a list")
        if not results:
            raise LocationNotFoundError(f"location not found: {clean_city}")
        first = results[0]
        if not isinstance(first, dict):
            raise WeatherValidationError("geocoding result must be an object")

        name = self._required_text(first.get("name"), "geocoding.name", MAX_CITY_LENGTH)
        latitude = self._coordinate(first.get("latitude"), "geocoding.latitude", -90.0, 90.0)
        longitude = self._coordinate(
            first.get("longitude"), "geocoding.longitude", -180.0, 180.0
        )
        timezone_value = first.get("timezone")
        timezone_name = (
            None
            if timezone_value is None
            else self._timezone_name(str(timezone_value), allow_auto=False)
        )
        result = {
            "query": clean_city,
            "name": name,
            "admin1": self._optional_text(first.get("admin1"), "geocoding.admin1", 120),
            "country": self._optional_text(first.get("country"), "geocoding.country", 120),
            "country_code": self._optional_text(
                first.get("country_code"), "geocoding.country_code", 8
            ),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
        }
        self._cache_put(
            self._geocode_cache,
            cache_key,
            result,
            self.geocode_ttl_seconds,
        )
        return copy.deepcopy(result)

    @staticmethod
    def _required_text(value: object, field_name: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise WeatherValidationError(f"{field_name} must be a string")
        cleaned = value.strip()
        if not 1 <= len(cleaned) <= maximum or _CONTROL_CHARACTER_PATTERN.search(cleaned):
            raise WeatherValidationError(f"{field_name} is invalid")
        return cleaned

    @classmethod
    def _optional_text(cls, value: object, field_name: str, maximum: int) -> str | None:
        if value is None:
            return None
        return cls._required_text(value, field_name, maximum)

    @staticmethod
    def _finite_number(value: object, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WeatherValidationError(f"{field_name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise WeatherValidationError(f"{field_name} must be a finite number")
        return numeric

    @staticmethod
    def _weather_code(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WeatherValidationError(f"{field_name} must be an integer")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise WeatherValidationError(f"{field_name} must be an integer")
        code = int(numeric)
        if not 0 <= code <= 999:
            raise WeatherValidationError(f"{field_name} is out of range")
        return code

    @staticmethod
    def _iso_date(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise WeatherValidationError(f"{field_name} must be an ISO date")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise WeatherValidationError(f"{field_name} must be an ISO date") from exc
        return parsed.isoformat()

    @staticmethod
    def _iso_datetime(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise WeatherValidationError(f"{field_name} must be an ISO datetime")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WeatherValidationError(f"{field_name} must be an ISO datetime") from exc
        return value

    @staticmethod
    def _array(value: object, field_name: str, expected_length: int) -> list[object]:
        if not isinstance(value, list) or len(value) != expected_length:
            raise WeatherValidationError(
                f"{field_name} must be a list of length {expected_length}"
            )
        return value

    def _validated_forecast(
        self,
        payload: dict[str, Any],
        *,
        location: dict[str, Any],
        days: int,
    ) -> dict[str, Any]:
        provider_timezone = self._timezone_name(
            self._required_text(payload.get("timezone"), "forecast.timezone", 64),
            allow_auto=False,
        )
        current = payload.get("current")
        daily = payload.get("daily")
        if not isinstance(current, dict):
            raise WeatherValidationError("forecast.current must be an object")
        if not isinstance(daily, dict):
            raise WeatherValidationError("forecast.daily must be an object")

        current_code = self._weather_code(current.get("weather_code"), "current.weather_code")
        current_result = {
            "observed_at": self._iso_datetime(current.get("time"), "current.time"),
            "temperature_c": self._finite_number(
                current.get("temperature_2m"), "current.temperature_2m"
            ),
            "apparent_temperature_c": self._finite_number(
                current.get("apparent_temperature"), "current.apparent_temperature"
            ),
            "precipitation_mm": self._finite_number(
                current.get("precipitation"), "current.precipitation"
            ),
            "wind_speed_kmh": self._finite_number(
                current.get("wind_speed_10m"), "current.wind_speed_10m"
            ),
            "weather_code": current_code,
            "weather": weather_code_zh(current_code),
        }

        dates = self._array(daily.get("time"), "daily.time", days)
        max_temperatures = self._array(
            daily.get("temperature_2m_max"), "daily.temperature_2m_max", days
        )
        min_temperatures = self._array(
            daily.get("temperature_2m_min"), "daily.temperature_2m_min", days
        )
        precipitation_probabilities = self._array(
            daily.get("precipitation_probability_max"),
            "daily.precipitation_probability_max",
            days,
        )
        weather_codes = self._array(daily.get("weather_code"), "daily.weather_code", days)
        forecast: list[dict[str, Any]] = []
        for index in range(days):
            maximum = self._finite_number(
                max_temperatures[index], f"daily.temperature_2m_max[{index}]"
            )
            minimum = self._finite_number(
                min_temperatures[index], f"daily.temperature_2m_min[{index}]"
            )
            if minimum > maximum:
                raise WeatherValidationError("daily minimum temperature exceeds maximum")
            probability = self._finite_number(
                precipitation_probabilities[index],
                f"daily.precipitation_probability_max[{index}]",
            )
            if not 0 <= probability <= 100:
                raise WeatherValidationError("daily precipitation probability is out of range")
            code = self._weather_code(weather_codes[index], f"daily.weather_code[{index}]")
            forecast.append(
                {
                    "date": self._iso_date(dates[index], f"daily.time[{index}]"),
                    "min_temperature_c": minimum,
                    "max_temperature_c": maximum,
                    "precipitation_probability_percent": probability,
                    "weather_code": code,
                    "weather": weather_code_zh(code),
                }
            )

        fetched_at = self._utcnow()
        if not isinstance(fetched_at, datetime) or fetched_at.tzinfo is None:
            raise WeatherValidationError("utcnow must return a timezone-aware datetime")
        safe_location = copy.deepcopy(location)
        safe_location["timezone"] = provider_timezone
        return {
            "location": safe_location,
            "current": current_result,
            "forecast": forecast,
            "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
            "source": {
                "name": "Open-Meteo",
                "attribution_url": ATTRIBUTION_URL,
                "forecast_api_url": FORECAST_URL,
            },
        }

    def get_weather(
        self,
        *,
        city: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        timezone_name: str | None = None,
        days: int = 3,
        language: str = "zh",
    ) -> dict[str, Any]:
        """Return current weather and a 1..7 day forecast as JSON-safe data.

        The caller must provide either a city or both coordinates. No IP lookup,
        timezone-to-city inference, or other silent location guess is performed.
        """

        clean_days = self._days(days)
        has_city = city is not None
        has_any_coordinate = latitude is not None or longitude is not None
        if has_city and has_any_coordinate:
            raise WeatherValidationError("provide either city or coordinates, not both")
        if not has_city and not has_any_coordinate:
            raise WeatherValidationError("an explicit city or coordinate pair is required")

        if has_city:
            clean_city = self._city(city)
            geocoded = self.geocode_city(clean_city, language=language)
            latitude_value = float(geocoded["latitude"])
            longitude_value = float(geocoded["longitude"])
            lookup_timezone = self._timezone_name(
                timezone_name or geocoded.get("timezone"),
                allow_auto=True,
            )
            location = {
                "kind": "city",
                "query": geocoded["query"],
                "name": geocoded["name"],
                "admin1": geocoded["admin1"],
                "country": geocoded["country"],
                "country_code": geocoded["country_code"],
                "latitude": latitude_value,
                "longitude": longitude_value,
                "timezone": lookup_timezone,
            }
            location_key: tuple[object, ...] = (
                "city",
                self._normalized_city_key(clean_city),
            )
        else:
            if latitude is None or longitude is None:
                raise WeatherValidationError("latitude and longitude must be provided together")
            latitude_value = self._coordinate(latitude, "latitude", -90.0, 90.0)
            longitude_value = self._coordinate(longitude, "longitude", -180.0, 180.0)
            lookup_timezone = self._timezone_name(timezone_name, allow_auto=True)
            location = {
                "kind": "authorized_coarse_coordinates",
                "name": "用户授权的大致位置",
                "latitude": latitude_value,
                "longitude": longitude_value,
                "timezone": lookup_timezone,
            }
            location_key = (
                "coordinates",
                round(latitude_value, 5),
                round(longitude_value, 5),
            )

        cache_key = (*location_key, lookup_timezone, clean_days)
        cached = self._cache_get(self._weather_cache, cache_key)
        if cached is not None:
            return cached
        payload = self._request_json(
            FORECAST_URL,
            {
                "latitude": latitude_value,
                "longitude": longitude_value,
                "timezone": lookup_timezone,
                "current": (
                    "temperature_2m,apparent_temperature,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": clean_days,
            },
        )
        result = self._validated_forecast(
            payload,
            location=location,
            days=clean_days,
        )
        self._cache_put(
            self._weather_cache,
            cache_key,
            result,
            self.weather_ttl_seconds,
        )
        return copy.deepcopy(result)
