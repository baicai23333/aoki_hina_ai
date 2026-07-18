"""Privacy-safe presentation helpers for persisted chat UI artifacts.

The tool and storage layers deliberately keep richer JSON payloads than the UI
needs.  This module applies a strict display allowlist before anything reaches
Streamlit: coordinates, snippets, raw page content, and internal diagnostics are
never copied into the presentation objects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, TypeAlias
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


OPEN_METEO_NAME = "Open-Meteo"
OPEN_METEO_URL = "https://open-meteo.com/"
MAX_SOURCE_CARDS = 10
MAX_FORECAST_DAYS = 7

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class ForecastDayView:
    date_label: str
    weather: str | None = None
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    precipitation_probability_percent: float | None = None


@dataclass(frozen=True)
class WeatherCardView:
    location: str | None
    weather: str | None
    temperature_c: float | None
    apparent_temperature_c: float | None
    precipitation_mm: float | None
    wind_speed_kmh: float | None
    updated_label: str | None
    forecast: tuple[ForecastDayView, ...]
    attribution_name: str = OPEN_METEO_NAME
    attribution_url: str = OPEN_METEO_URL


@dataclass(frozen=True)
class SourceCardView:
    title: str
    url: str
    published_label: str | None
    confirmation_label: str


@dataclass(frozen=True)
class SourceCardsView:
    items: tuple[SourceCardView, ...]


@dataclass(frozen=True)
class SearchStatusView:
    message: str
    level: str = "info"


PresentableArtifact: TypeAlias = WeatherCardView | SourceCardsView | SearchStatusView


def _clean_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _CONTROL_CHARACTERS.sub(" ", value).strip()
    if not cleaned:
        return None
    return cleaned[:maximum]


def safe_http_url(value: object) -> str | None:
    """Return a bounded HTTP(S) URL, rejecting credentials and odd schemes."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 2_048 or _CONTROL_CHARACTERS.search(cleaned):
        return None
    if "\\" in cleaned:
        return None
    try:
        parsed = urlsplit(cleaned)
        # Accessing ``port`` also validates malformed port syntax.
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"}:
        return None
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    if any(character.isspace() for character in parsed.netloc):
        return None
    return cleaned


def _finite_number(
    value: object, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if minimum is not None and result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


def _first_number(
    payload: Mapping[str, Any],
    *keys: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    for key in keys:
        result = _finite_number(payload.get(key), minimum=minimum, maximum=maximum)
        if result is not None:
            return result
    return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 100:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _timezone(value: object) -> ZoneInfo | None:
    name = _clean_text(value, maximum=100)
    if name is None:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _datetime_label(value: object, timezone_name: object = None) -> str | None:
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    target_timezone = _timezone(timezone_name)
    if target_timezone is not None and parsed.tzinfo is not None:
        parsed = parsed.astimezone(target_timezone)
    return parsed.strftime("%Y年%m月%d日 %H:%M")


def _date_label(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 32:
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        parsed_datetime = _parse_datetime(value)
        if parsed_datetime is None:
            return None
        parsed = parsed_datetime.date()
    return parsed.strftime("%m月%d日")


def _normalize_forecast(value: object) -> tuple[ForecastDayView, ...]:
    if not isinstance(value, list):
        return ()
    result: list[ForecastDayView] = []
    for item in value[:MAX_FORECAST_DAYS]:
        if not isinstance(item, Mapping):
            continue
        date_label = _date_label(item.get("date"))
        if date_label is None:
            continue
        minimum = _first_number(
            item,
            "min_temperature_c",
            "min_temperature",
            "temperature_2m_min",
            minimum=-100,
            maximum=100,
        )
        maximum = _first_number(
            item,
            "max_temperature_c",
            "max_temperature",
            "temperature_2m_max",
            minimum=-100,
            maximum=100,
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum = maximum = None
        probability = _first_number(
            item,
            "precipitation_probability_percent",
            "precipitation_probability",
            "rain_probability",
            minimum=0,
            maximum=100,
        )
        result.append(
            ForecastDayView(
                date_label=date_label,
                weather=_clean_text(item.get("weather"), maximum=40),
                min_temperature_c=minimum,
                max_temperature_c=maximum,
                precipitation_probability_percent=probability,
            )
        )
    return tuple(result)


def normalize_weather_card(payload: object) -> WeatherCardView | None:
    if not isinstance(payload, Mapping):
        return None
    raw_location = payload.get("location")
    location_payload = raw_location if isinstance(raw_location, Mapping) else {}
    location = _clean_text(
        location_payload.get("name", location_payload.get("city")), maximum=80
    )
    timezone_name = location_payload.get("timezone")

    raw_current = payload.get("current")
    current = raw_current if isinstance(raw_current, Mapping) else payload
    weather = _clean_text(current.get("weather"), maximum=40)
    temperature = _first_number(
        current,
        "temperature_c",
        "temperature",
        "temperature_2m",
        minimum=-100,
        maximum=100,
    )
    apparent_temperature = _first_number(
        current,
        "apparent_temperature_c",
        "apparent_temperature",
        minimum=-120,
        maximum=120,
    )
    precipitation = _first_number(
        current,
        "precipitation_mm",
        "precipitation",
        minimum=0,
        maximum=10_000,
    )
    wind_speed = _first_number(
        current,
        "wind_speed_kmh",
        "wind_speed",
        "wind_speed_10m",
        minimum=0,
        maximum=1_000,
    )
    forecast = _normalize_forecast(payload.get("forecast"))
    updated_label = _datetime_label(
        payload.get("fetched_at", payload.get("observed_at")), timezone_name
    )
    if updated_label is None:
        updated_label = _datetime_label(current.get("observed_at"), timezone_name)

    if not any(
        (
            location,
            weather,
            temperature is not None,
            apparent_temperature is not None,
            precipitation is not None,
            wind_speed is not None,
            updated_label,
            forecast,
        )
    ):
        return None
    return WeatherCardView(
        location=location,
        weather=weather,
        temperature_c=temperature,
        apparent_temperature_c=apparent_temperature,
        precipitation_mm=precipitation,
        wind_speed_kmh=wind_speed,
        updated_label=updated_label,
        forecast=forecast,
    )


def _confirmation_label(item: Mapping[str, Any]) -> str:
    verification = item.get("verification_status", item.get("review_status"))
    if item.get("approved") is True or verification == "approved":
        return "官方确认"
    if item.get("approved") is False or verification in {"pending", "rejected"}:
        return "尚未官方确认"
    trust_level = _finite_number(item.get("trust_level"), minimum=0, maximum=100)
    if _clean_text(item.get("official_source"), maximum=80) is not None or (
        trust_level is not None and trust_level >= 90
    ):
        return "官方来源"
    return "来源待核实"


def _published_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    try:
        parsed_date = date.fromisoformat(stripped)
    except ValueError:
        parsed_datetime = _parse_datetime(stripped)
        if parsed_datetime is None:
            return None
        return parsed_datetime.strftime("%Y年%m月%d日 %H:%M")
    return parsed_date.strftime("%Y年%m月%d日")


def normalize_source_cards(payload: object) -> SourceCardsView | None:
    if not isinstance(payload, Mapping):
        return None
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return None
    items: list[SourceCardView] = []
    for item in raw_items[:MAX_SOURCE_CARDS]:
        if not isinstance(item, Mapping):
            continue
        title = _clean_text(item.get("title"), maximum=300)
        url = safe_http_url(item.get("url"))
        if title is None or url is None:
            continue
        items.append(
            SourceCardView(
                title=title,
                url=url,
                published_label=_published_label(item.get("published_at")),
                confirmation_label=_confirmation_label(item),
            )
        )
    return SourceCardsView(tuple(items)) if items else None


_TOOL_LABELS = {
    "get_weather": "天气",
    "search_web": "网页信息",
    "search_hina_official": "官方信息",
    "query_recent_updates": "近期官方信息",
}


def _bounded_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 10_000 else None


def normalize_search_status(payload: object) -> SearchStatusView | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("ok") is False:
        label = _TOOL_LABELS.get(payload.get("name"), "即时信息")
        error_code = payload.get("error_code")
        if error_code == "tool_dependency_unavailable":
            message = f"{label}服务暂未配置或暂不可用。"
        elif error_code == "tool_round_limit":
            message = "即时信息查询步骤已达到上限，请稍后重试。"
        else:
            message = f"{label}查询暂时失败，请稍后再试。"
        return SearchStatusView(message=message, level="warning")

    count = _bounded_count(payload.get("count"))
    approved_count = _bounded_count(payload.get("approved_count"))
    if payload.get("kind") == "recent_updates":
        if count == 0:
            return SearchStatusView("暂未在已审核的近期信息中找到结果。")
        if count is not None:
            if approved_count is not None:
                return SearchStatusView(
                    f"近期信息共找到 {count} 条，其中 {approved_count} 条已审核。"
                )
            return SearchStatusView(f"近期信息共找到 {count} 条。")
        return None
    if count is not None:
        return SearchStatusView(f"即时信息查询完成，共找到 {count} 条结果。")
    if payload.get("ok") is True:
        return SearchStatusView("即时信息查询已完成。")
    return None


def normalize_artifact(artifact_type: object, payload: object) -> PresentableArtifact | None:
    """Normalize one persisted artifact without propagating malformed input."""

    try:
        if artifact_type == "weather_card":
            return normalize_weather_card(payload)
        if artifact_type == "source_cards":
            return normalize_source_cards(payload)
        if artifact_type == "search_status":
            return normalize_search_status(payload)
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def normalize_message_artifact(artifact: object) -> PresentableArtifact | None:
    """Accept ``MessageArtifact``, ``UIArtifact``, or an equivalent mapping."""

    if isinstance(artifact, Mapping):
        artifact_type = artifact.get("artifact_type")
        payload = artifact.get("payload")
    else:
        artifact_type = getattr(artifact, "artifact_type", None)
        payload = getattr(artifact, "payload", None)
    return normalize_artifact(artifact_type, payload)


def normalize_message_artifacts(
    artifacts: Iterable[object],
) -> tuple[PresentableArtifact, ...]:
    try:
        candidates = iter(artifacts)
    except TypeError:
        return ()
    normalized: list[PresentableArtifact] = []
    for artifact in candidates:
        value = normalize_message_artifact(artifact)
        if value is not None:
            normalized.append(value)
    return tuple(normalized)


def _display_number(value: float) -> str:
    rounded = round(value, 1)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"


def _forecast_line(day: ForecastDayView) -> str:
    details: list[str] = []
    if day.weather:
        details.append(day.weather)
    if day.min_temperature_c is not None and day.max_temperature_c is not None:
        details.append(
            f"{_display_number(day.min_temperature_c)}–"
            f"{_display_number(day.max_temperature_c)}°C"
        )
    if day.precipitation_probability_percent is not None:
        details.append(
            f"降雨概率 {_display_number(day.precipitation_probability_percent)}%"
        )
    suffix = "，".join(details) if details else "暂无详细预报"
    return f"{day.date_label}：{suffix}"


def render_presentable_artifact(
    artifact: PresentableArtifact, *, st_module: Any | None = None
) -> None:
    """Render an already normalized artifact with Streamlit-safe primitives."""

    if st_module is None:
        import streamlit as st_module  # Local import keeps pure tests lightweight.

    if isinstance(artifact, WeatherCardView):
        title = f"{artifact.location} · 天气" if artifact.location else "天气"
        with st_module.container(border=True):
            # Provider/user-derived labels use plain text so they cannot create
            # extra Markdown links.  Only the allowlisted link buttons below are
            # interactive.
            st_module.text(title)
            current: list[str] = []
            if artifact.weather:
                current.append(artifact.weather)
            if artifact.temperature_c is not None:
                current.append(f"{_display_number(artifact.temperature_c)}°C")
            if artifact.apparent_temperature_c is not None:
                current.append(
                    f"体感 {_display_number(artifact.apparent_temperature_c)}°C"
                )
            if artifact.precipitation_mm is not None:
                current.append(f"降水 {_display_number(artifact.precipitation_mm)} mm")
            if artifact.wind_speed_kmh is not None:
                current.append(f"风速 {_display_number(artifact.wind_speed_kmh)} km/h")
            if current:
                st_module.text(" · ".join(current))
            for day in artifact.forecast:
                st_module.text(_forecast_line(day))
            if artifact.updated_label:
                st_module.caption(f"更新时间：{artifact.updated_label}")
            st_module.caption(f"天气数据来自 {artifact.attribution_name}")
            st_module.link_button(
                f"查看 {artifact.attribution_name}", artifact.attribution_url
            )
        return

    if isinstance(artifact, SourceCardsView):
        with st_module.container(border=True):
            st_module.markdown("**参考来源**")
            for item in artifact.items:
                st_module.link_button(item.title, item.url, use_container_width=True)
                metadata = [item.confirmation_label]
                if item.published_label:
                    metadata.append(f"发布于 {item.published_label}")
                st_module.caption(" · ".join(metadata))
        return

    if isinstance(artifact, SearchStatusView):
        if artifact.level == "warning":
            st_module.warning(artifact.message)
        else:
            st_module.caption(artifact.message)


def render_message_artifacts(
    artifacts: Iterable[object], *, st_module: Any | None = None
) -> None:
    """Normalize and render a collection; bad artifacts are silently skipped."""

    for artifact in normalize_message_artifacts(artifacts):
        render_presentable_artifact(artifact, st_module=st_module)


__all__ = [
    "ForecastDayView",
    "OPEN_METEO_NAME",
    "OPEN_METEO_URL",
    "PresentableArtifact",
    "SearchStatusView",
    "SourceCardView",
    "SourceCardsView",
    "WeatherCardView",
    "normalize_artifact",
    "normalize_message_artifact",
    "normalize_message_artifacts",
    "normalize_search_status",
    "normalize_source_cards",
    "normalize_weather_card",
    "render_message_artifacts",
    "render_presentable_artifact",
    "safe_http_url",
]
