"""Small adapters between one chat request and the real-time tool layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from grounding import GroundedFact, GroundingBundle, UIArtifact
from information_store import OfficialUpdate, query_recent_approved_updates
from runtime_context import RuntimeContext
from weather_service import WeatherService


class RealtimeToolInputError(ValueError):
    """Raised when a tool cannot run without guessing user data."""


def build_realtime_unavailable_bundle(name: str = "realtime") -> GroundingBundle:
    """Describe an orchestration failure without exposing exception details."""

    safe_name = name if name in {
        "weather",
        "recent_updates",
        "official_search",
        "web_search",
        "realtime",
    } else "realtime"
    return GroundingBundle(
        facts=(
            GroundedFact(
                text="本轮实时信息工具没有返回可用数据；不得使用模型记忆补全实时事实。",
                untrusted=False,
                fact_eligible=False,
            ),
        ),
        ui_artifacts=(
            UIArtifact(
                "search_status",
                {
                    "name": safe_name,
                    "ok": False,
                    "error_code": "tool_execution_failed",
                },
            ),
        ),
    )


def build_weather_lookup(
    service: WeatherService,
    runtime_context: RuntimeContext,
) -> Callable[..., dict[str, Any]]:
    """Bind one request's explicit location to the weather tool.

    A city supplied in the user's current message takes precedence. Otherwise
    only a city or coarse coordinates explicitly saved by the user are used.
    """

    if not isinstance(service, WeatherService):
        raise TypeError("service must be WeatherService")
    if not isinstance(runtime_context, RuntimeContext):
        raise TypeError("runtime_context must be RuntimeContext")

    def get_weather(*, location: str | None = None, days: int = 3) -> dict[str, Any]:
        if location is not None:
            return service.get_weather(city=location, days=days)

        saved = runtime_context.weather_location()
        if saved is None:
            raise RealtimeToolInputError(
                "weather requires a city in the message or an explicitly saved location"
            )
        city = saved.get("city")
        if isinstance(city, str) and city.strip():
            return service.get_weather(city=city, days=days)
        latitude = saved.get("latitude")
        longitude = saved.get("longitude")
        if isinstance(latitude, (int, float)) and not isinstance(latitude, bool) and isinstance(
            longitude, (int, float)
        ) and not isinstance(longitude, bool):
            return service.get_weather(
                latitude=float(latitude),
                longitude=float(longitude),
                timezone_name=runtime_context.timezone_name,
                days=days,
            )
        raise RealtimeToolInputError("saved weather location is incomplete")

    return get_weather


def serialize_approved_updates(
    updates: Iterable[OfficialUpdate],
) -> dict[str, object]:
    """Expose only reviewed fields required by the chat tool."""

    items: list[dict[str, object]] = []
    for update in updates:
        if not isinstance(update, OfficialUpdate):
            raise TypeError("updates must contain OfficialUpdate values")
        if update.verification_status != "approved":
            continue
        items.append(
            {
                "id": update.id,
                "title": update.title,
                "summary": update.summary,
                "category": update.category,
                "status": update.status,
                "event_start_at": update.event_start_at,
                "event_end_at": update.event_end_at,
                "venue": update.venue,
                "published_at": update.published_at,
                "canonical_url": update.canonical_url,
                "source_name": update.source_name,
                "replaces_update_id": update.replaces_update_id,
                "verification_status": "approved",
                "approved": True,
            }
        )
    return {"approved": True, "updates": items}


def build_recent_updates_lookup(
    db_path: str | Path,
) -> Callable[..., dict[str, object]]:
    """Bind the reviewed-update query to the active application database."""

    def query_recent_updates(
        *, days_ahead: int = 60, limit: int = 10
    ) -> dict[str, object]:
        updates = query_recent_approved_updates(
            db_path,
            days_back=7,
            days_ahead=days_ahead,
            limit=limit,
        )
        return serialize_approved_updates(updates)

    return query_recent_updates


def realtime_grounding_is_unavailable(
    grounding: GroundingBundle,
    route: str,
) -> bool:
    """Return whether a routed real-time request has no usable result."""

    if not isinstance(grounding, GroundingBundle):
        raise TypeError("grounding must be GroundingBundle")
    failed = any(
        artifact.artifact_type == "search_status"
        and artifact.payload.get("ok") is False
        for artifact in grounding.ui_artifacts
    )
    if route == "web_search":
        return failed or not bool(grounding.facts)
    if route in {"weather", "recent_updates", "official_search"}:
        return not any(fact.fact_eligible for fact in grounding.facts)
    return False


__all__ = [
    "RealtimeToolInputError",
    "build_realtime_unavailable_bundle",
    "build_recent_updates_lookup",
    "build_weather_lookup",
    "realtime_grounding_is_unavailable",
    "serialize_approved_updates",
]
