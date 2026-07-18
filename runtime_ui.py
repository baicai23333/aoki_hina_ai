"""Streamlit controls for explicit time, city, and opt-in coarse location data."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from browser_geolocation import BrowserGeolocationError, browser_geolocation
from runtime_context import RuntimeContext, build_runtime_context
from runtime_profile import (
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeProfileValidationError,
    clear_authorized_coordinates,
    clear_home_city,
    clear_temporary_city,
    get_runtime_profile,
    set_authorized_coordinates,
    set_browser_context,
    set_home_city,
    set_temporary_city,
)


_NOTICE_KEY_PREFIX = "runtime_profile_notice_"
_GEO_VERSION_KEY_PREFIX = "runtime_geo_version_"
_LOCATION_ERROR_MESSAGES = {
    "permission_denied": "你没有授予定位权限；仍可手动填写城市。",
    "position_unavailable": "浏览器暂时无法取得位置，请手动填写城市或稍后再试。",
    "timeout": "定位等待超时，请重试或手动填写城市。",
    "unsupported": "当前浏览器不支持定位，请手动填写城市。",
    "insecure_context": "浏览器定位需要 HTTPS；本机访问也可使用手动城市。",
    "unknown": "暂时无法取得位置，请手动填写城市或稍后再试。",
}


def _browser_context_values() -> tuple[str | None, str | None]:
    """Read browser-provided context without treating absence as an error."""

    try:
        timezone_name = st.context.timezone
    except Exception:
        timezone_name = None
    try:
        locale = st.context.locale
    except Exception:
        locale = None
    return (
        timezone_name if isinstance(timezone_name, str) and timezone_name else None,
        locale if isinstance(locale, str) and locale else None,
    )


def _notice(username: str, message: str, level: str = "success") -> None:
    st.session_state[f"{_NOTICE_KEY_PREFIX}{username}"] = (level, message)


def _show_notice(username: str) -> None:
    value = st.session_state.pop(f"{_NOTICE_KEY_PREFIX}{username}", None)
    if not isinstance(value, tuple) or len(value) != 2:
        return
    level, message = value
    if level == "warning":
        st.warning(message)
    else:
        st.success(message)


def _sync_browser_context(
    db_path: str | Path,
    username: str,
    profile: RuntimeProfile | None,
    timezone_name: str | None,
    locale: str | None,
) -> RuntimeProfile | None:
    if timezone_name is None and locale is None:
        return profile
    desired_timezone = (
        timezone_name
        if timezone_name is not None
        else (profile.browser_timezone if profile is not None else None)
    )
    desired_locale = (
        locale
        if locale is not None
        else (profile.browser_locale if profile is not None else None)
    )
    if (
        profile is not None
        and profile.browser_timezone == desired_timezone
        and profile.browser_locale == desired_locale
    ):
        return profile
    return set_browser_context(
        db_path, username, desired_timezone, desired_locale
    )


def _bump_geolocation_component(username: str) -> None:
    key = f"{_GEO_VERSION_KEY_PREFIX}{username}"
    value = st.session_state.get(key, 0)
    st.session_state[key] = value + 1 if isinstance(value, int) else 1


def _location_caption(profile: RuntimeProfile | None, now_utc: datetime) -> str:
    if profile is None:
        return "尚未设置天气位置"
    if (
        profile.temporary_city
        and profile.temporary_city_expires_at
        and profile.temporary_city_expires_at > now_utc
    ):
        return f"当前城市：{profile.temporary_city}（临时）"
    if (
        profile.coarse_latitude is not None
        and profile.coarse_longitude is not None
        and profile.coarse_coordinates_expires_at
        and profile.coarse_coordinates_expires_at > now_utc
    ):
        return "当前使用：浏览器授权的大致位置"
    if profile.home_city:
        return f"常住城市：{profile.home_city}"
    return "尚未设置天气位置"


def render_runtime_sidebar(
    db_path: str | Path,
    username: str,
    *,
    fallback_timezone: str = "Asia/Shanghai",
) -> RuntimeContext:
    """Render profile controls and return this request's ephemeral context."""

    browser_timezone, browser_locale = _browser_context_values()
    profile: RuntimeProfile | None = None
    profile_available = True
    try:
        profile = get_runtime_profile(db_path, username)
        profile = _sync_browser_context(
            db_path,
            username,
            profile,
            browser_timezone,
            browser_locale,
        )
    except (RuntimeProfileError, RuntimeProfileValidationError, sqlite3.Error):
        profile_available = False

    try:
        context = build_runtime_context(
            profile,
            browser_timezone=browser_timezone,
            browser_locale=browser_locale,
            fallback_timezone=fallback_timezone,
        )
    except Exception:
        # A malformed browser value must never stop ordinary chat.
        context = build_runtime_context(
            None,
            browser_timezone=None,
            browser_locale=None,
            fallback_timezone="UTC",
        )

    with st.sidebar.expander("时间与天气位置", expanded=False):
        _show_notice(username)
        st.caption(
            f"浏览器时区：{context.timezone_name} · "
            f"当地时间：{context.local_datetime:%Y-%m-%d %H:%M}"
        )
        st.caption(_location_caption(profile, context.utc_datetime))
        st.caption("位置只在天气或出行相关对话中使用，不会持续追踪。")
        st.caption("浏览器时区和语言会随账号保存；它们不会被用来推断城市。")

        if not profile_available:
            st.warning("位置设置暂时不可用；时间上下文和普通聊天仍可继续。")
            return context

        with st.form(f"runtime_home_city_{username}", clear_on_submit=False):
            home_city = st.text_input(
                "常住城市",
                value=(profile.home_city if profile else "") or "",
                max_chars=120,
                placeholder="例如：广州",
            )
            save_home = st.form_submit_button("保存常住城市", use_container_width=True)
        if save_home:
            try:
                set_home_city(db_path, username, home_city)
                _notice(username, "常住城市已保存。")
                st.rerun()
            except (RuntimeProfileError, RuntimeProfileValidationError, sqlite3.Error):
                st.error("城市没有保存成功，请检查名称后重试。")

        if profile and profile.home_city:
            if st.button(
                "清除常住城市",
                key=f"runtime_clear_home_{username}",
                use_container_width=True,
            ):
                try:
                    clear_home_city(db_path, username)
                    _notice(username, "常住城市已清除。")
                    st.rerun()
                except (RuntimeProfileError, sqlite3.Error):
                    st.error("暂时无法清除，请稍后再试。")

        st.divider()
        with st.form(f"runtime_temp_city_{username}", clear_on_submit=True):
            current_city = st.text_input(
                "临时所在城市",
                max_chars=120,
                placeholder="旅行时可临时设置",
            )
            current_days = st.selectbox(
                "保留时间",
                options=(1, 3, 7, 30),
                index=2,
                format_func=lambda value: f"{value} 天",
            )
            save_current = st.form_submit_button(
                "使用这个临时城市", use_container_width=True
            )
        if save_current:
            try:
                set_temporary_city(
                    db_path,
                    username,
                    current_city,
                    datetime.now(timezone.utc) + timedelta(days=current_days),
                )
                _bump_geolocation_component(username)
                _notice(username, "临时城市已保存，到期后会自动恢复常住城市。")
                st.rerun()
            except (RuntimeProfileError, RuntimeProfileValidationError, sqlite3.Error):
                st.error("临时城市没有保存成功，请检查名称后重试。")

        has_temporary = bool(
            profile
            and (
                profile.temporary_city
                or (
                    profile.coarse_latitude is not None
                    and profile.coarse_longitude is not None
                )
            )
        )
        if has_temporary and st.button(
            "恢复使用常住城市",
            key=f"runtime_clear_current_{username}",
            use_container_width=True,
        ):
            try:
                clear_temporary_city(db_path, username)
                clear_authorized_coordinates(db_path, username)
                _bump_geolocation_component(username)
                _notice(username, "临时位置已清除。")
                st.rerun()
            except (RuntimeProfileError, sqlite3.Error):
                st.error("暂时无法清除，请稍后再试。")

        st.divider()
        st.caption("也可以让浏览器提供一次大致位置（需要 HTTPS 或本机访问）。")
        try:
            geo_version = st.session_state.get(
                f"{_GEO_VERSION_KEY_PREFIX}{username}", 0
            )
            geolocation = browser_geolocation(
                key=f"browser_geo_{username}_{geo_version}"
            )
        except BrowserGeolocationError:
            geolocation = {"status": "error", "code": "unknown"}
        if geolocation and geolocation.get("status") == "error":
            st.info(
                _LOCATION_ERROR_MESSAGES.get(
                    str(geolocation.get("code")), _LOCATION_ERROR_MESSAGES["unknown"]
                )
            )
        elif geolocation and geolocation.get("status") == "success":
            st.success("浏览器已返回大致位置；确认后仅保存约 1 公里精度，24 小时后删除。")
            if st.button(
                "确认用于天气",
                key=f"runtime_save_geo_{username}",
                use_container_width=True,
            ):
                try:
                    set_authorized_coordinates(
                        db_path,
                        username,
                        float(geolocation["latitude"]),
                        float(geolocation["longitude"]),
                        datetime.now(timezone.utc) + timedelta(hours=24),
                    )
                    _bump_geolocation_component(username)
                    _notice(username, "大致位置已保存 24 小时。")
                    st.rerun()
                except (
                    RuntimeProfileError,
                    RuntimeProfileValidationError,
                    sqlite3.Error,
                    ValueError,
                ):
                    st.error("位置没有保存成功，请改用手动城市。")

    return context


__all__ = ["render_runtime_sidebar"]
