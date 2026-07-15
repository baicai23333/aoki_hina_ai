"""Protected Streamlit management console for the local chat application."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path

import streamlit as st
from argon2 import PasswordHasher

from admin_auth import verify_admin_credentials
from admin_login_guard import (
    clear_login_failures as _clear_login_failures,
    login_wait_seconds as _login_wait_seconds,
    record_login_failure as _record_login_failure,
)
from admin_service import (
    AdminServiceError,
    clear_user_history,
    clear_user_memories,
    delete_user_account,
    get_database_health,
    get_overview,
    get_translation_breakdown,
    list_audit_entries,
    list_recent_messages,
    list_user_summaries,
    record_admin_action,
    replace_user_password_hash,
)
from tts_engine import load_env_var


ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "chat_history.db"

_AUTHENTICATED_KEY = "aoki_admin_authenticated"
_ACTOR_KEY = "aoki_admin_actor"
_LAST_ACTIVE_KEY = "aoki_admin_last_active"
_CONFIG_FINGERPRINT_KEY = "aoki_admin_config_fingerprint"
_LOGIN_SOURCE_TOKEN_KEY = "aoki_admin_login_source_token"
_NOTICE_KEY = "aoki_admin_notice"
_MAX_ADMIN_USERNAME_LENGTH = 256
_MAX_ADMIN_HASH_LENGTH = 1024
_PASSWORD_HASHER = PasswordHasher()


def _env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    value = (load_env_var(name, fallback) or fallback).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _session_ttl_seconds() -> int:
    raw_value = load_env_var("AOKI_ADMIN_SESSION_TTL_MINUTES", "30") or "30"
    try:
        minutes = int(raw_value)
    except ValueError:
        minutes = 30
    return max(5, min(minutes, 480)) * 60


def _configured_credentials() -> tuple[str | None, str | None]:
    username = load_env_var("AOKI_ADMIN_USERNAME")
    password_hash = load_env_var("AOKI_ADMIN_PASSWORD_HASH")
    return username, password_hash


def _configuration_issue(
    username: str | None, password_hash: str | None
) -> str | None:
    if not username or not password_hash:
        return "管理后台尚未启用。请先设置管理员账号和密码哈希。"
    if username != username.strip() or len(username) > _MAX_ADMIN_USERNAME_LENGTH:
        return "管理员账号配置无效，请使用 1 到 256 个非空白边界字符。"
    if len(password_hash) > _MAX_ADMIN_HASH_LENGTH or not password_hash.startswith(
        ("$argon2id$", "$argon2i$", "$argon2d$")
    ):
        return "管理员密码哈希配置无效，请重新生成。"
    return None


def _credential_fingerprint(username: str, password_hash: str) -> str:
    material = f"{username}\0{password_hash}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _login_guard_key(credential_fingerprint: str) -> str:
    # This is deliberately a browser-session cooldown, not a network security
    # boundary. Streamlit's reported client IP can be spoofed and must not be
    # trusted for perimeter rate limiting.
    source = st.session_state.get(_LOGIN_SOURCE_TOKEN_KEY)
    if not isinstance(source, str):
        source = secrets.token_urlsafe(24)
        st.session_state[_LOGIN_SOURCE_TOKEN_KEY] = source
    return hashlib.sha256(
        f"{credential_fingerprint}\0{source}".encode("utf-8")
    ).hexdigest()


def _session_matches_credentials(
    actor: object,
    session_fingerprint: object,
    configured_username: str | None,
    configured_password_hash: str | None,
) -> bool:
    if _configuration_issue(configured_username, configured_password_hash):
        return False
    if not isinstance(actor, str) or not isinstance(session_fingerprint, str):
        return False
    expected_fingerprint = _credential_fingerprint(
        configured_username or "", configured_password_hash or ""
    )
    return hmac.compare_digest(
        actor.encode("utf-8"), (configured_username or "").encode("utf-8")
    ) and hmac.compare_digest(session_fingerprint, expected_fingerprint)


def _clear_admin_session() -> None:
    for key in (
        _AUTHENTICATED_KEY,
        _ACTOR_KEY,
        _LAST_ACTIVE_KEY,
        _CONFIG_FINGERPRINT_KEY,
    ):
        st.session_state.pop(key, None)


def _current_admin(*, touch: bool = True) -> str | None:
    configured_username, configured_password_hash = _configured_credentials()
    if not st.session_state.get(_AUTHENTICATED_KEY):
        return None

    actor = st.session_state.get(_ACTOR_KEY)
    session_fingerprint = st.session_state.get(_CONFIG_FINGERPRINT_KEY)
    if not _session_matches_credentials(
        actor,
        session_fingerprint,
        configured_username,
        configured_password_hash,
    ):
        _clear_admin_session()
        return None

    now = time.time()
    last_active = st.session_state.get(_LAST_ACTIVE_KEY)
    if not isinstance(last_active, (int, float)) or (
        now - float(last_active) > _session_ttl_seconds()
    ):
        _clear_admin_session()
        return None

    if touch:
        st.session_state[_LAST_ACTIVE_KEY] = now
    return actor


@st.fragment(run_every="30s")
def _session_expiry_watch() -> None:
    """Remove sensitive UI shortly after an idle session expires."""

    if _current_admin(touch=False) is None:
        st.rerun()


def _set_notice(level: str, message: str) -> None:
    st.session_state[_NOTICE_KEY] = (level, message)


def _show_notice() -> None:
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if not notice:
        return
    level, message = notice
    renderer = getattr(st, level, st.info)
    renderer(message)


def _login() -> None:
    st.title("管理后台")
    st.caption("站点数据与用户维护，仅限管理员访问。")

    configured_username, configured_password_hash = _configured_credentials()
    configuration_issue = _configuration_issue(
        configured_username, configured_password_hash
    )
    if configuration_issue:
        st.warning(configuration_issue)
        st.code(
            ".\\.venv\\Scripts\\python.exe scripts\\create_admin_password_hash.py",
            language="powershell",
        )
        st.stop()

    fingerprint = _credential_fingerprint(
        configured_username, configured_password_hash
    )
    guard_key = _login_guard_key(fingerprint)
    wait_seconds = _login_wait_seconds(guard_key)
    if wait_seconds:
        st.error(f"尝试次数过多，请在 {wait_seconds} 秒后再试。")
        if st.button("重新检查登录状态"):
            st.rerun()
        st.stop()

    with st.form("admin_login_form", clear_on_submit=True):
        username = st.text_input(
            "管理员账号", max_chars=_MAX_ADMIN_USERNAME_LENGTH
        )
        password = st.text_input("管理员密码", type="password", max_chars=1024)
        submitted = st.form_submit_button("进入后台", type="primary")

    if not submitted:
        st.stop()

    if verify_admin_credentials(
        username,
        password,
        configured_username,
        configured_password_hash,
    ):
        try:
            record_admin_action(DB_FILE, configured_username, "login")
        except (AdminServiceError, sqlite3.Error, OSError):
            st.error("后台暂时无法建立安全会话，请稍后再试。")
            st.stop()
        st.session_state[_AUTHENTICATED_KEY] = True
        st.session_state[_ACTOR_KEY] = configured_username
        st.session_state[_LAST_ACTIVE_KEY] = time.time()
        st.session_state[_CONFIG_FINGERPRINT_KEY] = fingerprint
        _clear_login_failures(guard_key)
        st.rerun()

    if _record_login_failure(guard_key):
        st.error("尝试次数过多，后台已暂时锁定。")
    else:
        st.error("管理员账号或密码不正确。")
    st.stop()


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _safe_widget_suffix(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:12]


def _overview_tab() -> None:
    overview = get_overview(DB_FILE)
    translation = get_translation_breakdown(DB_FILE)

    columns = st.columns(4)
    columns[0].metric("用户", overview.total_users)
    columns[1].metric("消息", overview.total_messages)
    columns[2].metric("有过聊天的用户", overview.users_with_messages)
    columns[3].metric("用户记忆", overview.total_memories)

    st.subheader("消息与翻译")
    message_columns = st.columns(3)
    message_columns[0].metric("用户消息", overview.human_messages)
    message_columns[1].metric("AI 回复", overview.ai_messages)
    message_columns[2].metric(
        "翻译可播放",
        translation.validated + translation.fixed,
    )

    st.dataframe(
        [
            {"状态": "已验证", "数量": translation.validated},
            {"状态": "已修复并验证", "数量": translation.fixed},
            {"状态": "审核拒绝", "数量": translation.rejected},
            {"状态": "处理失败", "数量": translation.failed},
            {"状态": "无翻译", "数量": translation.none},
            {"状态": "旧版未验证", "数量": translation.legacy_unverified},
        ],
        hide_index=True,
        use_container_width=True,
    )
    if overview.latest_message_at:
        st.caption(f"最近一条消息：{overview.latest_message_at}")


def _run_user_action(action, success_message: str, *args) -> None:
    try:
        action(*args)
    except (AdminServiceError, sqlite3.Error, OSError):
        _set_notice("error", "操作未完成，数据没有按预期更新。请稍后再试。")
    else:
        _set_notice("success", success_message)
    st.rerun()


def _users_tab(actor: str) -> None:
    search = st.text_input(
        "搜索用户",
        placeholder="输入完整或部分用户名",
        max_chars=120,
    )
    users = list_user_summaries(DB_FILE, search=search, limit=200)
    if not users:
        st.info("没有找到用户。")
        return

    st.dataframe(
        [
            {
                "用户名": item.username,
                "消息": item.message_count,
                "记忆": item.memory_count,
                "最近聊天": item.last_message_at or "—",
            }
            for item in users
        ],
        hide_index=True,
        use_container_width=True,
    )

    usernames = [item.username for item in users]
    selected = st.selectbox("维护账号", usernames)
    selected_summary = next(item for item in users if item.username == selected)
    suffix = _safe_widget_suffix(selected)
    st.text(f"当前维护账号：{selected}")

    detail_columns = st.columns(3)
    detail_columns[0].metric("消息", selected_summary.message_count)
    detail_columns[1].metric("记忆", selected_summary.memory_count)
    detail_columns[2].metric("最近聊天", selected_summary.last_message_at or "暂无")

    st.subheader("账号维护")
    with st.expander("重置密码"):
        with st.form(f"reset_password_{suffix}", clear_on_submit=True):
            password = st.text_input(
                "新密码",
                type="password",
                max_chars=1024,
            )
            confirmation = st.text_input(
                "再次输入新密码",
                type="password",
                max_chars=1024,
            )
            reset = st.form_submit_button("确认重置")
        if reset:
            if len(password) < 8:
                st.error("新密码至少需要 8 个字符。")
            elif password != confirmation:
                st.error("两次输入的密码不一致。")
            else:
                password_hash = _PASSWORD_HASHER.hash(password)
                _run_user_action(
                    replace_user_password_hash,
                    "账号密码已重置。",
                    DB_FILE,
                    actor,
                    selected,
                    password_hash,
                )

    with st.expander("清空聊天记录"):
        st.caption("只删除数据库中的聊天记录；共享语音缓存不会被自动删除。")
        st.text(f"确认用户名：{selected}")
        with st.form(f"clear_history_{suffix}", clear_on_submit=True):
            confirmation = st.text_input("输入上方用户名以确认")
            clear_history = st.form_submit_button("永久清空聊天记录")
        if clear_history:
            if not hmac.compare_digest(
                confirmation.encode("utf-8"), selected.encode("utf-8")
            ):
                st.error("确认用户名不匹配。")
            else:
                _run_user_action(
                    clear_user_history,
                    "该账号的聊天记录已清空。",
                    DB_FILE,
                    actor,
                    selected,
                )

    with st.expander("清空用户记忆"):
        st.text(f"确认用户名：{selected}")
        with st.form(f"clear_memories_{suffix}", clear_on_submit=True):
            confirmation = st.text_input("输入上方用户名以确认")
            clear_memory = st.form_submit_button("永久清空用户记忆")
        if clear_memory:
            if not hmac.compare_digest(
                confirmation.encode("utf-8"), selected.encode("utf-8")
            ):
                st.error("确认用户名不匹配。")
            else:
                _run_user_action(
                    clear_user_memories,
                    "该账号的用户记忆已清空。",
                    DB_FILE,
                    actor,
                    selected,
                )

    with st.expander("删除账号"):
        st.warning(
            "这会永久删除账号、聊天记录和用户记忆，无法撤销；"
            "后台审计会保留目标用户名与删除数量。"
        )
        phrase = f"删除 {selected}"
        st.text(f"确认文字：{phrase}")
        with st.form(f"delete_user_{suffix}", clear_on_submit=True):
            confirmation = st.text_input("输入上方确认文字")
            delete_account = st.form_submit_button("永久删除账号")
        if delete_account:
            if not hmac.compare_digest(
                confirmation.encode("utf-8"), phrase.encode("utf-8")
            ):
                st.error("确认文字不匹配。")
            else:
                _run_user_action(
                    delete_user_account,
                    "账号及其聊天记录和用户记忆已永久删除。",
                    DB_FILE,
                    actor,
                    selected,
                )


def _content_tab(actor: str) -> None:
    search = st.text_input(
        "搜索需要审阅的用户",
        placeholder="输入完整或部分用户名",
        max_chars=120,
        key="admin_content_search",
    )
    users = list_user_summaries(DB_FILE, search=search, limit=200)
    if not users:
        st.info("暂无用户数据。")
        return

    selected = st.selectbox(
        "选择用户",
        [item.username for item in users],
        key="admin_content_user",
    )
    limit = st.slider("最近消息条数", 10, 100, 30, step=10)
    metadata = list_recent_messages(
        DB_FILE,
        selected,
        limit=limit,
        include_content=False,
    )
    if not metadata:
        st.info("该用户暂无聊天记录。")
        return

    st.dataframe(
        [
            {
                "ID": item.id,
                "角色": "用户" if item.type == "human" else "AI",
                "时间": item.timestamp,
                "翻译状态": item.translation_status,
                "日语": "有" if item.has_japanese else "无",
                "语音": "有" if item.has_audio else "无",
            }
            for item in metadata
        ],
        hide_index=True,
        use_container_width=True,
    )

    if not _env_flag("AOKI_ADMIN_ALLOW_MESSAGE_CONTENT"):
        st.info("聊天正文读取目前关闭；后台只加载消息元数据。")
        return

    st.warning("加载正文会读取该用户的私人聊天内容，并自动写入后台审计日志。")
    if not st.button("加载最近聊天正文", type="primary"):
        return

    messages = list_recent_messages(
        DB_FILE,
        selected,
        limit=limit,
        include_content=True,
        actor=actor,
    )
    for item in reversed(messages):
        role = "user" if item.type == "human" else "assistant"
        with st.chat_message(role):
            st.text(item.content or "")
            if item.japanese_content:
                st.caption("日语")
                st.text(item.japanese_content)
            st.caption(f"#{item.id} · {item.timestamp} · {item.translation_status}")


def _audit_tab() -> None:
    entries = list_audit_entries(DB_FILE, limit=200)
    if not entries:
        st.info("暂无管理操作记录。")
        return
    st.dataframe(
        [
            {
                "时间": item.created_at,
                "管理员": item.actor,
                "操作": item.action,
                "目标用户": item.target_username or "—",
                "结果摘要": item.detail or "—",
            }
            for item in entries
        ],
        hide_index=True,
        use_container_width=True,
    )


def _system_tab() -> None:
    api_key_configured = bool(load_env_var("DEEPSEEK_API_KEY"))
    tts_enabled = _env_flag("AOKI_TTS_ENABLED")
    content_enabled = _env_flag("AOKI_ADMIN_ALLOW_MESSAGE_CONTENT")

    status_columns = st.columns(3)
    status_columns[0].metric("聊天模型", "已配置" if api_key_configured else "未配置")
    status_columns[1].metric("语音", "已开启" if tts_enabled else "已关闭")
    status_columns[2].metric("正文审阅", "已开启" if content_enabled else "已关闭")

    st.subheader("数据库状态")
    st.caption("完整检查可能需要一点时间，只会在你点击后运行。")
    if not st.button("运行数据库检查"):
        st.caption(f"当前数据库大小：{_format_bytes(DB_FILE.stat().st_size)}")
        st.caption("后台不会显示 API 密钥、密码哈希或本地文件路径。")
        return

    health = get_database_health(DB_FILE)
    st.metric("检查结果", "正常" if health.ok else "需要检查")
    st.dataframe(
        [
            {"项目": "完整性检查", "状态": health.integrity_check},
            {"项目": "外键异常", "状态": health.foreign_key_violations},
            {"项目": "日志模式", "状态": health.journal_mode},
            {"项目": "数据库大小", "状态": _format_bytes(health.db_size_bytes)},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("后台不会显示 API 密钥、密码哈希或本地文件路径。")


def main() -> None:
    actor = _current_admin()
    if actor is None:
        _login()
        return

    _session_expiry_watch()

    st.sidebar.text(f"管理员：{actor}")
    st.sidebar.caption("聊天登录与后台登录相互独立。")
    if st.sidebar.button("退出管理后台", use_container_width=True):
        try:
            record_admin_action(DB_FILE, actor, "logout")
        except (AdminServiceError, sqlite3.Error, OSError):
            pass
        _clear_admin_session()
        st.rerun()

    st.title("管理后台")
    st.caption("默认只展示统计和元数据；敏感内容读取与维护操作都会留下审计记录。")
    _show_notice()

    try:
        section = st.radio(
            "后台栏目",
            ["总览", "用户", "聊天审阅", "操作审计", "系统"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if section == "总览":
            _overview_tab()
        elif section == "用户":
            _users_tab(actor)
        elif section == "聊天审阅":
            _content_tab(actor)
        elif section == "操作审计":
            _audit_tab()
        else:
            _system_tab()
    except (AdminServiceError, sqlite3.Error, OSError):
        st.error("后台暂时无法读取站点数据。聊天页面和现有数据未被修改。")


if __name__ == "__main__":
    main()
