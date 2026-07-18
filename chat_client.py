import streamlit as st
import sqlite3
import socket
import time
from pathlib import Path
from openai import APIConnectionError, APITimeoutError
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from account_auth import (
    get_account_auth_version,
    init_account_auth_schema,
    verify_account,
)
from artifact_presentation import render_message_artifacts
from chat_history_context import build_model_history
from chat_realtime import (
    build_realtime_unavailable_bundle,
    build_recent_updates_lookup,
    build_weather_lookup,
    realtime_grounding_is_unavailable,
)
from chat_storage import (
    PLAYABLE_TRANSLATION_STATUSES,
    init_chat_storage_schema,
    list_messages,
    save_exchange,
    update_failed_message_translation,
    update_message_audio as update_stored_message_audio,
)
from collector_worker import seed_information_sources_from_registry
from chat_presentation import (
    manual_translation_retry_available,
    translation_status_message,
)
from grounding import GroundingBundle
from information_store import init_information_schema
from message_artifacts import (
    init_message_artifacts_schema,
    list_artifacts_for_messages,
    save_ui_artifacts,
)
from pipeline_debug import build_debug_trace
from persona_pipeline import PersonaPipeline
from response_text_policy import has_hidden_or_redacted_content
from response_translation import ResponseTranslationService, TranslationResult
from runtime_profile import init_runtime_profile_schema
from runtime_ui import render_runtime_sidebar
from search_service import SearchService
from tts_engine import TTSEngine, load_env_var, safe_cached_wav_path
from tool_orchestrator import ToolOrchestrator, ToolRoute
from translation_audit import (
    DEFAULT_TRANSLATION_AUDIT_SINK,
    new_translation_operation_id,
    record_provider_exception,
    record_stored_outcome,
    record_terminal_failure,
)
from user_memory import (
    MEMORY_CATEGORIES,
    UserMemoryLimitError,
    UserMemoryValidationError,
    clear_memories,
    delete_memory,
    init_user_memory_schema,
    list_memories,
    upsert_memory,
)
from weather_service import WeatherService
# ================== DeepSeek API Key ==================
API_KEY = load_env_var("DEEPSEEK_API_KEY")
if not API_KEY:
    st.error("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
    st.stop()

API_BASE_URL = load_env_var("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = load_env_var("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEFAULT_TIMEZONE = load_env_var("AOKI_DEFAULT_TIMEZONE", "Asia/Shanghai")

# Windows DNS on this machine occasionally returns WSAHOST_NOT_FOUND (11001)
# for api.deepseek.com. Keep normal DNS as the first choice and only fall back
# for this one host; TLS still validates the original api.deepseek.com name.
DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_FALLBACK_IPS = tuple(
    item.strip()
    for item in (load_env_var("DEEPSEEK_FALLBACK_IPS", "") or "").split(",")
    if item.strip()
)

if not hasattr(socket, "_aoki_original_getaddrinfo"):
    socket._aoki_original_getaddrinfo = socket.getaddrinfo


def getaddrinfo_with_deepseek_fallback(host, port, family=0, type=0, proto=0, flags=0):
    original = socket._aoki_original_getaddrinfo
    try:
        return original(host, port, family, type, proto, flags)
    except socket.gaierror:
        if str(host).lower() != DEEPSEEK_HOST or not DEEPSEEK_FALLBACK_IPS:
            raise
        addresses = []
        for ip_address in DEEPSEEK_FALLBACK_IPS:
            addresses.extend(original(ip_address, port, family, type, proto, flags))
        return addresses


socket.getaddrinfo = getaddrinfo_with_deepseek_fallback

def create_llm(temperature, max_tokens, *, max_retries=6):
    return ChatDeepSeek(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=90,
        extra_body={"thinking": {"type": "disabled"}},
    )


planner_llm = create_llm(temperature=0.2, max_tokens=700)
generator_llm = create_llm(temperature=0.8, max_tokens=1024)
validator_llm = create_llm(temperature=0.0, max_tokens=700)
# Translation retries are explicitly bounded and audited by
# ResponseTranslationService. Disable hidden SDK retries for both translation
# stages so an application attempt maps to one provider request.
translator_llm = create_llm(temperature=0.0, max_tokens=1024, max_retries=0)
translation_reviewer_llm = create_llm(
    temperature=0.0,
    max_tokens=700,
    max_retries=0,
)
tool_router_llm = create_llm(temperature=0.0, max_tokens=500)

PROJECT_DIR = Path(__file__).resolve().parent
try:
    PERSONA_HISTORY_MESSAGES = max(2, int(load_env_var("AOKI_PERSONA_HISTORY_MESSAGES", "12")))
except (TypeError, ValueError):
    PERSONA_HISTORY_MESSAGES = 12
persona_pipeline = PersonaPipeline(
    planner_llm=planner_llm,
    generator_llm=generator_llm,
    validator_llm=validator_llm,
    persona_dir=PROJECT_DIR / "persona",
    max_history_messages=PERSONA_HISTORY_MESSAGES,
)
translation_service = ResponseTranslationService(
    translator_llm,
    translation_reviewer_llm,
)
try:
    SEARCH_TIMEOUT_SECONDS = float(
        load_env_var("AOKI_SEARCH_TIMEOUT_SECONDS", "15") or "15"
    )
except (TypeError, ValueError):
    SEARCH_TIMEOUT_SECONDS = 15.0
if not 0 < SEARCH_TIMEOUT_SECONDS <= 60:
    SEARCH_TIMEOUT_SECONDS = 15.0
SEARCH_SERVICE_INIT_FAILED = False
try:
    search_service = SearchService(
        tavily_api_key=load_env_var("TAVILY_API_KEY"),
        brave_api_key=load_env_var("BRAVE_SEARCH_API_KEY"),
        timeout_seconds=SEARCH_TIMEOUT_SECONDS,
    )
except Exception:
    search_service = None
    SEARCH_SERVICE_INIT_FAILED = True
weather_service = WeatherService()

# ================== 安装 argon2（如果未安装会提示） ==================
try:
    from argon2 import PasswordHasher
    ph = PasswordHasher()
except ImportError:
    st.error("缺少 argon2-cffi 包，请在虚拟环境中运行：pip install argon2-cffi")
    st.stop()

# ================== SQLite 数据库 ==================
DB_FILE = PROJECT_DIR / "chat_history.db"
MAX_USERNAME_LENGTH = 80

def init_db():
    optional_issues = []
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                auth_version INTEGER NOT NULL DEFAULT 1 CHECK (auth_version >= 1)
            )
        ''')
        init_account_auth_schema(conn)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(chat_history)")}
        if "japanese_content" not in columns:
            cursor.execute("ALTER TABLE chat_history ADD COLUMN japanese_content TEXT")
        if "audio_path" not in columns:
            cursor.execute("ALTER TABLE chat_history ADD COLUMN audio_path TEXT")
        init_chat_storage_schema(conn)
        init_user_memory_schema(conn)
        conn.commit()
        for name, initializer in (
            ("runtime_profile", init_runtime_profile_schema),
            ("message_artifacts", init_message_artifacts_schema),
            ("official_information", init_information_schema),
        ):
            try:
                initializer(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                optional_issues.append(name)
    finally:
        conn.close()
    return tuple(optional_issues)

OPTIONAL_INIT_ISSUES = init_db()
if SEARCH_SERVICE_INIT_FAILED:
    OPTIONAL_INIT_ISSUES = (*OPTIONAL_INIT_ISSUES, "search_registry")


@st.cache_resource(show_spinner=False)
def _seed_information_sources_once(db_path: str, registry_path: str):
    return seed_information_sources_from_registry(db_path, registry_path)


try:
    if "official_information" not in OPTIONAL_INIT_ISSUES:
        _seed_information_sources_once(
            str(DB_FILE),
            str(PROJECT_DIR / "official_sources.json"),
        )
except Exception:
    OPTIONAL_INIT_ISSUES = (*OPTIONAL_INIT_ISSUES, "official_source_seed")

if OPTIONAL_INIT_ISSUES:
    st.warning("部分即时信息功能暂时无法初始化；普通聊天和已有记录仍可继续使用。")

# ================== TTS 配置 ==================
def is_tts_enabled():
    value = (load_env_var("AOKI_TTS_ENABLED", "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_debug_enabled():
    value = (load_env_var("AOKI_DEBUG_UI", "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_tool_calling_enabled():
    value = (load_env_var("AOKI_TOOL_CALLING_ENABLED", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_weather_enabled():
    value = (load_env_var("AOKI_WEATHER_ENABLED", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}

# ================== 用户函数（argon2） ==================
def register_user(username, password):
    if not username or len(username) > MAX_USERNAME_LENGTH or not password:
        return False
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        password_hash = ph.hash(password)  # Argon2 支持任意长度
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    return verify_account(DB_FILE, username, password, ph)


def get_user_auth_version(username):
    return get_account_auth_version(DB_FILE, username)

# ================== Streamlit 界面 ==================
PORTAL_LINKS = [
    ("AI Write", "https://aiwrite.top"),
    ("Blog", "https://blog.baicai-qwq.xyz"),
    ("Chat 子域", "https://chat.baicai-qwq.xyz"),
]


def render_portal_links():
    st.subheader("站点入口")
    cols = st.columns(2)
    for index, (label, url) in enumerate(PORTAL_LINKS):
        cols[index % 2].markdown(f"[{label}]({url})")


MEMORY_CATEGORY_LABELS = {
    "preferred_name": "希望使用的称呼",
    "interest": "兴趣",
    "conversation_preference": "聊天偏好",
    "goal": "目标",
}


def memory_category_label(category):
    return MEMORY_CATEGORY_LABELS.get(category, category)


def set_memory_notice(username, message, level="success"):
    st.session_state[f"memory_notice_{username}"] = (level, message)


def render_memory_sidebar(username):
    notice = st.session_state.pop(f"memory_notice_{username}", None)
    try:
        memories = list_memories(DB_FILE, username)
        memory_store_available = True
    except Exception:
        memories = []
        memory_store_available = False
    category_options = sorted(MEMORY_CATEGORIES)

    with st.sidebar.expander("我的聊天记忆", expanded=False):
        st.caption("你可以决定让我长期记住什么；这里的内容不会自动从聊天中提取。")
        st.caption("与当前话题相关的记忆会随本轮消息发送给 DeepSeek，用于个性化回复。")
        st.warning("请不要保存密码、住址、联系方式、证件号码或其他敏感信息。")
        if not memory_store_available:
            st.error("暂时无法读取聊天记忆；普通聊天仍可继续。")
            return

        if notice:
            level, message = notice
            if level == "error":
                st.error(message)
            else:
                st.success(message)

        with st.form(f"memory_add_{username}", clear_on_submit=True):
            category = st.selectbox(
                "类型",
                category_options,
                format_func=memory_category_label,
            )
            memory_key = st.text_input(
                "记忆名称",
                placeholder="例如：喜欢的称呼",
                max_chars=80,
            )
            memory_value = st.text_area(
                "希望记住的内容",
                placeholder="例如：希望被叫作白菜",
                max_chars=500,
            )
            save_new = st.form_submit_button("保存这条记忆", use_container_width=True)

        if save_new:
            clean_key = memory_key.strip()
            clean_value = memory_value.strip()
            if not clean_key or not clean_value:
                st.error("请把记忆名称和内容都填写完整。")
            else:
                try:
                    upsert_memory(
                        DB_FILE,
                        username,
                        category,
                        clean_key,
                        clean_value,
                    )
                    set_memory_notice(
                        username,
                        "已经记住啦。同一类型和名称再次保存时，会更新原来的内容。",
                    )
                    st.rerun()
                except UserMemoryLimitError:
                    st.error("每个账号最多保存 50 条记忆；请先删除一条再添加。")
                except UserMemoryValidationError:
                    st.error("这条记忆的类型、名称或内容不符合保存要求。")
                except Exception:
                    st.error("这条记忆暂时没有保存成功，请稍后再试。")

        st.caption("同一类型和名称再次保存，会更新原来的内容。")
        st.divider()

        if not memories:
            st.caption("还没有保存任何聊天记忆。")
        else:
            st.markdown("**已经保存**")
            for memory in memories:
                st.markdown(
                    f"**{memory_category_label(memory.category)} · {memory.memory_key}**"
                )
                edited_value = st.text_area(
                    "记忆内容",
                    value=memory.memory_value,
                    key=f"memory_value_{username}_{memory.id}",
                    label_visibility="collapsed",
                    max_chars=500,
                )
                update_col, delete_col = st.columns(2)
                if update_col.button(
                    "保存修改",
                    key=f"memory_update_{username}_{memory.id}",
                    use_container_width=True,
                ):
                    clean_value = edited_value.strip()
                    if not clean_value:
                        st.error("记忆内容不能为空；如果不再需要，请使用永久删除。")
                    else:
                        try:
                            upsert_memory(
                                DB_FILE,
                                username,
                                memory.category,
                                memory.memory_key,
                                clean_value,
                            )
                            set_memory_notice(username, "这条记忆已经更新。")
                            st.rerun()
                        except UserMemoryValidationError:
                            st.error("记忆内容不符合保存要求。")
                        except Exception:
                            st.error("这条记忆暂时没有更新成功，请稍后再试。")
                if delete_col.button(
                    "永久删除",
                    key=f"memory_delete_{username}_{memory.id}",
                    use_container_width=True,
                ):
                    try:
                        delete_memory(DB_FILE, username, memory.id)
                        set_memory_notice(username, "这条记忆已经永久删除。")
                        st.rerun()
                    except Exception:
                        st.error("这条记忆暂时没有删除成功，请稍后再试。")
                st.divider()

            clear_version_key = f"memory_clear_version_{username}"
            clear_version = st.session_state.get(clear_version_key, 0)
            confirm_clear = st.checkbox(
                "我确认要永久删除全部聊天记忆",
                key=f"memory_clear_confirm_{username}_{clear_version}",
            )
            if st.button(
                "清空全部记忆",
                key=f"memory_clear_all_{username}_{clear_version}",
                disabled=not confirm_clear,
                use_container_width=True,
            ):
                try:
                    clear_memories(DB_FILE, username)
                    st.session_state[clear_version_key] = clear_version + 1
                    set_memory_notice(username, "全部聊天记忆已经永久删除。")
                    st.rerun()
                except Exception:
                    st.error("暂时无法清空记忆，请稍后再试。")


def render_debug_sidebar(username):
    if not is_debug_enabled():
        return
    trace = st.session_state.get(f"pipeline_trace_{username}")
    with st.sidebar.expander("管线调试（脱敏）", expanded=False):
        st.caption("仅显示路由、证据 ID、状态和耗时；不包含聊天内容、提示词、路径或原始错误。")
        if trace is None:
            st.caption("完成一轮对话后，这里会显示最近一次脱敏记录。")
        else:
            st.json(trace)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.auth_version = None

if not st.session_state.authenticated:
    st.title("🥹 青木阳菜AI Project")
    st.caption("喵喵喵 made by baicai")
    st.caption("非官方粉丝创作 AI，与青木阳菜本人及相关官方组织无关。")
    render_portal_links()
    auth_notice = st.session_state.pop("auth_notice", None)
    if auth_notice:
        st.warning(auth_notice)

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_register:
        st.subheader("注册新账号")
        reg_username = st.text_input(
            "用户名",
            key="reg_user",
            max_chars=MAX_USERNAME_LENGTH,
        )
        reg_password = st.text_input("密码（支持中文/表情/任意长度）", type="password", key="reg_pass")
        if st.button("注册"):
            if register_user(reg_username.strip(), reg_password):
                st.success(f"注册成功！欢迎 {reg_username}～🥹✨")
            else:
                st.error("用户名已存在或为空，请换一个～")

    with tab_login:
        st.subheader("登录")
        login_username = st.text_input(
            "用户名",
            key="login_user",
            max_chars=MAX_USERNAME_LENGTH,
        )
        login_password = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录"):
            auth_version = verify_user(login_username.strip(), login_password)
            if auth_version is not None:
                st.session_state.authenticated = True
                st.session_state.username = login_username.strip()
                st.session_state.auth_version = auth_version
                st.success(f"欢迎回来 {st.session_state.username}！🐈💙")
                st.rerun()
            else:
                st.error("用户名或密码错误哦～🥹")

else:
    username = st.session_state.username
    current_auth_version = get_user_auth_version(username)
    if (
        current_auth_version is None
        or st.session_state.get("auth_version") != current_auth_version
    ):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.auth_version = None
        st.session_state.auth_notice = "账号登录状态已失效，请重新登录。"
        st.rerun()
    auto_tts_key = f"auto_tts_{username}"
    pending_autoplay_key = f"pending_autoplay_message_id_{username}"
    tts_flash_key = f"tts_flash_error_{username}"
    translation_retry_flash_key = f"translation_retry_flash_{username}"
    st.sidebar.write(f"当前用户：**{username}** 🩵")
    if st.sidebar.button("退出登录"):
        st.session_state.pop(pending_autoplay_key, None)
        st.session_state.pop(tts_flash_key, None)
        st.session_state.pop(translation_retry_flash_key, None)
        st.session_state.pop(f"runtime_geo_version_{username}", None)
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.auth_version = None
        st.rerun()

    render_memory_sidebar(username)
    runtime_context = render_runtime_sidebar(
        DB_FILE,
        username,
        fallback_timezone=DEFAULT_TIMEZONE or "Asia/Shanghai",
    )
    render_debug_sidebar(username)

    tts_enabled = is_tts_enabled()
    if auto_tts_key not in st.session_state:
        st.session_state[auto_tts_key] = tts_enabled
    if not tts_enabled:
        st.session_state[auto_tts_key] = False
        st.session_state.pop(pending_autoplay_key, None)

    auto_tts = st.sidebar.toggle(
        "Auto TTS",
        key=auto_tts_key,
        disabled=not tts_enabled,
    )
    if not tts_enabled:
        st.sidebar.caption("AOKI_TTS_ENABLED=0，已关闭语音播放")
    tts_flash_error = st.session_state.pop(tts_flash_key, None)
    if tts_flash_error:
        st.warning(tts_flash_error)
    translation_retry_flash = st.session_state.pop(
        translation_retry_flash_key,
        None,
    )
    if translation_retry_flash == "validated":
        st.success("日语译文已重新生成并通过复核。需要语音时请点击 Play。")
    elif translation_retry_flash == "rejected":
        st.warning("重新生成的日语译文仍未通过安全复核，本条继续仅显示中文。")
    elif translation_retry_flash == "failed":
        st.warning("日语翻译仍暂时失败，你可以稍后再次点击“重新翻译”。")
    elif translation_retry_flash == "stale":
        st.info("这条消息的翻译状态已在其他页面更新，当前记录已经刷新。")

    # SQLite is the canonical history. Rebuild model context from immutable row IDs.
    try:
        stored_messages = list_messages(DB_FILE, username)
    except Exception:
        st.error("暂时无法读取聊天记录，请稍后再试。")
        st.stop()
    try:
        artifacts_by_message = list_artifacts_for_messages(
            DB_FILE,
            [record.id for record in stored_messages if record.type == "ai"],
        )
    except Exception:
        artifacts_by_message = {}
        st.warning("聊天记录已读取，但部分天气或来源卡片暂时无法显示。")
    history = StreamlitChatMessageHistory(key=f"chat_{username}")
    history.clear()
    for message in build_model_history(stored_messages):
        history.add_message(message)

    st.caption("Hina Bot 是非官方粉丝创作 AI，不是青木阳菜本人；日语语音由 AI 合成。")
    st.caption("和阳菜聊点什么吧～🐈✨")

    for record in stored_messages:
        if record.type == "human":
            st.chat_message("user").write(record.content)
        elif record.type == "ai":
            with st.chat_message("assistant"):
                st.markdown("**中文**")
                st.write(record.content)

                source_has_hidden_content = has_hidden_or_redacted_content(
                    record.content
                )
                is_playable_translation = (
                    record.translation_status in PLAYABLE_TRANSLATION_STATUSES
                    and not source_has_hidden_content
                )
                is_legacy_translation = (
                    record.translation_status == "legacy_unverified"
                    and not source_has_hidden_content
                )
                japanese_history = (
                    record.japanese_content
                    if is_playable_translation or is_legacy_translation
                    else None
                )
                if source_has_hidden_content:
                    if st.session_state.get(pending_autoplay_key) == record.id:
                        st.session_state.pop(pending_autoplay_key, None)
                    st.warning(
                        "这条中文包含隐藏、删除线或屏蔽标记；为避免双语内容不一致，"
                        "已停用其日语和语音。"
                    )
                if japanese_history:
                    st.markdown("**日本語**")
                    st.write(japanese_history)
                    if is_legacy_translation:
                        st.caption("旧版未复核译文：可以阅读，但不会用于语音播放。")
                translation_notice = translation_status_message(
                    record.translation_status
                )
                if translation_notice:
                    st.warning(translation_notice)

                if manual_translation_retry_available(
                    record.translation_status,
                    record.translation_issue_code,
                    source_has_hidden_content=source_has_hidden_content,
                ):
                    retry_key = f"translation_retry_{username}_{record.id}"
                    if st.button("重新翻译", key=retry_key):
                        retry_operation_id = new_translation_operation_id()
                        try:
                            with st.spinner("正在重新生成并复核日语译文..."):
                                retry_result = translation_service.translate_ordinary(
                                    record.content,
                                    operation_id=retry_operation_id,
                                )
                        except Exception as exception:
                            record_provider_exception(
                                DEFAULT_TRANSLATION_AUDIT_SINK,
                                operation_id=retry_operation_id,
                                stage="orchestration",
                                application_attempt=0,
                                retry_scheduled=False,
                                exception=exception,
                                issue_code="translator_exception",
                            )
                            record_terminal_failure(
                                DEFAULT_TRANSLATION_AUDIT_SINK,
                                operation_id=retry_operation_id,
                                stage="orchestration",
                                application_attempt=0,
                                issue_code="translator_exception",
                            )
                            retry_result = TranslationResult(
                                text="",
                                status="failed",
                                issue_codes=("translator_exception",),
                            )

                        retry_japanese = (
                            retry_result.text
                            if retry_result.status in PLAYABLE_TRANSLATION_STATUSES
                            else None
                        )
                        retry_issue_code = (
                            retry_result.issue_codes[0]
                            if retry_result.issue_codes
                            else None
                        )
                        try:
                            updated = update_failed_message_translation(
                                DB_FILE,
                                username,
                                record.id,
                                record.content,
                                record.translation_issue_code,
                                retry_japanese,
                                retry_result.status,
                                retry_issue_code,
                            )
                        except Exception:
                            record_terminal_failure(
                                DEFAULT_TRANSLATION_AUDIT_SINK,
                                operation_id=retry_operation_id,
                                stage="storage",
                                application_attempt=0,
                                issue_code="storage_exception",
                            )
                            st.warning("译文已经处理，但更新聊天记录失败，请稍后再试。")
                        else:
                            if not updated:
                                record_terminal_failure(
                                    DEFAULT_TRANSLATION_AUDIT_SINK,
                                    operation_id=retry_operation_id,
                                    stage="storage",
                                    application_attempt=0,
                                    issue_code="storage_compare_and_swap_miss",
                                )
                                st.session_state[translation_retry_flash_key] = "stale"
                                st.rerun()
                            else:
                                record_stored_outcome(
                                    DEFAULT_TRANSLATION_AUDIT_SINK,
                                    operation_id=retry_operation_id,
                                    translation_status=retry_result.status,
                                    issue_code=retry_issue_code,
                                    message_id=record.id,
                                )
                                if (
                                    st.session_state.get(pending_autoplay_key)
                                    == record.id
                                ):
                                    st.session_state.pop(pending_autoplay_key, None)
                                st.session_state[translation_retry_flash_key] = (
                                    retry_result.status
                                )
                                st.rerun()

                render_message_artifacts(
                    artifacts_by_message.get(record.id, ()),
                    st_module=st,
                )

                verified_audio_path = (
                    safe_cached_wav_path(record.audio_path)
                    if tts_enabled and is_playable_translation and record.audio_path
                    else None
                )
                if verified_audio_path is not None:
                    should_autoplay_history = (
                        st.session_state.get(pending_autoplay_key) == record.id
                    )
                    st.audio(
                        str(verified_audio_path),
                        format="audio/wav",
                        autoplay=should_autoplay_history,
                    )
                    if should_autoplay_history:
                        st.session_state.pop(pending_autoplay_key, None)
                elif tts_enabled and is_playable_translation and japanese_history:
                    if st.session_state.get(pending_autoplay_key) == record.id:
                        st.session_state.pop(pending_autoplay_key, None)
                    play_key = f"tts_history_play_{record.id}"
                    if st.button("Play", key=play_key, disabled=not tts_enabled):
                        try:
                            with st.spinner("正在合成日语语音..."):
                                wav_path = TTSEngine.get().synthesize_to_file(japanese_history)
                            updated = update_stored_message_audio(
                                DB_FILE,
                                username,
                                record.id,
                                str(wav_path),
                            )
                            if not updated:
                                raise RuntimeError("audio metadata update was rejected")
                            st.session_state[pending_autoplay_key] = record.id
                            st.rerun()
                        except Exception:
                            st.warning("语音暂时生成失败，请稍后再试。")

    if user_input := st.chat_input("说点什么给阳菜听吧～"):
        st.chat_message("user").write(user_input)

        stage_duration_ms = {}
        pipeline_started = time.perf_counter()

        with st.chat_message("assistant"):
            with st.spinner("阳菜在思考中...🥹"):
                try:
                    try:
                        current_user_memories = list_memories(
                            DB_FILE,
                            username,
                        )
                    except Exception:
                        current_user_memories = []
                        st.warning("本轮暂时无法读取长期记忆，将按普通对话继续。")
                    intent = persona_pipeline.classifier.classify(user_input)
                    realtime_grounding = GroundingBundle.empty()
                    realtime_unavailable = False
                    route_probe = ToolOrchestrator(None).route(
                        user_input,
                        safety_label=intent.value,
                    )
                    if route_probe is not ToolRoute.NONE:
                        if is_tool_calling_enabled():
                            orchestrator = ToolOrchestrator(
                                tool_router_llm,
                                search_service=search_service,
                                get_weather=(
                                    build_weather_lookup(
                                        weather_service,
                                        runtime_context,
                                    )
                                    if is_weather_enabled()
                                    else None
                                ),
                                query_recent_updates=build_recent_updates_lookup(
                                    DB_FILE
                                ),
                            )
                            try:
                                tool_result = orchestrator.orchestrate(
                                    user_input,
                                    (),
                                    safety_label=intent.value,
                                )
                                realtime_grounding = tool_result.grounding
                            except Exception:
                                realtime_grounding = build_realtime_unavailable_bundle(
                                    route_probe.value
                                )
                        else:
                            realtime_grounding = build_realtime_unavailable_bundle(
                                route_probe.value
                            )
                        if not realtime_grounding.facts:
                            realtime_grounding = realtime_grounding.merge(
                                build_realtime_unavailable_bundle(route_probe.value)
                            )
                        realtime_unavailable = realtime_grounding_is_unavailable(
                            realtime_grounding,
                            route_probe.value,
                        )
                    if realtime_unavailable:
                        pipeline_result = persona_pipeline.realtime_unavailable_result(
                            user_input,
                            route_probe.value,
                        )
                    else:
                        pipeline_result = persona_pipeline.respond(
                            user_input,
                            history.messages,
                            user_memories=current_user_memories,
                            runtime_context=runtime_context,
                            grounding=realtime_grounding,
                        )
                    response_text = pipeline_result.content
                    if has_hidden_or_redacted_content(response_text):
                        st.error(
                            "本轮回复包含无法安全呈现的隐藏或删除内容，"
                            "已停止翻译和保存，请重新发送。"
                        )
                        st.stop()
                    if (
                        pipeline_result.intent.value
                        in {"daily_chat", "emotion_support", "music_advice", "fan_chat"}
                        and persona_pipeline.rule_validator.has_unexpected_japanese_output(
                            response_text
                        )
                    ):
                        st.error("本轮回复未通过输出语言检查，请换一种说法再试。")
                        st.stop()
                except (APIConnectionError, APITimeoutError):
                    st.error("暂时无法连接 DeepSeek，请稍后再试。应用会自动重试连接。")
                    st.stop()
                except Exception:
                    st.error("这轮对话暂时没有生成成功，请稍后再试。")
                    st.stop()
            stage_duration_ms["pipeline"] = (
                time.perf_counter() - pipeline_started
            ) * 1000

            translation_started = time.perf_counter()
            translation_operation_id = new_translation_operation_id()
            try:
                with st.spinner("日本語に翻訳しています..."):
                    translation_result = translation_service.translate(
                        response_text,
                        pipeline_result.intent.value,
                        pipeline_result.plan.get("boundary_action", "none"),
                        operation_id=translation_operation_id,
                    )
            except Exception as exception:
                record_provider_exception(
                    DEFAULT_TRANSLATION_AUDIT_SINK,
                    operation_id=translation_operation_id,
                    stage="orchestration",
                    application_attempt=0,
                    retry_scheduled=False,
                    exception=exception,
                    issue_code="translator_exception",
                )
                record_terminal_failure(
                    DEFAULT_TRANSLATION_AUDIT_SINK,
                    operation_id=translation_operation_id,
                    stage="orchestration",
                    application_attempt=0,
                    issue_code="translator_exception",
                )
                translation_result = TranslationResult(
                    text="",
                    status="failed",
                    issue_codes=("translator_exception",),
                )
            stage_duration_ms["translation"] = (
                time.perf_counter() - translation_started
            ) * 1000
            japanese_text = (
                translation_result.text
                if translation_result.status in PLAYABLE_TRANSLATION_STATUSES
                else ""
            )

            st.markdown("**中文**")
            st.write(response_text)
            if japanese_text:
                st.markdown("**日本語**")
                st.write(japanese_text)
            else:
                translation_notice = translation_status_message(
                    translation_result.status
                )
                if translation_notice:
                    st.warning(translation_notice)

            issue_code = (
                translation_result.issue_codes[0]
                if translation_result.issue_codes
                else None
            )
            try:
                _, ai_message_id = save_exchange(
                    DB_FILE,
                    username,
                    user_input,
                    response_text,
                    japanese_text or None,
                    translation_result.status,
                    issue_code,
                    None,
                )
            except Exception:
                record_terminal_failure(
                    DEFAULT_TRANSLATION_AUDIT_SINK,
                    operation_id=translation_operation_id,
                    stage="storage",
                    application_attempt=0,
                    issue_code="storage_exception",
                )
                st.error("回复已经生成，但聊天记录暂时没有保存成功；请稍后重试。")
                st.stop()
            record_stored_outcome(
                DEFAULT_TRANSLATION_AUDIT_SINK,
                operation_id=translation_operation_id,
                translation_status=translation_result.status,
                issue_code=issue_code,
                message_id=ai_message_id,
            )

            if realtime_grounding.ui_artifacts:
                try:
                    save_ui_artifacts(
                        DB_FILE,
                        ai_message_id,
                        realtime_grounding.ui_artifacts,
                    )
                except Exception:
                    st.warning("回复已保存，但本轮天气或来源卡片暂时无法保存。")

            audio_attached = False
            tts_status = "disabled" if not tts_enabled else "not_requested"
            should_auto_play = tts_enabled and auto_tts and bool(japanese_text)
            if should_auto_play:
                tts_started = time.perf_counter()
                try:
                    with st.spinner("正在合成日语语音..."):
                        wav_path = TTSEngine.get().synthesize_to_file(japanese_text)
                    audio_attached = update_stored_message_audio(
                        DB_FILE,
                        username,
                        ai_message_id,
                        str(wav_path),
                    )
                    if not audio_attached:
                        raise RuntimeError("audio metadata update was rejected")
                    tts_status = "succeeded"
                except Exception:
                    tts_status = "failed"
                    st.session_state[tts_flash_key] = "语音暂时生成失败，请稍后再试。"
                stage_duration_ms["tts"] = (
                    time.perf_counter() - tts_started
                ) * 1000

        history.add_user_message(user_input)
        history.add_ai_message(response_text)
        st.session_state[f"pipeline_trace_{username}"] = build_debug_trace(
            pipeline_result,
            translation_status=translation_result.status,
            tts_status=tts_status,
            stage_duration_ms=stage_duration_ms,
        )
        if audio_attached:
            st.session_state[pending_autoplay_key] = ai_message_id
        st.rerun()
