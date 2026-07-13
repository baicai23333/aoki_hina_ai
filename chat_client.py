import streamlit as st
import sqlite3
import socket
import time
from pathlib import Path
from openai import APIConnectionError, APITimeoutError
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from chat_storage import (
    PLAYABLE_TRANSLATION_STATUSES,
    init_chat_storage_schema,
    list_messages,
    save_exchange,
    update_message_audio as update_stored_message_audio,
)
from pipeline_debug import build_debug_trace
from persona_pipeline import PersonaPipeline
from response_translation import ResponseTranslationService, TranslationResult
from tts_engine import TTSEngine, load_env_var, safe_cached_wav_path
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
# ================== DeepSeek API Key ==================
API_KEY = load_env_var("DEEPSEEK_API_KEY")
if not API_KEY:
    st.error("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
    st.stop()

API_BASE_URL = load_env_var("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = load_env_var("DEEPSEEK_MODEL", "deepseek-v4-flash")

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

def create_llm(temperature, max_tokens):
    return ChatDeepSeek(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=6,
        timeout=90,
        extra_body={"thinking": {"type": "disabled"}},
    )


planner_llm = create_llm(temperature=0.2, max_tokens=700)
generator_llm = create_llm(temperature=0.8, max_tokens=1024)
validator_llm = create_llm(temperature=0.0, max_tokens=700)
translator_llm = create_llm(temperature=0.0, max_tokens=1024)

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
translation_service = ResponseTranslationService(translator_llm, validator_llm)

# ================== 安装 argon2（如果未安装会提示） ==================
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    ph = PasswordHasher()
except ImportError:
    st.error("缺少 argon2-cffi 包，请在虚拟环境中运行：pip install argon2-cffi")
    st.stop()

# ================== SQLite 数据库 ==================
DB_FILE = PROJECT_DIR / "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
        ''')
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
    finally:
        conn.close()

init_db()

# ================== TTS 配置 ==================
def is_tts_enabled():
    value = (load_env_var("AOKI_TTS_ENABLED", "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_debug_enabled():
    value = (load_env_var("AOKI_DEBUG_UI", "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}

# ================== 用户函数（argon2） ==================
def register_user(username, password):
    if not username or not password:
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
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        try:
            ph.verify(row[0], password)
            return True
        except VerifyMismatchError:
            return False
    return False

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

if not st.session_state.authenticated:
    st.title("🥹 青木阳菜AI Project")
    st.caption("喵喵喵 made by baicai")
    st.caption("非官方粉丝创作 AI，与青木阳菜本人及相关官方组织无关。")
    render_portal_links()

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_register:
        st.subheader("注册新账号")
        reg_username = st.text_input("用户名", key="reg_user")
        reg_password = st.text_input("密码（支持中文/表情/任意长度）", type="password", key="reg_pass")
        if st.button("注册"):
            if register_user(reg_username.strip(), reg_password):
                st.success(f"注册成功！欢迎 {reg_username}～🥹✨")
            else:
                st.error("用户名已存在或为空，请换一个～")

    with tab_login:
        st.subheader("登录")
        login_username = st.text_input("用户名", key="login_user")
        login_password = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录"):
            if verify_user(login_username.strip(), login_password):
                st.session_state.authenticated = True
                st.session_state.username = login_username.strip()
                st.success(f"欢迎回来 {st.session_state.username}！🐈💙")
                st.rerun()
            else:
                st.error("用户名或密码错误哦～🥹")

else:
    username = st.session_state.username
    auto_tts_key = f"auto_tts_{username}"
    pending_autoplay_key = f"pending_autoplay_message_id_{username}"
    tts_flash_key = f"tts_flash_error_{username}"
    st.sidebar.write(f"当前用户：**{username}** 🩵")
    if st.sidebar.button("退出登录"):
        st.session_state.pop(pending_autoplay_key, None)
        st.session_state.pop(tts_flash_key, None)
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    render_memory_sidebar(username)
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

    # SQLite is the canonical history. Rebuild model context from immutable row IDs.
    try:
        stored_messages = list_messages(DB_FILE, username)
    except Exception:
        st.error("暂时无法读取聊天记录，请稍后再试。")
        st.stop()
    history = StreamlitChatMessageHistory(key=f"chat_{username}")
    history.clear()
    for record in stored_messages:
        if record.type == "human":
            history.add_message(HumanMessage(content=record.content))
        elif record.type == "ai":
            history.add_message(AIMessage(content=record.content))

    st.caption("Hina Bot 是非官方粉丝创作 AI，不是青木阳菜本人；日语语音由 AI 合成。")
    st.caption("和阳菜聊点什么吧～🐈✨")

    for record in stored_messages:
        if record.type == "human":
            st.chat_message("user").write(record.content)
        elif record.type == "ai":
            with st.chat_message("assistant"):
                st.markdown("**中文**")
                st.write(record.content)

                is_playable_translation = (
                    record.translation_status in PLAYABLE_TRANSLATION_STATUSES
                )
                is_legacy_translation = record.translation_status == "legacy_unverified"
                japanese_history = (
                    record.japanese_content
                    if is_playable_translation or is_legacy_translation
                    else None
                )
                if japanese_history:
                    st.markdown("**日本語**")
                    st.write(japanese_history)
                    if is_legacy_translation:
                        st.caption("旧版未复核译文：可以阅读，但不会用于语音播放。")

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
                    pipeline_result = persona_pipeline.respond(
                        user_input,
                        history.messages,
                        user_memories=current_user_memories,
                    )
                    response_text = pipeline_result.content
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
            try:
                with st.spinner("日本語に翻訳しています..."):
                    translation_result = translation_service.translate(
                        response_text,
                        pipeline_result.intent.value,
                        pipeline_result.plan.get("boundary_action", "none"),
                    )
            except Exception:
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
                st.warning("日语译文未通过本轮复核，这条消息仅显示中文。")

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
                st.error("回复已经生成，但聊天记录暂时没有保存成功；请稍后重试。")
                st.stop()

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
