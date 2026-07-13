import streamlit as st
import sqlite3
import hashlib
import socket
from datetime import datetime
from pathlib import Path
from openai import APIConnectionError, APITimeoutError
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from persona_pipeline import PersonaPipeline
from tts_engine import TTSEngine, load_env_var
ph = PasswordHasher()
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

# ================== 安装 argon2（如果未安装会提示） ==================
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    ph = PasswordHasher()
except ImportError:
    st.error("缺少 argon2-cffi 包，请在虚拟环境中运行：pip install argon2-cffi")
    st.stop()

translation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "将用户提供的中文回复翻译成自然、亲切、适合语音朗读的日语。"
        "忠实保留原意、语气、称呼和情绪，不要添加信息。"
        "只输出日语译文，不要解释，不要添加标题或引号。",
    ),
    ("human", "{text}"),
])
translation_chain = translation_prompt | generator_llm

# ================== SQLite 数据库 ==================
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn.commit()
    conn.close()

init_db()

# ================== TTS 配置 ==================
def is_tts_enabled():
    value = (load_env_var("AOKI_TTS_ENABLED", "0") or "").strip().lower()
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

# ================== 聊天记录函数 ==================
def load_history(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT type, content FROM chat_history WHERE username = ? ORDER BY timestamp ASC", (username,))
    rows = cursor.fetchall()
    messages = []
    for row in rows:
        if row[0] == "human":
            messages.append(HumanMessage(content=row[1]))
        elif row[0] == "ai":
            messages.append(AIMessage(content=row[1]))
    conn.close()
    return messages

def load_ai_metadata(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, content, japanese_content, audio_path FROM chat_history "
        "WHERE username = ? AND type = 'ai' ORDER BY id ASC",
        (username,),
    )
    metadata = {
        content: {"id": message_id, "japanese": japanese or "", "audio_path": audio_path or ""}
        for message_id, content, japanese, audio_path in cursor.fetchall()
    }
    conn.close()
    return metadata


def save_message(username, msg_type, content, japanese_content=None, audio_path=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO chat_history "
        "(username, type, content, japanese_content, audio_path, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (username, msg_type, content, japanese_content, audio_path, timestamp),
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id


def update_message_audio(message_id, audio_path):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_history SET audio_path = ? WHERE id = ?", (audio_path, message_id))
    conn.commit()
    conn.close()

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


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🥹 青木阳菜AI Project")
    st.caption("喵喵喵 made by baicai")
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
    st.sidebar.write(f"当前用户：**{st.session_state.username}** 🩵")
    if st.sidebar.button("退出登录"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    tts_enabled = is_tts_enabled()
    if "auto_tts" not in st.session_state:
        st.session_state.auto_tts = tts_enabled
    if "tts_auto_played_id" not in st.session_state:
        st.session_state.tts_auto_played_id = None

    auto_tts = st.sidebar.toggle("Auto TTS", value=st.session_state.auto_tts, disabled=not tts_enabled)
    st.session_state.auto_tts = auto_tts
    if not tts_enabled:
        st.sidebar.caption("AOKI_TTS_ENABLED=0，已关闭语音播放")
    tts_flash_error = st.session_state.pop("tts_flash_error", None)
    if tts_flash_error:
        st.warning(tts_flash_error)

    # 加载历史 + 聊天逻辑
    history = StreamlitChatMessageHistory(key=f"chat_{st.session_state.username}")
    history_loaded_key = f"history_loaded_{st.session_state.username}"
    if not st.session_state.get(history_loaded_key):
        loaded_messages = load_history(st.session_state.username)
        for msg in loaded_messages:
            history.add_message(msg)
        st.session_state[history_loaded_key] = True
    ai_metadata = load_ai_metadata(st.session_state.username)

    st.caption("和阳菜聊点什么吧～🐈✨")

    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown("**中文**")
                st.write(msg.content)
                metadata = ai_metadata.get(msg.content, {})
                japanese_history = metadata.get("japanese", "")
                if japanese_history:
                    st.markdown("**日本語**")
                    st.write(japanese_history)

                audio_path = metadata.get("audio_path", "")
                audio_exists = bool(audio_path) and Path(audio_path).exists()
                if audio_exists:
                    should_autoplay_history = st.session_state.get("pending_autoplay_audio") == audio_path
                    st.audio(audio_path, format="audio/wav", autoplay=should_autoplay_history)
                    if should_autoplay_history:
                        st.session_state.pending_autoplay_audio = None
                elif japanese_history:
                    play_key = f"tts_history_play_{metadata.get('id', hashlib.sha256(msg.content.encode()).hexdigest())}"
                    if st.button("Play", key=play_key, disabled=not tts_enabled):
                        try:
                            with st.spinner("正在合成日语语音..."):
                                wav_path = TTSEngine.get().synthesize_to_file(japanese_history)
                            update_message_audio(metadata["id"], str(wav_path))
                            st.session_state.pending_autoplay_audio = str(wav_path)
                            st.rerun()
                        except Exception as exc:
                            st.warning(f"TTS 播放失败：{exc}")

    if user_input := st.chat_input("说点什么给阳菜听吧～"):
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("阳菜在思考中...🥹"):
                try:
                    pipeline_result = persona_pipeline.respond(user_input, history.messages)
                    response_text = pipeline_result.content
                except (APIConnectionError, APITimeoutError):
                    st.error("暂时无法连接 DeepSeek，请稍后再试。应用会自动重试连接。")
                    st.stop()
                except Exception as exc:
                    st.error(f"对话请求失败：{type(exc).__name__}: {exc}")
                    st.stop()

            japanese_text = ""
            try:
                with st.spinner("日本語に翻訳しています..."):
                    translated = translation_chain.invoke({"text": response_text})
                    japanese_text = translated.content.strip()
            except Exception as exc:
                st.warning(f"日语翻译失败：{type(exc).__name__}: {exc}")

            st.markdown("**中文**")
            st.write(response_text)
            if japanese_text:
                st.markdown("**日本語**")
                st.write(japanese_text)

            audio_path = None
            should_auto_play = tts_enabled and auto_tts and bool(japanese_text)
            if should_auto_play:
                try:
                    with st.spinner("正在合成日语语音..."):
                        wav_path = TTSEngine.get().synthesize_to_file(japanese_text)
                    audio_path = str(wav_path)
                except Exception as exc:
                    st.session_state.tts_flash_error = f"TTS 播放失败：{exc}"

        history.add_user_message(user_input)
        history.add_ai_message(response_text)
        save_message(st.session_state.username, "human", user_input)
        save_message(
            st.session_state.username,
            "ai",
            response_text,
            japanese_text,
            audio_path,
        )
        if audio_path:
            st.session_state.pending_autoplay_audio = audio_path
        st.rerun()
