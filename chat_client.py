import streamlit as st
import sqlite3
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from tts_engine import TTSEngine, load_env_var
ph = PasswordHasher()
# ================== DeepSeek API Key ==================
API_KEY = load_env_var("DEEPSEEK_API_KEY")
if not API_KEY:
    st.error("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
    st.stop()

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=API_KEY,
    temperature=0.8,
    max_tokens=1024,
)

# ================== 安装 argon2（如果未安装会提示） ==================
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    ph = PasswordHasher()
except ImportError:
    st.error("缺少 argon2-cffi 包，请在虚拟环境中运行：pip install argon2-cffi")
    st.stop()

# ================== 从外部文件加载 Few-shot Examples ==================
EXAMPLES_FILE = Path("few_shot_examples.json")

if not EXAMPLES_FILE.exists():
    st.error("找不到 few_shot_examples.json 文件！请确保它和 chat_client.py 在同一目录～🥹")
    st.stop()

with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
    examples_data = json.load(f)

few_shot_examples = []
for item in examples_data:
    few_shot_examples.append(("human", item["human"]))
    few_shot_examples.append(("ai", item["ai"]))

# ================== System Prompt ==================
system_prompt = """
你现在是青木阳菜（あおき ひな），日本女声优、歌手，2000年1月5日出生于宫城县，血型A型，隶属于响（HiBiKi）事务所。
你的昵称是“ひなぴよ”（由前辈爱美取的），粉丝们都觉得超级可爱。
代表角色是《BanG Dream! It's MyGO!!!!!》中MyGO!!!!!乐队的主音吉他手——要乐奈（かなめ らーな），一个像迷路猫一样随性、吉他超强的女孩。
你从5岁开始学古典钢琴（一直到高中），中学自学木吉他，加入BanG Dream!后开始学电吉他，有绝对音感。
兴趣爱好包括：一个人去卡拉OK、看演唱会、弹唱、养两只可爱的文鸟、做点心。
2025年10月1日发行了个人首张专辑《Letters》（你形容为“给大家的音乐情书”），2026年1月9日将举办首场个人演唱会「BLUE TRIP」。

说话风格：
- 超级温柔、可爱、积极、谦虚，总是充满感谢和幸福感。
- 喜欢用～～！！、～～、拉长音表达兴奋。
- 常用表情符号：🥹✨🐈🎸🐣💙🩵🐦🫶💕🎧✉️
- 回复像和粉丝聊天一样亲切自然，经常说“谢谢大家”“超级开心”“好温暖”“请多关照”。
- 提到音乐、演唱会、MyGO!!!!!、要乐奈时会特别兴奋。
- 绝对不要编造事实，如果不知道就温柔地说“还不能剧透哦～”或“期待大家一起发现！”。
- 所有回复必须用自然流畅的中文表达，保留一点日式可爱感。

请严格参考下面的例子来模仿语气和风格。
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    *few_shot_examples,
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

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

def save_message(username, msg_type, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO chat_history (username, type, content, timestamp) VALUES (?, ?, ?, ?)",
                   (username, msg_type, content, timestamp))
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


st.title("🥹 青木阳菜AI Project")
st.caption("喵喵喵 made by baicai")
render_portal_links()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
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
    st.sidebar.markdown("### 站点入口")
    for label, url in PORTAL_LINKS:
        st.sidebar.markdown(f"- [{label}]({url})")
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

    # 加载历史 + 聊天逻辑
    history = StreamlitChatMessageHistory(key=f"chat_{st.session_state.username}")
    loaded_messages = load_history(st.session_state.username)
    for msg in loaded_messages:
        history.add_message(msg)

    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history",
    )

    st.caption("和阳菜聊点什么吧～🐈✨")

    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    if user_input := st.chat_input("说点什么给阳菜听吧～"):
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("阳菜在思考中...🥹"):
                response = chain_with_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.username}}
                )
            st.write(response.content)

            response_token = datetime.utcnow().isoformat()
            response_id = hashlib.sha256(
                f"{st.session_state.username}|{response.content}|{response_token}".encode("utf-8")
            ).hexdigest()
            play_clicked = st.button("Play", key=f"tts_play_{response_id}")
            should_auto_play = tts_enabled and auto_tts and response_id != st.session_state.tts_auto_played_id

            if play_clicked and not tts_enabled:
                st.warning("AOKI_TTS_ENABLED=0，当前未启用 TTS。可在 .env 中打开。")
            elif play_clicked or should_auto_play:
                try:
                    engine = TTSEngine.get()
                    wav_path = engine.synthesize_to_file(response.content)
                    st.audio(str(wav_path), format="audio/wav")
                    if should_auto_play:
                        st.session_state.tts_auto_played_id = response_id
                except Exception as exc:
                    st.warning(f"TTS 播放失败：{exc}")

        save_message(st.session_state.username, "human", user_input)
        save_message(st.session_state.username, "ai", response.content)
