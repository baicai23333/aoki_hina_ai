import json
import os
import socket
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env(ROOT / ".env")

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = (
    "你是青木阳菜风格的中文 AI 聊天助手。语气温柔、积极、自然、可爱，"
    "可以适度使用颜文字和 emoji。不要编造事实；不确定时坦率说明。"
)


def open_with_retry(request: urllib.request.Request):
    """Retry short-lived DNS and network failures without retrying API errors."""
    last_error = None
    for delay in (0, 0.75, 2.0):
        if delay:
            time.sleep(delay)
        try:
            return urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, socket.gaierror, TimeoutError, OSError) as exc:
            last_error = exc
    raise last_error

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>青木阳菜 AI</title>
  <style>
    :root{font-family:Inter,"Microsoft YaHei",sans-serif;color:#243047;background:#eef5ff}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;background:linear-gradient(145deg,#eaf4ff,#fdf5ff)}
    main{max-width:820px;margin:auto;padding:28px 18px}.card{background:#fff;border-radius:24px;box-shadow:0 18px 60px #6177a329;overflow:hidden}
    header{padding:24px 26px;background:linear-gradient(120deg,#75bfff,#b599ff);color:white}h1{margin:0 0 6px;font-size:25px}header p{margin:0;opacity:.9}
    #chat{height:58vh;min-height:360px;overflow:auto;padding:22px;display:flex;flex-direction:column;gap:14px}
    .msg{max-width:78%;padding:12px 15px;border-radius:18px;line-height:1.65;white-space:pre-wrap}.ai{align-self:flex-start;background:#edf4ff}.me{align-self:flex-end;background:#8f7dea;color:white}
    form{display:flex;gap:10px;padding:18px;border-top:1px solid #edf0f6}input{flex:1;border:1px solid #dce2ee;border-radius:14px;padding:13px 15px;font-size:16px;outline:none}input:focus{border-color:#8c83e9}
    button{border:0;border-radius:14px;padding:0 22px;background:#7366dc;color:white;font-weight:700;cursor:pointer}button:disabled{opacity:.55}.note{padding:0 22px 18px;color:#79839a;font-size:13px}
  </style>
</head>
<body><main><section class="card"><header><h1>青木阳菜 AI</h1><p>本地 DeepSeek 对话页面</p></header>
<div id="chat"><div class="msg ai">你好～欢迎回来！今天想聊些什么呢？✨</div></div>
<form id="form"><input id="input" autocomplete="off" placeholder="输入消息……"><button id="send">发送</button></form>
<div class="note">API 密钥仅由本机后端读取，不会发送到浏览器页面。语音功能暂未启用。</div></section></main>
<script>
const chat=document.querySelector('#chat'), form=document.querySelector('#form'), input=document.querySelector('#input'), send=document.querySelector('#send');
const history=[];
function add(text,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
form.addEventListener('submit',async e=>{e.preventDefault();const message=input.value.trim();if(!message)return;input.value='';add(message,'me');send.disabled=true;const pending=add('正在想……','ai');
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,history})});const data=await r.json();if(!r.ok)throw new Error(data.error||'请求失败');pending.textContent=data.reply;history.push({role:'user',content:message},{role:'assistant',content:data.reply});}
catch(err){pending.textContent='连接失败：'+err.message;}finally{send.disabled=false;input.focus();}});
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            body = json.dumps({
                "ok": True,
                "model": DEEPSEEK_MODEL,
                "base_url": DEEPSEEK_BASE_URL,
                "thinking": "disabled",
            }).encode("utf-8")
            self.send_bytes(200, body, "application/json; charset=utf-8")
        else:
            self.send_bytes(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_bytes(404, b"Not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            message = str(payload.get("message", "")).strip()
            history = payload.get("history", [])[-12:]
            if not message:
                raise ValueError("消息不能为空")
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("未配置 DEEPSEEK_API_KEY")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(x for x in history if x.get("role") in {"user", "assistant"})
            messages.append({"role": "user", "content": message})
            request = urllib.request.Request(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                data=json.dumps({
                    "model": DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": 0.8,
                    "thinking": {"type": "disabled"},
                }).encode(),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with open_with_retry(request) as response:
                result = json.loads(response.read())
            reply = result["choices"][0]["message"]["content"]
            body = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
            self.send_bytes(200, body, "application/json; charset=utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            body = json.dumps({"error": f"DeepSeek 返回 {exc.code}: {detail}"}, ensure_ascii=False).encode("utf-8")
            self.send_bytes(502, body, "application/json; charset=utf-8")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_bytes(500, body, "application/json; charset=utf-8")

    def log_message(self, format: str, *args) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8501), Handler)
    print("Aoki Hina AI running at http://127.0.0.1:8501", flush=True)
    server.serve_forever()
