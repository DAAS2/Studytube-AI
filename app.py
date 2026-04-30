import os
import json
import time
import uuid
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from pytubefix import YouTube

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

try:
    for key, val in st.secrets.items():
        os.environ.setdefault(key, val)
except Exception:
    pass

# ── Groq key rotation ─────────────────────────────────────────────────────────
def load_groq_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys and os.getenv("GROQ_API_KEY"):
        keys.append(os.getenv("GROQ_API_KEY"))
    return keys

GROQ_KEYS = load_groq_keys()

def get_next_groq_key() -> str:
    if not GROQ_KEYS:
        return None
    if "key_index" not in st.session_state:
        st.session_state.key_index = 0
    key = GROQ_KEYS[st.session_state.key_index % len(GROQ_KEYS)]
    st.session_state.key_index += 1
    return key

# ── Page config (must come before any st calls) ───────────────────────────────
st.set_page_config(page_title="StudyTube AI", page_icon="🎓", layout="wide")

# ── Auth setup ────────────────────────────────────────────────────────────────
CONFIG_FILE = "users.yaml"

# Create a default config file if it doesn't exist yet
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "studytube_auth",
            "key": "studytube_secret_key_change_me",
            "expiry_days": 30,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(default_config, f)

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0533 50%, #0d1b2a 100%);
    border: 1px solid #2a1a4a;
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.main-header h1 { font-size: 2rem; margin: 0; color: #e8e0ff; letter-spacing: -0.5px; }
.main-header p  { color: #8877aa; margin: 0.25rem 0 0; font-size: 0.9rem; }

.login-wrap {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.5rem;
    background: #0f0f1e;
    border: 1px solid #2a1a4a;
    border-radius: 16px;
}
.login-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    color: #e8e0ff;
    margin-bottom: 0.3rem;
}
.login-sub { color: #7766aa; font-size: 0.9rem; margin-bottom: 1.5rem; }

.video-card {
    background: #0f0f1e;
    border: 1px solid #222240;
    border-radius: 12px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
}
.video-card img   { width: 100%; border-radius: 8px; margin-bottom: 0.5rem; }
.video-title      { font-size: 0.82rem; font-weight: 600; color: #ddd8ff; line-height: 1.3; }
.video-author     { font-size: 0.72rem; color: #7766aa; margin-top: 0.2rem; }
.video-duration   { font-size: 0.7rem;  color: #9966ff; margin-top: 0.2rem; font-weight: 500; }

.stat-box {
    background: #0f0f1e;
    border: 1px solid #222240;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.stat-number { font-size: 1.8rem; font-weight: 800; color: #9966ff; font-family: 'Syne', sans-serif; }
.stat-label  { font-size: 0.72rem; color: #7766aa; margin-top: 0.1rem; }

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6655aa;
    margin: 1.2rem 0 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a1a30;
}

.msg-time { font-size: 0.62rem; color: #554477; margin-top: 0.25rem; }

.tag {
    display: inline-block;
    background: #1a1030;
    color: #9988cc;
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.7rem;
    margin: 0.15rem;
    border: 1px solid #2d1f50;
}

.user-badge {
    background: #1a1030;
    border: 1px solid #2d1f50;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #c0aeff;
    display: inline-block;
}

.stChatMessage { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Per-user paths ────────────────────────────────────────────────────────────

def user_dir(username: str) -> str:
    """Each user gets their own isolated folder."""
    path = os.path.join("video_library", username)
    os.makedirs(path, exist_ok=True)
    return path

def user_chats_file(username: str) -> str:
    return os.path.join(user_dir(username), "chats.json")

# ADD these instead:
SHARED_DIR = "video_library/_shared"
os.makedirs(SHARED_DIR, exist_ok=True)

def user_index_file(username: str = None) -> str:
    return os.path.join(SHARED_DIR, "index.json")

def user_faiss_path(username: str, video_id: str) -> str:
    path = os.path.join(SHARED_DIR, f"faiss_{video_id}")
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return url

def format_duration(seconds: int) -> str:
    if not seconds:
        return "?"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"

def ts() -> str:
    return time.strftime("%H:%M")

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Library persistence ───────────────────────────────────────────────────────

def load_library(username: str) -> dict:
    path = user_index_file()          # no longer user-specific
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_library(username: str, lib: dict):
    with open(user_index_file(), "w") as f:  # no longer user-specific
        json.dump(lib, f, indent=2)


# ── Chat persistence ──────────────────────────────────────────────────────────

def load_chats(username: str) -> dict:
    path = user_chats_file(username)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_chats(username: str, chats: dict):
    with open(user_chats_file(username), "w") as f:
        json.dump(chats, f, indent=2)

def new_chat(username: str, name: str = "") -> str:
    cid   = str(uuid.uuid4())[:8]
    label = name.strip() or f"Chat {time.strftime('%b %d %H:%M')}"
    st.session_state.chats[cid] = {
        "name": label,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "messages": []
    }
    save_chats(username, st.session_state.chats)
    return cid

def current_chat() -> dict | None:
    cid = st.session_state.active_chat_id
    return st.session_state.chats.get(cid) if cid else None

def add_message(username: str, role: str, content: str):
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.chats:
        st.session_state.chats[cid]["messages"].append({
            "role": role, "content": content, "time": ts()
        })
        save_chats(username, st.session_state.chats)

def chat_to_langchain(messages: list) -> list:
    result = []
    for m in messages:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result

def export_chat_markdown(chat: dict) -> str:
    lines = [f"# {chat['name']}", f"*Created: {chat['created']}*", ""]
    for m in chat["messages"]:
        prefix = "**You**" if m["role"] == "user" else "**AI Tutor**"
        lines.append(f"{prefix} _{m['time']}_")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines)


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def process_and_save_video(username: str, url: str) -> dict | None:
    video_id = extract_video_id(url)
    try:
        yt        = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        title     = yt.title or "Untitled"
        author    = yt.author or "Unknown"
        thumbnail = yt.thumbnail_url or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        duration  = yt.length or 0
    except Exception:
        title = video_id; author = "Unknown"
        thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"; duration = 0

    try:
        # Check if we have proxy credentials configured
        proxy_user = os.getenv("WEBSHARE_USER")
        proxy_pass = os.getenv("WEBSHARE_PASS")
        
        if proxy_user and proxy_pass:
            proxy_config = WebshareProxyConfig(
                proxy_username=proxy_user, 
                proxy_password=proxy_pass
            )
            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            ytt_api = YouTubeTranscriptApi()
            
        fetched = ytt_api.fetch(video_id)
        transcript_text = " ".join([s.text for s in fetched])
    except Exception as e:
        st.error(f"Could not fetch transcript: {e}")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents([
        Document(page_content=transcript_text, metadata={"source": video_id, "title": title})
    ])
    vs = FAISS.from_documents(docs, get_embeddings_model())
    vs.save_local(user_faiss_path(username, video_id))

    return {
        "video_id": video_id, "url": url, "title": title,
        "author": author, "thumbnail": thumbnail,
        "duration": duration, "chunks": len(docs),
        "added_at": time.strftime("%Y-%m-%d %H:%M"),
    }

def load_video_store(username: str, video_id: str) -> FAISS | None:
    path = user_faiss_path(username, video_id)
    if os.path.exists(path):
        return FAISS.load_local(path, get_embeddings_model(), allow_dangerous_deserialization=True)
    return None

def merge_stores(username: str, video_ids: list[str]) -> FAISS | None:
    stores = [s for s in (load_video_store(username, v) for v in video_ids) if s]
    if not stores:
        return None
    base = stores[0]
    for s in stores[1:]:
        base.merge_from(s)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("authentication_status"):
    st.markdown("""
    <div style="max-width:420px;margin:3rem auto;text-align:center;">
        <div style="font-family:Syne,sans-serif;font-size:2.5rem;color:#e8e0ff;">🎓</div>
        <div style="font-family:Syne,sans-serif;font-size:1.8rem;color:#e8e0ff;font-weight:700;">StudyTube AI</div>
        <div style="color:#7766aa;font-size:0.9rem;margin-bottom:1.5rem;">Your personal AI study assistant</div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        auth_tab, reg_tab = st.tabs(["🔐 Login", "📝 Register"])

        with auth_tab:
            authenticator.login(location="main")
            if st.session_state.get("authentication_status") is False:
                st.error("❌ Incorrect username or password.")
            elif st.session_state.get("authentication_status") is None:
                st.info("👆 Enter your credentials to continue.")

        with reg_tab:
            st.markdown("#### Create an account")
            reg_name     = st.text_input("Display name",       placeholder="Alice Smith",        key="reg_name")
            reg_email    = st.text_input("Email",              placeholder="alice@example.com",  key="reg_email")
            reg_username = st.text_input("Username",           placeholder="alice",              key="reg_username")
            reg_password = st.text_input("Password",           type="password",                  key="reg_password")
            reg_confirm  = st.text_input("Confirm password",   type="password",                  key="reg_confirm")

            if st.button("Create account", use_container_width=True, key="reg_btn"):
                # ── Validation ──────────────────────────────────────────
                if not all([reg_name, reg_email, reg_username, reg_password, reg_confirm]):
                    st.error("Please fill in all fields.")
                elif reg_password != reg_confirm:
                    st.error("Passwords don't match.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_username in config["credentials"]["usernames"]:
                    st.error(f"Username '{reg_username}' is already taken.")
                else:
                    import bcrypt
                    hashed = bcrypt.hashpw(reg_password.encode(), bcrypt.gensalt()).decode()
                    config["credentials"]["usernames"][reg_username] = {
                        "name":                  reg_name,
                        "email":                 reg_email,
                        "password":              hashed,
                        "failed_login_attempts": 0,
                        "logged_in":             False,
                    }
                    with open(CONFIG_FILE, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    st.success(f"✅ Account created! Switch to the Login tab and sign in as **{reg_username}**.")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATED — get current user
# ══════════════════════════════════════════════════════════════════════════════

username = st.session_state.get("username", "")
name     = st.session_state.get("name", username)

# Init per-user session state (reset if user changes)
if st.session_state.get("_loaded_user") != username:
    st.session_state._loaded_user    = username
    st.session_state.chats           = load_chats(username)
    st.session_state.active_chat_id  = None
    st.session_state.selected_videos = []
    st.session_state.active_store    = None
    st.session_state.watching_video  = None
    st.session_state.key_index       = 0
    st.session_state.chat_search     = ""

# Always reload the shared library (outside the if block)
st.session_state.library = load_library(username)

# ── Header ────────────────────────────────────────────────────────────────────

hcol, ucol = st.columns([5, 1])
with hcol:
    st.markdown(f"""
    <div class="main-header">
        <h1>🎓 StudyTube AI</h1>
        <p>Your personal AI study assistant — build a video library, run multiple chats, never lose a note.</p>
    </div>
    """, unsafe_allow_html=True)
with ucol:
    st.write("")
    st.write("")
    st.markdown(f'<div class="user-badge">👤 {name}</div>', unsafe_allow_html=True)
    st.write("")
    authenticator.logout(button_name="Log out", location="main", key="logout_btn")

left, right = st.columns([1, 2.2], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Library
# ══════════════════════════════════════════════════════════════════════════════
with left:

    total_videos = len(st.session_state.library)
    total_chunks = sum(v.get("chunks", 0) for v in st.session_state.library.values())
    total_chats  = len(st.session_state.chats)

    c1, c2, c3 = st.columns(3)
    for col, num, label in [(c1, total_videos, "Videos"), (c2, total_chunks, "Chunks"), (c3, total_chats, "Chats")]:
        with col:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Add Video</div>', unsafe_allow_html=True)
    url_input = st.text_input("url", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")

    if st.button("⬇ Extract & Save", use_container_width=True):
        if not GROQ_KEYS:
            st.error("Add GROQ_API_KEY_1 to your .env file.")
        elif not url_input.strip():
            st.warning("Paste a YouTube URL first.")
        else:
            vid_id = extract_video_id(url_input)
            if vid_id in st.session_state.library:
                st.info("Already in your library!")
            else:
                with st.spinner("Fetching transcript & building index…"):
                    record = process_and_save_video(username, url_input)
                    if record:
                        st.session_state.library[vid_id] = record
                        save_library(username, st.session_state.library)
                        st.success(f"✅ {record['title']}")
                        st.rerun()

    st.markdown('<div class="section-header">Your Library</div>', unsafe_allow_html=True)

    if not st.session_state.library:
        st.caption("No videos yet. Add one above!")
    else:
        for vid_id, meta in st.session_state.library.items():
            st.markdown(f"""<div class="video-card">
                <img src="{meta['thumbnail']}" alt="thumb"/>
                <div class="video-title">{meta['title']}</div>
                <div class="video-author">📺 {meta['author']}</div>
                <div class="video-duration">⏱ {format_duration(meta['duration'])} · {meta['chunks']} chunks · {meta['added_at']}</div>
            </div>""", unsafe_allow_html=True)

            ca, cb, cc = st.columns(3)
            with ca:
                if st.button("▶ Watch", key=f"w_{vid_id}", use_container_width=True):
                    st.session_state.watching_video = vid_id
                    st.rerun()
            with cb:
                if st.button("💬 Chat", key=f"c_{vid_id}", use_container_width=True):
                    with st.spinner("Loading…"):
                        vs = merge_stores(username, [vid_id])
                        if vs:
                            st.session_state.active_store    = vs
                            st.session_state.selected_videos = [vid_id]
                            cid = new_chat(username, meta['title'][:30])
                            st.session_state.active_chat_id  = cid
                            st.rerun()
            with cc:
                if st.button("🗑", key=f"d_{vid_id}", use_container_width=True):
                    del st.session_state.library[vid_id]
                    save_library(username, st.session_state.library)
                    if vid_id in st.session_state.selected_videos:
                        st.session_state.selected_videos.remove(vid_id)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with right:

    # Video player
    if st.session_state.watching_video:
        vid_id = st.session_state.watching_video
        meta   = st.session_state.library.get(vid_id)
        if meta:
            hc, xc = st.columns([5, 1])
            with hc:
                st.markdown('<div class="section-header">Now Watching</div>', unsafe_allow_html=True)
                st.caption(f"**{meta['title']}** — {meta['author']}")
            with xc:
                st.write("")
                if st.button("✕ Close", use_container_width=True):
                    st.session_state.watching_video = None
                    st.rerun()
            st.video(meta["url"])
            st.divider()

    tab_chat, tab_select = st.tabs(["💬 Chat", "🎯 Select Videos & Load AI"])

    # ── SELECT VIDEOS TAB ────────────────────────────────────────────────────
    with tab_select:
        if not st.session_state.library:
            st.info("👈 Add videos in the left panel to get started.")
        else:
            all_ids    = list(st.session_state.library.keys())
            select_all = st.checkbox(
                "Select all",
                value=len(st.session_state.selected_videos) > 0 and
                      set(st.session_state.selected_videos) == set(all_ids)
            )

            if select_all:
                st.session_state.selected_videos = all_ids
            else:
                selected = []
                cols = st.columns(2)
                for i, (vid_id, meta) in enumerate(st.session_state.library.items()):
                    with cols[i % 2]:
                        if st.checkbox(
                            meta["title"][:42] + ("…" if len(meta["title"]) > 42 else ""),
                            value=vid_id in st.session_state.selected_videos,
                            key=f"chk_{vid_id}"
                        ):
                            selected.append(vid_id)
                st.session_state.selected_videos = selected

            if st.session_state.selected_videos:
                tags = "".join(
                    f'<span class="tag">📹 {st.session_state.library[v]["title"][:28]}</span>'
                    for v in st.session_state.selected_videos
                )
                st.markdown(tags, unsafe_allow_html=True)
                st.write("")

                new_name = st.text_input("Chat name (optional)", placeholder="e.g. Week 3 Lecture Notes")
                if st.button("🔗 Load into AI & Start New Chat", use_container_width=True):
                    with st.spinner(f"Merging {len(st.session_state.selected_videos)} video(s)…"):
                        merged = merge_stores(username, st.session_state.selected_videos)
                        if merged:
                            st.session_state.active_store = merged
                            cid = new_chat(username, new_name)
                            st.session_state.active_chat_id = cid
                            st.success("✅ Ready! Switch to the Chat tab.")
                            st.rerun()

    # ── CHAT TAB ─────────────────────────────────────────────────────────────
    with tab_chat:

        cm1, cm2, cm3 = st.columns([3, 1, 1])
        with cm1:
            chat_ids = list(st.session_state.chats.keys())
            if chat_ids:
                if st.session_state.active_chat_id not in chat_ids:
                    st.session_state.active_chat_id = chat_ids[-1]
                chosen = st.selectbox(
                    "chat_select",
                    options=chat_ids,
                    format_func=lambda c: st.session_state.chats[c]["name"],
                    index=chat_ids.index(st.session_state.active_chat_id),
                    label_visibility="collapsed"
                )
                if chosen != st.session_state.active_chat_id:
                    st.session_state.active_chat_id = chosen
                    st.rerun()
            else:
                st.caption("No chats yet — load videos first.")

        with cm2:
            if st.button("➕ New", use_container_width=True):
                if st.session_state.active_store:
                    cid = new_chat(username)
                    st.session_state.active_chat_id = cid
                    st.rerun()
                else:
                    st.warning("Load videos first (Select Videos tab).")

        with cm3:
            if st.session_state.active_chat_id and st.session_state.active_chat_id in st.session_state.chats:
                if st.button("🗑 Del", use_container_width=True):
                    del st.session_state.chats[st.session_state.active_chat_id]
                    save_chats(username, st.session_state.chats)
                    remaining = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = remaining[-1] if remaining else None
                    st.rerun()

        chat = current_chat()
        if chat:
            msg_count = len(chat["messages"])
            with st.expander(f"✏️ Rename  ·  {msg_count} message{'s' if msg_count != 1 else ''}  ·  {chat['created']}"):
                new_name_val = st.text_input("New name", value=chat["name"], key="rename_inp")
                if st.button("Save name", key="save_rename"):
                    st.session_state.chats[st.session_state.active_chat_id]["name"] = new_name_val
                    save_chats(username, st.session_state.chats)
                    st.rerun()

        st.divider()

        if not chat:
            st.info("Select or create a chat above.")
        else:
            # Search bar
            sc, xc = st.columns([5, 1])
            with sc:
                search_q = st.text_input(
                    "search", placeholder="🔍  Search messages…",
                    value=st.session_state.chat_search,
                    label_visibility="collapsed"
                )
                st.session_state.chat_search = search_q
            with xc:
                if st.button("✕", key="clr_search"):
                    st.session_state.chat_search = ""
                    st.rerun()

            if st.session_state.active_store is None:
                st.warning("⚠️ This chat has no active video context. Go to **Select Videos** tab and reload to ask new questions. You can still read past messages below.")
            
            # Quick prompts
            quick = [
                ("📝 Notes",    "Give me comprehensive, well-structured notes from all the content, with headings and bullet points."),
                ("📋 Summary",  "Summarize the key points covered in the video(s) in 3–5 clear paragraphs."),
                ("❓ Concepts", "What are the most important concepts I should understand? List and explain each one."),
                ("🧪 Quiz",     "Create a 5-question quiz based on this content, with answers at the end."),
                ("🗺 Mind Map", "Create a structured mind map outline showing the main topics and subtopics covered."),
            ]
            qcols = st.columns(len(quick))
            for i, (lbl, prm) in enumerate(quick):
                with qcols[i]:
                    if st.button(lbl, use_container_width=True, key=f"qp_{i}"):
                        st.session_state._inject_prompt = prm

            # Scrollable messages
            messages = chat["messages"]
            if st.session_state.chat_search.strip():
                q        = st.session_state.chat_search.lower()
                messages = [m for m in messages if q in m["content"].lower()]
                st.caption(f"🔍 {len(messages)} matching message(s)")

            chat_container = st.container(height=460)
            with chat_container:
                if not messages:
                    st.markdown(
                        '<div style="text-align:center;color:#554477;padding:3rem 0;font-size:0.9rem;">'
                        '✨ No messages yet — ask something below!</div>',
                        unsafe_allow_html=True
                    )
                else:
                    for m in messages:
                        with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🤖"):
                            st.markdown(m["content"])
                            st.markdown(f'<div class="msg-time">🕐 {m["time"]}</div>', unsafe_allow_html=True)

            # Export + Clear
            ec, cc2 = st.columns(2)
            with ec:
                if chat["messages"]:
                    st.download_button(
                        "⬇ Export as Markdown",
                        data=export_chat_markdown(chat),
                        file_name=f"{chat['name'].replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            with cc2:
                if chat["messages"]:
                    if st.button("🗑 Clear messages", use_container_width=True):
                        st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
                        save_chats(username, st.session_state.chats)
                        st.rerun()

            # Chat input
            injected   = st.session_state.pop("_inject_prompt", None)
            user_query = st.chat_input("Ask anything about your videos…") or injected

            if user_query:
                add_message(username, "user", user_query)

                history_snapshot = chat_to_langchain(
                    st.session_state.chats[st.session_state.active_chat_id]["messages"][:-1]
                )

                llm = ChatGroq(
                    api_key=get_next_groq_key(),
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    streaming=True,
                )

                system_prompt = (
                    "You are an expert study assistant and note-taker for a university student. "
                    "You have access to transcripts from YouTube educational videos. "
                    "Always use markdown formatting: headings, bullet points, bold key terms. "
                    "When asked for notes, produce detailed structured notes. "
                    "When asked questions, answer from the transcript first. "
                    "If the answer isn't in the transcripts, use your knowledge but say so clearly.\n\n"
                    "Transcript Context:\n{context}"
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                retriever = st.session_state.active_store.as_retriever(search_kwargs={"k": 6})

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "input": RunnablePassthrough(),
                        "chat_history": lambda _: history_snapshot,
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        answer = st.write_stream(rag_chain.stream(user_query))
                        st.markdown(f'<div class="msg-time">🕐 {ts()}</div>', unsafe_allow_html=True)

                add_message(username, "assistant", answer)
                st.rerun()