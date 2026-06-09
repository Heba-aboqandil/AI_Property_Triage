"""
app.py
------
Streamlit WebUI for the Property Triage Platform.

Storage: SQLite (chat_history.db) — see db.py
Chat:    Groq Llama 3.1 with Perplexity Sonar as a tool — see chat_client.py
"""
# Load API keys from .env file FIRST, before any other imports that
# might need them (chat_client, pinecone_client, sonar_client all read
# os.environ at import time). `find_dotenv()` walks UP the directory
# tree, so the .env at the project root is picked up even when
# Streamlit is launched from the Layer1_UserInterface folder.
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import streamlit as st
import requests
import base64
import re
from pathlib import Path
from datetime import datetime
import rag_client
import db
from chat_client import ChatClient
import router
import sonar_client
import pinecone_client

# ============================================================
# CONFIG
# ============================================================
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/property-triage"
IMAGE_ANALYSER_UPLOAD_URL = "http://54.84.168.9:8002/analyse/upload"
REQUEST_TIMEOUT = 900
HERO_IMAGE_PATH = Path("assets/hero.jpg")
chat_client = ChatClient()

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="AI Property Triage",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
if "current_session_id" not in st.session_state:
    sessions = db.list_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0]["id"]
    else:
        st.session_state.current_session_id = db.create_session("Chat 1")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def generate_chat_title(first_message: str) -> str:
    title = first_message.strip().split("\n")[0]
    if len(title) > 30:
        title = title[:27] + "..."
    return title or "New Chat"


def build_listing_context() -> str | None:
    """Pull the most recent listing from SQLite and format it as context."""
    last = db.get_last_listing()
    if not last:
        return None
    payload = last.get("payload") or {}
    report = last.get("report") or {}

    lines = []
    desc = payload.get("description", "").strip()
    if desc:
        lines.append(f"Description (as written by the agent):\n{desc}")

    if report:
        lines.append("\nStructured report from the triage pipeline:")
        for key in [
            "property_type", "routing_decision", "location",
            "price_ils", "num_rooms", "size_sqm",
            "key_features", "rag_insight", "enrichment_notes",
        ]:
            v = report.get(key)
            if v not in (None, "", []):
                lines.append(f"- {key}: {v}")
    return "\n".join(lines) if lines else None


# ============================================================
# CSS + HERO
# ============================================================
def get_hero_image_data_url():
    if not HERO_IMAGE_PATH.exists():
        return ""
    with open(HERO_IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = HERO_IMAGE_PATH.suffix.lower().strip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"url('data:image/{ext};base64,{img_b64}')"


css_path = Path("style.css")
css_content = ""
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()

hero_url = get_hero_image_data_url()
hero_var = f":root {{ --hero-bg: {hero_url}; }}" if hero_url else ""

dark_mode_css = ""
if st.session_state.dark_mode:
    dark_mode_css = """
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
        --bg-base: #0a0e1a;
        --surface: rgba(20, 25, 40, 0.92);
        --surface-strong: #161b2e;
        --border: rgba(255, 255, 255, 0.12);
        --text-primary: #f8f6f1;
        --text-secondary: #c5cbd9;
        --text-muted: #b7bfd3;
    }
    /* Chat input — fix every nested element so no white patches remain */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    [data-baseweb="textarea"],
    [data-baseweb="base-input"],
    [data-baseweb="textarea"] > div {
        background: #161b2e !important;
        background-color: #161b2e !important;
        color: #f8f6f1 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #b7bfd3 !important;
        opacity: 0.7 !important;
        -webkit-text-fill-color: #b7bfd3 !important;
    }
    /* Regular text inputs / textareas */
    .stTextInput input,
    .stTextArea textarea {
        background: #161b2e !important;
        color: #f8f6f1 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stAppViewContainer"]::after {
        background: linear-gradient(180deg,
            rgba(10, 14, 26, 0.7) 0%,
            rgba(10, 14, 26, 0.9) 100%) !important;
    }
    /* Disable expensive transitions in dark mode so the swap feels instant */
    * { transition: none !important; animation-duration: 0s !important; }
    """

no_scroll_css = """
    html, body { overflow-y: auto !important; height: auto !important; }
    [data-testid="stAppViewContainer"] { overflow: auto !important; }
    .block-container { overflow: visible !important; padding-bottom: 4rem !important; }
    .hero-title { font-size: clamp(1.4rem, 2.5vw, 2rem) !important; margin: 0 0 0.1rem 0 !important; }
    .hero-subtitle { font-size: 0.9rem !important; margin-bottom: 0.3rem !important; }
    .hero-badge { margin-bottom: 0.2rem !important; padding: 0.25rem 0.75rem !important; }
"""

st.markdown(
    f"<style>{hero_var}{css_content}{dark_mode_css}{no_scroll_css}</style>",
    unsafe_allow_html=True,
)


# ============================================================
# STATUS CHECKS
# ============================================================
# Cached AGGRESSIVELY (5 minutes) so re-runs don't re-hit Groq/Perplexity/
# n8n/Pinecone on every button click or theme change. The status badges
# don't need to update every minute — once every 5 minutes is plenty.
# This is the main fix for sluggish "New Chat" / theme switching.
@st.cache_data(ttl=300, show_spinner=False)
def check_groq():
    return chat_client.health_check()


@st.cache_data(ttl=300, show_spinner=False)
def check_perplexity():
    return ChatClient.perplexity_health_check()


@st.cache_data(ttl=300, show_spinner=False)
def check_n8n():
    try:
        base_url = N8N_WEBHOOK_URL.rsplit("/webhook", 1)[0]
        r = requests.get(base_url, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def check_pinecone():
    try:
        return pinecone_client.health_check()
    except Exception:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def get_pinecone_count():
    try:
        return pinecone_client.count_listings()
    except Exception:
        return 0


# ============================================================
# REPORT RENDERER
# ============================================================
def fmt_price(p):
    if p is None:
        return "—"
    try:
        return f"₪{int(p):,}"
    except Exception:
        return str(p)


def fmt_num(n, suffix=""):
    if n is None:
        return "—"
    return f"{n}{suffix}"


def fmt_pct(c):
    if c is None:
        return "—"
    try:
        return f"{int(float(c) * 100)}%"
    except Exception:
        return str(c)


def render_property_report(result):
    channel = result.get("channel", "unknown")
    success = result.get("success", False)
    human_review = result.get("human_review_required", False)
    rejected = result.get("rejected", False)
    report = result.get("report") or {}

    if rejected:
        st.error(
            f"❌  **Submission rejected at input guardrails.** "
            f"Reason: _{result.get('reason', 'unspecified')}_"
        )
        return
    if human_review:
        st.warning(
            f"⚠️  **Flagged for human review.** "
            f"Reason: _{result.get('flag_reason', 'unspecified')}_"
        )
    elif success:
        ch_label = (channel or "unknown").title()
        st.success(
            f"✅  **Listing submitted successfully** — routed to **{ch_label}** team."
        )

    if not report:
        st.info("No structured report returned.")
        return

    location = report.get("location") or "Unknown location"
    prop_type = (report.get("property_type") or "Property").title()
    num_rooms = report.get("num_rooms")
    title_bits = []
    if num_rooms:
        title_bits.append(f"{num_rooms}-Room")
    title_bits.append(prop_type)
    title = " ".join(title_bits)

    routing = report.get("routing_decision", "—")
    badge_color = {"residential": "#0C447C", "commercial": "#854F0B"}.get(routing, "#5F5E5A")
    badge_bg = {"residential": "#E6F1FB", "commercial": "#FAEEDA"}.get(routing, "#F1EFE8")

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
                    margin: 1.25rem 0 1.25rem;">
            <div>
                <h3 style="margin:0 0 4px; font-weight:600;">Property analysis report</h3>
                <p style="margin:0; color:#6b7385; font-size:0.95rem;">
                    {title} · {location}
                </p>
            </div>
            <span style="background:{badge_bg}; color:{badge_color}; font-size:0.75rem;
                         padding:5px 14px; border-radius:8px; font-weight:600;
                         letter-spacing:0.05em; text-transform:uppercase;">
                {routing}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    metric_style = "background:#f5f3ee; padding:1rem; border-radius:10px; text-align:left;"
    label_style = "font-size:0.78rem; color:#6b7385; margin-bottom:6px;"
    value_style = "font-size:1.35rem; font-weight:600; color:#0a0e1a;"

    for col, label, value in [
        (col1, "Asking price", fmt_price(report.get("price_ils"))),
        (col2, "Size",          fmt_num(report.get("size_sqm"), " sqm")),
        (col3, "Rooms",         fmt_num(report.get("num_rooms"))),
        (col4, "Confidence",    fmt_pct(report.get("confidence"))),
    ]:
        col.markdown(
            f"""<div style="{metric_style}">
                <div style="{label_style}">{label}</div>
                <div style="{value_style}">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    md_summary = (report.get("markdown_summary") or "").strip()
    if md_summary:
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.8rem; color:#6b7385; font-weight:600; "
            "margin-bottom:8px;'>SUMMARY</div>",
            unsafe_allow_html=True,
        )
        st.markdown(md_summary)

    features = report.get("key_features") or []
    if features:
        pills = "".join(
            f"<span style='background:#E1F5EE; color:#085041; font-size:0.85rem; "
            f"padding:5px 12px; border-radius:100px; margin:0 6px 6px 0; "
            f"display:inline-block;'>{f}</span>"
            for f in features
        )
        st.markdown(
            "<div style='margin-top:1.25rem; font-size:0.8rem; color:#6b7385; "
            "font-weight:600; margin-bottom:8px;'>KEY FEATURES</div>"
            f"<div>{pills}</div>",
            unsafe_allow_html=True,
        )

    image_scores = report.get("image_scores") or []
    if image_scores:
        st.markdown(
            "<div style='margin-top:1.5rem; font-size:0.8rem; color:#6b7385; "
            "font-weight:600; margin-bottom:8px;'>IMAGE CONDITION SCORES</div>",
            unsafe_allow_html=True,
        )
        for img in image_scores:
            room = img.get("room_type", "—")
            score = img.get("condition_score", "—")
            conf = img.get("confidence")
            conf_str = f" ({int(float(conf)*100)}%)" if conf is not None else ""
            try:
                s_int = int(score)
                stars = "★" * s_int + "☆" * (5 - s_int)
            except Exception:
                stars = "—"
            st.markdown(
                f"<div style='padding:8px 12px; background:#fbf6e6; border-radius:8px; "
                f"margin-bottom:6px; display:flex; justify-content:space-between;'>"
                f"<span><strong>{str(room).title()}</strong>{conf_str}</span>"
                f"<span style='color:#b08820;'>{stars}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    similar = report.get("similar_listings") or []
    if similar:
        st.markdown(
            "<div style='margin-top:1.5rem; font-size:0.8rem; color:#6b7385; "
            "font-weight:600; margin-bottom:8px;'>SIMILAR LISTINGS</div>",
            unsafe_allow_html=True,
        )
        for s in similar:
            if isinstance(s, dict):
                t = s.get("title", "Listing")
                desc = s.get("description", "")
                sim = s.get("similarity_score")
                sim_str = f" — {int(float(sim)*100)}% match" if sim is not None else ""
                st.markdown(
                    f"<div style='padding:10px 14px; background:#ffffff; "
                    f"border:1px solid rgba(20,25,40,0.1); border-radius:8px; "
                    f"margin-bottom:6px;'>"
                    f"<div style='font-weight:600;'>{t}{sim_str}</div>"
                    f"<div style='font-size:0.85rem; color:#3a4156; margin-top:4px;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"- {s}")

    rag_insight = (report.get("rag_insight") or "").strip()
    if rag_insight:
        st.markdown(
            "<div style='margin-top:1.25rem; font-size:0.8rem; color:#6b7385; "
            "font-weight:600; margin-bottom:8px;'>MARKET INSIGHT</div>",
            unsafe_allow_html=True,
        )
        st.info(rag_insight)

    notes = (report.get("enrichment_notes") or "").strip()
    if notes and notes.lower() not in ("none", "n/a"):
        st.markdown(
            f"<div style='margin-top:1.25rem; padding:10px 14px; background:#fef3e6; "
            f"border-left:3px solid #ef9f27; border-radius:6px; font-size:0.88rem;'>"
            f"<strong>Note:</strong> {notes}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("View raw JSON response"):
        st.json(result)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    groq_ok = check_groq()
    n8n_ok = check_n8n()
    pplx_ok = check_perplexity()
    pc_ok = check_pinecone()
    pc_count = get_pinecone_count() if pc_ok else 0
    g_cls  = "online" if groq_ok else "offline"
    n_cls  = "online" if n8n_ok else "offline"
    p_cls  = "online" if pplx_ok else "offline"
    pc_cls = "online" if pc_ok else "offline"

    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <div class="footer-left" style="margin-bottom:0.4rem;">System Status</div>
            <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
                <span class="status-badge {g_cls}">Groq · Llama 3.1</span>
                <span class="status-badge {p_cls}">Perplexity · Sonar</span>
                <span class="status-badge {n_cls}">n8n · Workflow</span>
                <span class="status-badge {pc_cls}">Pinecone · {pc_count} reports</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_listing = db.get_last_listing()
    if last_listing:
        addr = (last_listing.get("payload") or {}).get("description", "")[:50]
        st.markdown(
            f"""
            <div style="background:#E6F1FB; padding:8px 12px; border-radius:8px;
                        font-size:0.78rem; color:#0C447C; margin-bottom:1rem;">
                📎 <strong>Listing loaded</strong><br>{addr}...
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Chat History")
    if st.button("New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        n = len(db.list_sessions()) + 1
        new_id = db.create_session(f"Chat {n}")
        st.session_state.current_session_id = new_id
        st.rerun()

    st.markdown("---")

    sessions = db.list_sessions()
    if not sessions:
        st.caption("No chats yet.")
    else:
        for s in sessions:
            sid = s["id"]
            is_active = (sid == st.session_state.current_session_id)
            msgs = db.get_messages(sid)
            display_name = s["title"]
            if msgs:
                first_user = next((m for m in msgs if m["role"] == "user"), None)
                if first_user:
                    display_name = generate_chat_title(first_user["content"])

            label = f"> {display_name}" if is_active else display_name
            if st.button(label, key=f"chat_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()

            if st.button("Delete", key=f"del_{sid}", use_container_width=True):
                db.delete_session(sid)
                if sid == st.session_state.current_session_id:
                    remaining = db.list_sessions()
                    if remaining:
                        st.session_state.current_session_id = remaining[0]["id"]
                    else:
                        st.session_state.current_session_id = db.create_session("Chat 1")
                st.rerun()

            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)


# ============================================================
# TOP BAR
# ============================================================
top_left, top_right = st.columns([6, 1])
with top_left:
    st.markdown(
        '<div class="hero-badge">AI · PROPERTY INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
with top_right:
    btn_label = "Dark" if not st.session_state.dark_mode else "Light"
    if st.button(btn_label, key="theme_btn", use_container_width=True, type="secondary"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <h1 class="hero-title">AI Property Triage <span class="accent">System</span></h1>
    <p class="hero-subtitle">AI-powered real estate listing intake and evaluation</p>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab_chat, tab_form = st.tabs(["AI Assistant", "Submit Listing"])

# ============================================================
# TAB 1 — AI ASSISTANT
# ============================================================
with tab_chat:
    st.markdown("### Real Estate Assistant")

    listing_ctx = build_listing_context()
    if listing_ctx:
        st.caption(
            "💡 Your most recent listing is loaded as context. "
            "Ask questions like: \"What do you think of my listing?\" or "
            "\"Is this price reasonable?\""
        )
    else:
        st.caption(
            "Ask questions about property listings, prices, housing markets, "
            "or investment opportunities. (Submit a listing in the next tab "
            "to enable context-aware answers.)"
        )

    sid = st.session_state.current_session_id
    messages = db.get_messages(sid)

    chat_container = st.container(height=300)
    with chat_container:
        for message in messages:
            avatar = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(avatar):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask something about real estate...")

    if prompt:
        db.add_message(sid, "user", prompt)

        # ROUTER: decide whether this message goes to Groq, Perplexity Sonar,
        # or Pinecone retrieval. Keyword-based, deterministic, fast.
        # See router.py for the rules.
        decision = router.route(prompt, has_listing_context=bool(listing_ctx))

        # If the router asked for "pinecone", do the vector search NOW and
        # build an enriched context block to feed into Groq. We still answer
        # via Groq — Pinecone only supplies the retrieved listings.
        pinecone_context = None
        if decision == "pinecone":
            try:
                # Broad questions like "show me my listings", "compare my
                # apartments", "list all my submissions" don't have specific
                # content to semantic-match against. For those we return
                # ALL listings instead of top-3-by-similarity.
                broad_query = bool(re.search(
                    r"\b(all|every|list|show\s+me|compare)\b.*"
                    r"\b(my\s+)?(listings|properties|reports|submissions|"
                    r"apartments|houses|villas|offices)\b",
                    prompt, re.IGNORECASE,
                ))
                if broad_query:
                    results = pinecone_client.list_all_listings(limit=20)
                else:
                    results = pinecone_client.search_listings(prompt, top_k=5)

                if results:
                    header = (
                        f"MATCHING PAST LISTINGS ({len(results)} found in vector store). "
                        "Use these in your answer — name specific facts (location, price, "
                        "size, rooms, features) for each one the user is asking about. "
                        "If the user asked for ALL listings, summarise every one of them."
                    )
                    lines = [header]
                    for i, r in enumerate(results, 1):
                        md = r.get("metadata") or {}
                        score = r.get("score", 0)
                        lines.append(
                            f"\n[{i}] match score {score:.2f} | id {r.get('id')}"
                        )
                        for field in ("property_type", "location", "price_ils",
                                      "num_rooms", "size_sqm", "key_features",
                                      "routing_decision", "submitted_at",
                                      "description"):
                            v = md.get(field)
                            if v not in (None, ""):
                                lines.append(f"    {field}: {v}")
                    pinecone_context = "\n".join(lines)
                else:
                    pinecone_context = (
                        "PINECONE LOOKUP RESULT: no past listings in the vector "
                        "store matched the user's query. Politely tell the user "
                        f"you couldn't find a past submission matching \"{prompt}\". "
                        "Ask them to clarify the listing they're thinking of "
                        "(e.g. location, type, approximate date). "
                        "Do NOT use the identity-attack refusal. "
                        "Do NOT use the off-topic refusal. "
                        "This is a normal search-miss, not an attack."
                    )
            except Exception as e:
                pinecone_context = f"[Vector store error: {e}]"

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Tiny indicator so the user can SEE which backend answered.
                # Useful for the demo video and for the instructor.
                if decision == "perplexity":
                    backend_label = "🌐 *Searching the web via Perplexity Sonar…*"
                elif decision == "pinecone":
                    backend_label = "📦 *Looking up past listings in vector store…*"
                elif decision == "rag":
                    backend_label = "📊 *Comparing with market database (ChromaDB)…*"
                else:
                    backend_label = "💬 *Answering from local knowledge (Groq Llama 3.1)…*"
                badge = st.empty()
                badge.caption(backend_label)

                placeholder = st.empty()
                full_response = ""
                history = db.get_messages(sid)

                # For pinecone retrieval, send ONLY the current question
                # without the chat history. The 8B model otherwise mixes
                # prior turns ("you said only one listing") with the fresh
                # Pinecone matches and contradicts itself. Each retrieval
                # query is treated as standalone.
                groq_history = history
                if decision == "pinecone":
                    groq_history = [{"role": "user", "content": prompt}]

                try:
                    if decision == "perplexity":
                        # Stream from Perplexity Sonar
                        for content in sonar_client.stream_sonar(
                                user_message=prompt,
                                history=history,
                                listing_context=listing_ctx,
                        ):
                            full_response += content
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)

                    elif decision == "rag":
                        # Query ChromaDB via EC2 RAG service, then ground Groq's answer
                        # in the returned similar listings.
                        query_text = prompt

                        if listing_ctx:
                            query_text = f"""
                        User question:
                        {prompt}

                        Most recent submitted listing context:
                        {listing_ctx}
                        """

                        if len(query_text.strip()) < 10:
                            full_response = (
                                "⚠️ Please provide more details (at least 10 characters) "
                                "about the property you want to compare with the market."
                            )
                            placeholder.markdown(full_response)
                        else:
                            try:
                                rag_result = rag_client.query_rag(query_text)
                                similar = (
                                        rag_result.get("similar_listings")
                                        or rag_result.get("results")
                                        or rag_result.get("matches")
                                        or rag_result.get("documents")
                                        or rag_result.get("listings")
                                        or []
                                )
                                with st.expander("Debug RAG response"):
                                    st.json(rag_result)
                                insight = rag_result.get("insight", "") or rag_result.get("rag_insight", "")

                                if not similar and not insight:
                                    rag_context = (
                                        "MARKET COMPARISON: The ChromaDB market database returned "
                                        "no similar listings for this query. Politely tell the user "
                                        "you couldn't find comparable properties in the database, "
                                        "and suggest they provide more specific details (location, "
                                        "size, features)."
                                    )
                                else:
                                    # Format the similar listings for the LLM
                                    # Format the similar listings for the LLM
                                    best_listing = None

                                    if similar:
                                        def get_score(item):
                                            if isinstance(item, dict):
                                                return float(item.get("score") or item.get("similarity_score") or 0)
                                            return 0


                                        best_listing = max(similar, key=get_score)

                                    lines = [
                                        "MARKET COMPARISON FROM CHROMADB.",
                                        "CRITICAL RULES:",
                                        "1. Use ONLY the retrieved listings below.",
                                        "2. Do NOT invent prices, sizes, scores, floors, amenities, or locations.",
                                        "3. Do NOT change similarity scores.",
                                        "4. If a field is missing, say 'not provided'.",
                                        "5. First, clearly identify the BEST MATCH based on the highest similarity score.",
                                        "6. Then compare the retrieved listings briefly.",
                                        ""
                                    ]

                                    if best_listing and isinstance(best_listing, dict):
                                        lines.append("BEST MATCH BY HIGHEST CHROMADB SCORE:")
                                        lines.append(f"ID: {best_listing.get('id', 'not provided')}")
                                        lines.append(f"Title: {best_listing.get('title', 'not provided')}")
                                        lines.append(
                                            f"Score: {best_listing.get('score') or best_listing.get('similarity_score') or 'not provided'}")
                                        lines.append(
                                            f"Summary: {best_listing.get('summary') or best_listing.get('description') or 'not provided'}")
                                        lines.append("")

                                    lines.append("RETRIEVED LISTINGS:")
                                    for i, l in enumerate(similar[:5], 1):
                                        if isinstance(l, dict):
                                            title = l.get("title") or l.get("name") or f"Listing {i}"
                                            summary = l.get("summary") or l.get("description", "")
                                            score = l.get("score") or l.get("similarity_score") or 0
                                            try:
                                                pct = int(float(score) * 100)
                                            except Exception:
                                                pct = 0
                                            lines.append(
                                                f"""
                                            [{i}]
                                            ID: {l.get("id", "not provided") if isinstance(l, dict) else "not provided"}
                                            Title: {title}
                                            Raw similarity score: {score}
                                            Match percentage: {pct}%
                                            Summary: {summary[:500]}
                                            """
                                            )
                                        else:
                                            lines.append(f"\n[{i}] {l}")

                                    if insight:
                                        lines.append(f"\n\nMarket insight: {insight}")

                                    rag_context = "\n".join(lines)

                                # Stream from Groq with RAG context (no chat history to
                                # avoid contamination, like we do for Pinecone)
                                # For RAG, we want a longer, more detailed comparison response.
                                # Override the default 2-3 line limit by prepending a system note.
                                rag_system_override = (
                                    "IMPORTANT: For this query, provide a DETAILED market comparison "
                                    "(8-15 sentences). Quote specific facts from each similar listing. "
                                    "Compare prices, sizes, locations, and features. The user needs a "
                                    "thorough market analysis, NOT a brief 2-3 line summary."
                                )

                                full_rag_context = rag_system_override + "\n\n" + rag_context

                                for content in chat_client.stream_response(
                                        [{"role": "user", "content": prompt}],
                                        listing_context=full_rag_context,
                                ):
                                    if content.startswith("\x00REPLACE\x00"):
                                        full_response = content.replace("\x00REPLACE\x00", "")
                                        placeholder.markdown(full_response)
                                    else:
                                        full_response += content
                                        placeholder.markdown(full_response + "▌")
                                placeholder.markdown(full_response)

                            except ValueError as ve:
                                full_response = f"⚠️ {str(ve)}"
                                placeholder.markdown(full_response)
                            except requests.exceptions.Timeout:
                                full_response = "⚠️ RAG service timed out. EC2 services may be cold-starting — please try again in a moment."
                                placeholder.markdown(full_response)
                            except Exception as e:
                                full_response = f"⚠️ RAG service unavailable: {str(e)}"
                                placeholder.markdown(full_response)

                    else:
                        # Stream from Groq. If we have Pinecone results,
                        # we prepend them to the listing context so Groq
                        # can ground its answer in real past submissions.
                        full_ctx = listing_ctx or ""
                        if pinecone_context:
                            if full_ctx:
                                full_ctx = pinecone_context + "\n\n" + full_ctx
                            else:
                                full_ctx = pinecone_context

                        for content in chat_client.stream_response(
                                groq_history,
                                listing_context=full_ctx if full_ctx else None,
                        ):
                            if content.startswith("\x00REPLACE\x00"):
                                full_response = content.replace("\x00REPLACE\x00", "")
                                placeholder.markdown(full_response)
                            else:
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Error: {e}"
                    placeholder.error(full_response)
                db.add_message(sid, "assistant", full_response)


# ============================================================
# TAB 2 — SUBMIT LISTING
# ============================================================
with tab_form:
    st.markdown("### Submit Property Listing")
    st.caption("AI-powered validation, analysis, and routing.")

    with st.form("property_form"):
        agent_name = st.text_input("Agent Name", placeholder="")
        property_description = st.text_area(
            "Property Description",
            height=160,
            placeholder=(
                "Example: 3-bedroom apartment in central Tel Aviv, 95 sqm, renovated kitchen, "
                "two bathrooms, balcony with sea view. Asking price: 3,200,000 NIS."
            ),
        )
        st.markdown("##### Property Images")
        img_tab1, img_tab2 = st.tabs(["Upload Files", "Paste URLs"])

        with img_tab1:
            uploaded_files = st.file_uploader(
                "Upload photos",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

        with img_tab2:
            image_urls_text = st.text_area(
                "URLs",
                placeholder="https://example.com/photo1.jpg, https://example.com/photo2.jpg",
                height=70,
                label_visibility="collapsed",
            )
        submit_button = st.form_submit_button(
            "Submit for AI analysis",
            type="primary",
        )

    if submit_button:
        errors = []
        if not agent_name.strip():
            errors.append("Agent name is required.")
        if not property_description.strip():
            errors.append("Property description is required.")

        image_urls_list = []
        images_b64_list = []
        uploaded_image_scores = []

        if uploaded_files:
            for f in uploaded_files:
                file_bytes = f.getvalue()

                # Keep a light record of the uploaded image in the submitted payload
                b64_string = base64.b64encode(file_bytes).decode("utf-8")
                images_b64_list.append({
                    "filename": f.name,
                    "mime_type": f.type,
                    "data_base64": b64_string,
                })

                # Analyse uploaded file directly using Layer 3 image analyser
                try:
                    files = {
                        "file": (
                            f.name,
                            file_bytes,
                            f.type or "application/octet-stream",
                        )
                    }

                    img_response = requests.post(
                        IMAGE_ANALYSER_UPLOAD_URL,
                        files=files,
                        timeout=120,
                    )

                    img_response.raise_for_status()
                    img_data = img_response.json()

                    # Some APIs return the result directly, others wrap it in "result"
                    if isinstance(img_data, dict) and isinstance(img_data.get("result"), dict):
                        img_data = img_data["result"]

                    uploaded_image_scores.append({
                        "url": f"uploaded_file:{f.name}",
                        "room_type": img_data.get("room_type", "unknown"),
                        "condition_score": img_data.get("condition_score"),
                        "confidence": img_data.get("confidence"),
                    })

                except Exception as e:
                    errors.append(f"Image upload analysis failed for {f.name}: {e}")

        if image_urls_text and image_urls_text.strip():
            raw_urls = re.split(r'[,\n]+', image_urls_text)
            image_urls_list = [u.strip() for u in raw_urls if u.strip()]
        if not images_b64_list and not image_urls_list:
            errors.append("At least one image is required (upload or URL).")

        if errors:
            for err in errors:
                st.error(err)
        else:
            payload = {
                "submission_id": f"sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "submitted_at": datetime.now().isoformat(),
                "agent_name": agent_name.strip(),
                "description": property_description.strip(),
                "image_urls": image_urls_list,
                "images_base64": images_b64_list,
                "precomputed_image_scores": uploaded_image_scores,
            }

            with st.expander("Submitted payload"):
                preview = dict(payload)
                if preview["images_base64"]:
                    if preview.get("precomputed_image_scores"):
                        preview["precomputed_image_scores"] = preview["precomputed_image_scores"]
                    preview["images_base64"] = [
                        {
                            "filename": img["filename"],
                            "mime_type": img["mime_type"],
                            "data_base64": f"<{len(img['data_base64'])} chars>",
                        }
                        for img in preview["images_base64"]
                    ]
                st.json(preview)

            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        N8N_WEBHOOK_URL,
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            render_property_report(result)
                            # Save to SQLite so Tab 1 can reference "the last listing"
                            db.save_last_listing(payload, result.get("report"))
                            # Save to Pinecone so the chat can search ALL past listings,
                            # not just the most recent one. This is best-effort: if
                            # Pinecone is misconfigured we still show the report.
                            try:
                                vec_id = pinecone_client.upsert_listing(
                                    payload, result.get("report")
                                )
                                st.caption(
                                    f"📌 Report saved to vector store as `{vec_id}` — "
                                    f"you can ask about it later from any chat."
                                )
                            except Exception as pe:
                                st.caption(f"⚠️ Could not save to vector store: {pe}")
                        except Exception as e:
                            st.error(f"Could not parse response: {e}")
                            st.text(response.text)
                    elif response.status_code == 404:
                        st.error("n8n webhook not found (404). Ensure workflow is Active.")
                    elif response.status_code == 422:
                        try:
                            result = response.json()
                            render_property_report(result)
                        except Exception:
                            st.error(f"Submission rejected: {response.text}")
                    else:
                        st.error(f"Request failed: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.warning("Could not reach n8n.")
                except requests.exceptions.Timeout:
                    st.error(f"Request timed out after {REQUEST_TIMEOUT}s.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")