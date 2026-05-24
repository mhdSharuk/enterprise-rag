# ui.py
import os
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse


load_dotenv()

API_URL = "http://localhost:8000"

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL   = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


# ── Langfuse ────────────────────────────────────────────────────────────────────

def fetch_langfuse_scores() -> dict[str, float]:
    """Fetch latest score per metric name from Langfuse, return {name: value}."""
    try:
        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL
        )
        scores = {x.name: x.value for x in client.api.trace.get("c35c67f24f3e02df7a9122e9e327ed29").scores}
        return scores
    except Exception as e:
        return {"_error": str(e)}


# ── Employees ───────────────────────────────────────────────────────────────────

def load_employees() -> list[str]:
    path = Path("employees_list.txt")
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return sorted([line.strip() for line in lines if line.strip()])


# ── API helpers ─────────────────────────────────────────────────────────────────

def check_pipeline_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("pipeline_loaded", False), data.get("status", "unknown")
    except requests.exceptions.RequestException:
        pass
    return False, "unavailable"


def query_rag(question: str, user_name: str) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "user_name": user_name},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
    return None


# ── Styles ──────────────────────────────────────────────────────────────────────

def inject_global_styles():
    """Only inject styles that Streamlit won't strip — app-level chrome."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

        .stApp { background-color: #0c0e14; }

        [data-testid="stSidebar"] {
            background-color: #11131c;
            border-right: 1px solid #1c1f2e;
        }
        [data-testid="stSidebar"] .stButton > button {
            background-color: #1c1f2e;
            color: #7a7f98;
            border: 1px solid #252839;
            border-radius: 8px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #252839;
            color: #c0c5d8;
            border-color: #333749;
        }

        h1 {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1.4rem !important;
            color: #dde1ed !important;
            letter-spacing: -0.02em !important;
        }
        hr { border-color: #1c1f2e !important; }

        [data-testid="stChatInput"] textarea {
            font-family: 'DM Sans', sans-serif !important;
            font-size: 14px !important;
            color: #c0c5d8 !important;
            background-color: #11131c !important;
            border-color: #1c1f2e !important;
        }

        /* Hide the native chat message container — we render our own HTML */
        [data-testid="stChatMessage"] { display: none !important; }

        .stSpinner > div { border-top-color: #5b7cf6 !important; }

        /* User selector dropdown — minimal dark styling */
        [data-testid="stSelectbox"] > div > div {
            background-color: #11131c !important;
            border: 1px solid #1e2230 !important;
            border-radius: 8px !important;
            color: #7a7f98 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
        }
        [data-testid="stSelectbox"] label {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Render helpers ──────────────────────────────────────────────────────────────

FONT_SANS = "font-family:'DM Sans',sans-serif;"
FONT_MONO = "font-family:'DM Mono',monospace;"

# Score metric config: name → (accent color, bg, border, symbol)
SCORE_PALETTE = {
    "answer_correctness": ("#6dbf8a", "#0f1f16", "#1a3824", "◆"),
    "answer_relevancy":   ("#5b9cf6", "#0f1620", "#1a2a40", "◆"),
    "context_recall":     ("#a07ec8", "#1a1228", "#2e1f48", "◆"),
    "faithfullness":      ("#f6a05b", "#201508", "#3d2610", "◆"),
    "mrr":                ("#5bbfbf", "#0a1e1e", "#163636", "◆"),
    "recall@10":          ("#e07eb8", "#1e0f18", "#3d1a30", "◆"),
    "recall@5":           ("#c8c85b", "#1c1c08", "#363610", "◆"),
}
DEFAULT_PALETTE = ("#7a7f98", "#12141e", "#1e2030", "◆")


def render_score_card(name: str, value: float):
    color, bg, border, sym = SCORE_PALETTE.get(name, DEFAULT_PALETTE)
    pct = int(value * 100)
    # Bar width capped at 100%
    bar_w = min(pct, 100)

    st.markdown(
        f"""
        <div style="
            background:{bg};
            border:1px solid {border};
            border-radius:10px;
            padding:10px 13px 9px 13px;
            margin-bottom:7px;
        ">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
            <span style="{FONT_MONO}font-size:10px;color:#ffffff;letter-spacing:0.06em;text-transform:uppercase;">{name}</span>
            <span style="{FONT_MONO}font-size:13px;font-weight:500;color:{color};">{value:.4f}</span>
          </div>
          <div style="height:3px;background:#1c1f2e;border-radius:2px;overflow:hidden;">
            <div style="height:100%;width:{bar_w}%;background:{color};border-radius:2px;opacity:0.85;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scores_section(scores: dict[str, float]):
    if "_error" in scores:
        st.markdown(
            f"""<div style="{FONT_MONO}font-size:10px;color:#c05050;padding:6px 0;">
            ⚠ {scores['_error'][:60]}</div>""",
            unsafe_allow_html=True,
        )
        return

    if not scores:
        st.markdown(
            f"""<div style="{FONT_MONO}font-size:10px;color:#4a4f68;padding:6px 0;">No scores found.</div>""",
            unsafe_allow_html=True,
        )
        return

    for name, value in sorted(scores.items()):
        render_score_card(name, value)


def render_user_bubble(content: str):
    safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""<div style="display:flex;justify-content:flex-end;margin:6px 0;">
  <div style="
    max-width:68%;
    padding:12px 16px;
    border-radius:18px 18px 4px 18px;
    background-color:#1a1f38;
    border:1px solid #252d52;
    color:#c8ceea;
    {FONT_SANS}
    font-size:14px;
    line-height:1.65;
    word-break:break-word;
  ">{safe}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_pills(metrics: dict):
    is_cached = metrics.get("is_cached", False)
    tokens    = metrics.get("tokens_used", 0)
    time_s    = metrics.get("time_taken", 0)

    cache_bg    = "#1a2e22" if is_cached else "#1a2030"
    cache_bdr   = "#2a4d38" if is_cached else "#2a3050"
    cache_col   = "#6dbf8a" if is_cached else "#6a9ecf"
    cache_label = "● cached" if is_cached else "○ fresh"

    pill = (
        "display:inline-flex;align-items:center;"
        "padding:2px 9px;border-radius:20px;"
        f"{FONT_MONO}"
        "font-size:10px;font-weight:500;letter-spacing:0.03em;white-space:nowrap;"
    )

    st.markdown(
        f"""<div style="display:flex;gap:5px;flex-wrap:wrap;margin:4px 0 6px 0;">
  <span style="{pill}background:{cache_bg};border:1px solid {cache_bdr};color:{cache_col};">{cache_label}</span>
  <span style="{pill}background:#221828;border:1px solid #372540;color:#a07ec8;">⬡ {tokens:,} tok</span>
  <span style="{pill}background:#182226;border:1px solid #263640;color:#5ba8a8;">◷ {time_s}s</span>
</div>""",
        unsafe_allow_html=True,
    )


def render_assistant_bubble(content: str, sources: list | None = None):
    safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n\n", "</p><p style='margin:6px 0 0 0;'>").replace("\n", "<br>")
    safe = f"<p style='margin:0;'>{safe}</p>"

    sources_html = ""
    if sources:
        items = "".join(
            f"<div style='{FONT_MONO}font-size:11px;color:#5b7cf6;margin-top:4px;'>↗ {s}</div>"
            for s in sources
        )
        sources_html = (
            f"<div style='margin-top:10px;font-size:10px;font-weight:600;"
            f"letter-spacing:0.1em;text-transform:uppercase;color:#3e4560;'>Sources</div>"
            f"{items}"
        )

    st.markdown(
        f"""<div style="display:flex;justify-content:flex-start;margin:0 0 6px 0;">
  <div style="
    max-width:72%;
    padding:12px 16px;
    border-radius:18px 18px 18px 4px;
    background-color:#13161f;
    border:1px solid #1e2230;
    color:#c2c7d8;
    {FONT_SANS}
    font-size:14px;
    line-height:1.65;
    word-break:break-word;
  ">{safe}{sources_html}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG Assistant", page_icon="◈", layout="wide")
    inject_global_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    employees = load_employees()

    # ── Top bar: title left, user selector right ──
    title_col, spacer_col, user_col = st.columns([4, 2, 2])
    with title_col:
        st.title("Redwood Enterprise RAG Assistant")
    with user_col:
        selected_user = st.selectbox(
            "User",
            options=employees,
            index=0,
            label_visibility="collapsed",
        )

    # ── Sidebar ──
    with st.sidebar:
        # Pipeline status
        st.header("System Active Check")
        is_active, status = check_pipeline_status()
        if is_active:
            st.success("Pipeline active")
        elif status == "unavailable":
            st.error("Server unavailable")
        else:
            st.warning("Initializing...")

        st.divider()

        # Evaluation scores
        st.markdown(
            f"""<div style="
                display:flex;justify-content:space-between;align-items:center;
                margin-bottom:10px;
            ">
              <span style="{FONT_SANS}font-size:12px;font-weight:600;color:#dde1ed;
                           letter-spacing:0.04em;text-transform:uppercase;">
                Eval Scores
              </span>
              <span style="{FONT_MONO}font-size:9px;color:#3e4560;">langfuse</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("↻ Refresh scores", use_container_width=True):
            st.session_state.pop("lf_scores", None)

        if "lf_scores" not in st.session_state:
            with st.spinner(""):
                st.session_state.lf_scores = fetch_langfuse_scores()

        render_scores_section(st.session_state.lf_scores)

        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Chat history ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            if "metrics" in msg:
                render_pills(msg["metrics"])
            render_assistant_bubble(msg["content"], sources=msg.get("sources"))

    # ── Input ──
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_user_bubble(prompt)

        with st.spinner(""):
            result = query_rag(prompt, selected_user)

        if result:
            answer  = result.get("answer", "")
            sources = result.get("sources", [])
            metrics = {
                "is_cached":   result.get("is_cached", False),
                "tokens_used": result.get("tokens_used", 0),
                "time_taken":  result.get("time_taken", 0),
            }
            render_pills(metrics)
            render_assistant_bubble(answer, sources=sources)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "metrics": metrics,
                "sources": sources,
            })
        else:
            st.error("Failed to get a response. Please try again.")


if __name__ == "__main__":
    main()