import requests
import streamlit as st

API_URL = "http://localhost:8000"


def check_pipeline_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("pipeline_loaded", False), data.get("status", "unknown")
    except requests.exceptions.RequestException:
        pass
    return False, "unavailable"


def query_rag(question: str) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
    return None


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
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Inline-style render helpers ────────────────────────────────────────────────
# All styles are 100% inline so Streamlit cannot sanitize them away.

FONT_SANS = "font-family:'DM Sans',sans-serif;"
FONT_MONO = "font-family:'DM Mono',monospace;"


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
    # Preserve newlines
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG Assistant", page_icon="◈", layout="wide")
    inject_global_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("System Active Check")
        is_active, status = check_pipeline_status()
        if is_active:
            st.success("Pipeline active")
        elif status == "unavailable":
            st.error("Server unavailable")
        else:
            st.warning("Initializing...")
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Redwood Enterprise RAG Assistant")

    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            if "metrics" in msg:
                render_pills(msg["metrics"])
            render_assistant_bubble(msg["content"], sources=msg.get("sources"))

    # Input
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_user_bubble(prompt)

        with st.spinner(""):
            result = query_rag(prompt)

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