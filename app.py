import time
from datetime import datetime
import streamlit as st

from document_loader import load_and_split_document, validate_document
from qa_engine import generate_answer_stream, generate_answer, validate_api_configuration, get_model_info
from vector_store import VectorStore
from utils import export_chat_history, get_document_insights, is_conversational_query

st.set_page_config(
    page_title="Atlas Document Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_PRESETS = {
    "Light": {
        "bg": "#f3efe6",
        "bg2": "#eaf2ef",
        "panel": "rgba(255,255,255,0.95)",
        "panel_soft": "rgba(255,255,255,0.78)",
        "panel_ghost": "rgba(255,255,255,0.65)",
        "text": "#17212b",
        "muted": "#5e6b78",
        "line": "rgba(23,33,43,0.10)",
        "accent": "#0f766e",
        "accent_2": "#b45309",
        "accent_3": "#6d28d9",
        "shadow": "0 24px 60px rgba(23,33,43,0.10)",
        "sidebar_bg": "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,239,231,0.96))",
    },
    "Dark": {
        "bg": "#0d1117",
        "bg2": "#121a24",
        "panel": "rgba(17,24,39,0.94)",
        "panel_soft": "rgba(17,24,39,0.76)",
        "panel_ghost": "rgba(17,24,39,0.60)",
        "text": "#edf2f7",
        "muted": "#94a3b8",
        "line": "rgba(148,163,184,0.18)",
        "accent": "#5eead4",
        "accent_2": "#fbbf24",
        "accent_3": "#a78bfa",
        "shadow": "0 26px 64px rgba(0,0,0,0.35)",
        "sidebar_bg": "linear-gradient(180deg, rgba(15,23,42,0.97), rgba(17,24,39,0.98))",
    },
}

def inject_theme_styles(theme_name):
    theme = THEME_PRESETS[theme_name]
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

    :root {{
        --bg: {theme['bg']};
        --bg-2: {theme['bg2']};
        --panel: {theme['panel']};
        --panel-soft: {theme['panel_soft']};
        --panel-ghost: {theme['panel_ghost']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --line: {theme['line']};
        --accent: {theme['accent']};
        --accent-2: {theme['accent_2']};
        --accent-3: {theme['accent_3']};
        --shadow: {theme['shadow']};
        --radius-xl: 30px;
        --radius-lg: 22px;
        --radius-md: 16px;
    }}

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
        color: var(--text);
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 26%),
            linear-gradient(135deg, var(--bg), var(--bg-2));
    }}

    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }}

    section[data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']};
        border-right: 1px solid var(--line);
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 2.15rem 2rem 1.85rem;
        border-radius: var(--radius-xl);
        background: linear-gradient(135deg, var(--panel), var(--panel-soft));
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
    }}

    .hero::before {{
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 18% 30%, rgba(15,118,110,0.14), transparent 26%),
            radial-gradient(circle at 86% 18%, rgba(180,83,9,0.12), transparent 22%);
        pointer-events: none;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        inset: auto -6% -42% auto;
        width: 340px;
        height: 340px;
        background: radial-gradient(circle, rgba(15,118,110,0.18), transparent 68%);
        pointer-events: none;
    }}

    .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: var(--accent);
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.72rem;
        position: relative;
        z-index: 1;
    }}

    .hero h1 {{
        margin: 0.9rem 0 0.5rem;
        font-family: 'Fraunces', serif;
        font-size: clamp(2.25rem, 4vw, 4.6rem);
        line-height: 1.02;
        letter-spacing: -0.035em;
        color: var(--text) !important;
        position: relative;
        z-index: 1;
    }}

    .hero p {{
        margin: 0;
        max-width: 780px;
        color: var(--muted) !important;
        font-size: 1.06rem;
        line-height: 1.75;
        position: relative;
        z-index: 1;
    }}

    .metric-card, .panel, .empty-state, .sidebar-card {{
        backdrop-filter: blur(14px);
    }}

    .metric-card {{
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: 0 12px 30px rgba(23, 33, 43, 0.08);
    }}

    .metric-label, .section-title span, .sidebar-title {{
        color: var(--muted);
    }}

    .metric-label {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.35rem;
    }}

    .metric-value {{
        font-size: 1.55rem;
        font-weight: 900;
        color: var(--text);
    }}

    .metric-subtext {{
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.28rem;
    }}

    .panel {{
        padding: 1.15rem;
        border-radius: var(--radius-lg);
        background: linear-gradient(180deg, var(--panel), var(--panel-soft));
        border: 1px solid var(--line);
        box-shadow: 0 16px 40px rgba(23, 33, 43, 0.08);
    }}

    .upload-box {{
        padding: 1.15rem;
        border-radius: 20px;
        border: 1.5px dashed rgba(15, 118, 110, 0.38);
        background: var(--panel-ghost);
    }}

    .doc-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0.25rem 0.45rem 0.25rem 0;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        border: 1px solid rgba(15, 118, 110, 0.18);
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 700;
    }}

    .chat-shell {{
        margin-top: 1.1rem;
    }}

    .chat-message {{
        border-radius: 24px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0;
        box-shadow: 0 14px 36px rgba(23, 33, 43, 0.05);
        border: 1px solid var(--line);
        animation: floatIn 0.28s ease-out;
    }}

    .chat-user {{
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.16), rgba(15, 118, 110, 0.08));
        margin-left: 8%;
    }}

    .chat-assistant {{
        background: linear-gradient(135deg, var(--panel), var(--panel-soft));
        margin-right: 8%;
    }}

    .chat-meta {{
        margin-top: 0.7rem;
        padding-top: 0.7rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.85rem;
    }}

    .typing-box {{
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.9rem 1rem;
        border-radius: 999px;
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: 0 12px 28px rgba(23, 33, 43, 0.06);
    }}

    .typing-dots {{
        display: inline-flex;
        gap: 0.24rem;
    }}

    .typing-dots span {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulseDot 1s infinite ease-in-out;
    }}

    .typing-dots span:nth-child(2) {{ animation-delay: 0.15s; }}
    .typing-dots span:nth-child(3) {{ animation-delay: 0.3s; }}

    .empty-state {{
        padding: 2rem;
        border-radius: var(--radius-xl);
        border: 1px solid var(--line);
        background: linear-gradient(180deg, var(--panel), var(--panel-soft));
        box-shadow: var(--shadow);
        text-align: center;
    }}

    .section-title {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.2rem 0 0.75rem;
    }}

    .section-title h2 {{
        margin: 0;
        font-size: 1.1rem;
        letter-spacing: -0.02em;
        color: var(--text);
    }}

    .sidebar-card {{
        padding: 1rem;
        border-radius: 20px;
        background: var(--panel);
        border: 1px solid var(--line);
        margin-bottom: 0.85rem;
    }}

    .sidebar-title {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.3rem;
    }}

    .sidebar-value {{
        font-size: 1.25rem;
        font-weight: 900;
        color: var(--text);
    }}

    .stButton > button {{
        border-radius: 14px;
        border: 1px solid var(--line);
        font-weight: 700;
    }}

    .stChatInput {{
        background: var(--panel);
    }}

    @keyframes floatIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulseDot {{
        0%, 80%, 100% {{ transform: translateY(0); opacity: 0.55; }}
        40% {{ transform: translateY(-3px); opacity: 1; }}
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""",
        unsafe_allow_html=True,
    )

def initialize_session_state():
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "threads" not in st.session_state:
        st.session_state.threads = {
            "default": {"name": "Default Thread", "history": []}
        }
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = "default"
    
    # Point history reference to active thread's history
    st.session_state.history = st.session_state.threads[st.session_state.current_thread_id]["history"]
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    if "allowed_documents" not in st.session_state:
        st.session_state.allowed_documents = None
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Standard Q&A"
    if "comparison_doc_a" not in st.session_state:
        st.session_state.comparison_doc_a = ""
    if "comparison_doc_b" not in st.session_state:
        st.session_state.comparison_doc_b = ""
    
    # LLM Settings
    if "llm_settings" not in st.session_state:
        st.session_state.llm_settings = {
            "provider": "Mistral",
            "model": "mistral-small",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1500,
            "api_key": "",
            "ollama_url": "http://localhost:11434"
        }

def get_stats():
    if not st.session_state.vs:
        return {"documents": 0, "chunks": 0, "avg_confidence": 0.0}

    document_stats = st.session_state.vs.get_document_stats() or {}
    confidence_values = [item[3].get("confidence", 0) for item in st.session_state.history]
    return {
        "documents": document_stats.get("total_documents", len(st.session_state.uploaded_files)),
        "chunks": document_stats.get("total_chunks", 0),
        "avg_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
    }

def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Document Intelligence Workspace</div>
            <h1>Turn documents into a polished, conversation-first experience.</h1>
            <p>
                Upload PDFs, Word files, Markdown, or text files, then ask questions in a clean chat layout with
                response streaming, source tracking, and custom LLM integrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    stats = get_stats()

    st.sidebar.markdown("## Atlas Controls")
    selected_theme = st.sidebar.selectbox(
        "Appearance",
        list(THEME_PRESETS.keys()),
        index=list(THEME_PRESETS.keys()).index(st.session_state.theme),
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    # Thread Management
    st.sidebar.markdown("### Chat Threads")
    thread_names = {tid: tdata["name"] for tid, tdata in st.session_state.threads.items()}
    selected_thread_id = st.sidebar.selectbox(
        "Select Chat Thread",
        options=list(thread_names.keys()),
        format_func=lambda x: thread_names[x],
        key="thread_selector_sidebar"
    )
    if selected_thread_id != st.session_state.current_thread_id:
        st.session_state.current_thread_id = selected_thread_id
        st.session_state.history = st.session_state.threads[selected_thread_id]["history"]
        st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("+ New Thread", use_container_width=True):
            new_tid = f"thread_{int(time.time())}"
            st.session_state.threads[new_tid] = {"name": f"Thread {len(st.session_state.threads) + 1}", "history": []}
            st.session_state.current_thread_id = new_tid
            st.session_state.history = []
            st.rerun()
    with col2:
        if len(st.session_state.threads) > 1:
            if st.button("🗑️ Delete Thread", use_container_width=True):
                del st.session_state.threads[st.session_state.current_thread_id]
                first_remaining = list(st.session_state.threads.keys())[0]
                st.session_state.current_thread_id = first_remaining
                st.session_state.history = st.session_state.threads[first_remaining]["history"]
                st.rerun()

    # Search Source Filter
    if st.session_state.uploaded_files:
        st.sidebar.markdown("### Source File Filter")
        st.sidebar.caption("Search matches only checked files:")
        allowed = []
        for file_info in st.session_state.uploaded_files:
            fname = file_info["name"]
            is_checked = st.sidebar.checkbox(fname, value=True, key=f"filter_check_{fname}")
            if is_checked:
                allowed.append(fname)
        st.session_state.allowed_documents = allowed
    else:
        st.session_state.allowed_documents = None

    st.sidebar.markdown("### Overview")
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Documents</div>
            <div class="sidebar-value">%s</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-title">Chunks indexed</div>
            <div class="sidebar-value">%s</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-title">Average confidence</div>
            <div class="sidebar-value">%s%%</div>
        </div>
        """
        % (stats["documents"], stats["chunks"], int(stats["avg_confidence"] * 100)),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Session Controls")
    if st.sidebar.button("Clear current chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.threads[st.session_state.current_thread_id]["history"] = []
        st.rerun()

    if st.session_state.uploaded_files and st.sidebar.button("Remove all documents", use_container_width=True):
        st.session_state.vs = None
        st.session_state.uploaded_files = []
        st.session_state.history = []
        for tid in st.session_state.threads:
            st.session_state.threads[tid]["history"] = []
        st.rerun()

def render_upload_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>Upload documents</h2>
            <span>PDF, DOCX, TXT, MD</span>
        </div>
        <p style="margin-top:0;color:#66727f;line-height:1.6;">
            Add documents. The app will generate hierarchical parent-child vector index segments.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported formats: PDF, Word, Text, Markdown",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        process_uploaded_files(uploaded_files)

    if st.session_state.uploaded_files:
        st.markdown(
            f"""
            <div style="margin-top:1rem; padding:0.95rem 1rem; border-radius:18px; background: rgba(15,118,110,0.10); border: 1px solid rgba(15,118,110,0.16);">
                <strong>Ready:</strong> {len(st.session_state.uploaded_files)} document(s) indexed using parent-child chunking.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    new_files = []
    for uploaded_file in uploaded_files:
        if not any(file_info["name"] == uploaded_file.name for file_info in st.session_state.uploaded_files):
            new_files.append(uploaded_file)

    if not new_files:
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, uploaded_file in enumerate(new_files):
        status_text.markdown(f"**Hierarchical Indexing: {uploaded_file.name}**")
        progress_bar.progress((index + 1) / len(new_files))

        try:
            chunks = load_and_split_document(uploaded_file)

            if st.session_state.vs is None:
                st.session_state.vs = VectorStore()

            st.session_state.vs.add_documents(chunks, uploaded_file.name)

            # Stitch parent contents to execute document insights analysis
            full_text_for_analysis = "\n".join([c.get("parent", "") if isinstance(c, dict) else c for c in chunks])
            insights = get_document_insights(full_text_for_analysis)

            st.session_state.uploaded_files.append(
                {
                    "name": uploaded_file.name,
                    "chunks": len(chunks),
                    "uploaded_at": datetime.now().isoformat(),
                    "insights": insights
                }
            )
        except Exception as exc:
            st.error(f"Error processing {uploaded_file.name}: {exc}")

    progress_bar.empty()
    status_text.empty()
    time.sleep(0.35)
    st.rerun()

def render_metrics_row():
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Documents</div>
                <div class="metric-value">{stats['documents']}</div>
                <div class="metric-subtext">Uploaded and indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Chunks</div>
                <div class="metric-value">{stats['chunks']}</div>
                <div class="metric-subtext">Hierarchical search units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Turns</div>
                <div class="metric-value">{len(st.session_state.history)}</div>
                <div class="metric-subtext">Questions answered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{int(stats['avg_confidence'] * 100)}%</div>
                <div class="metric-subtext">Average retrieval score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_citations(metadata):
    confidence = metadata.get("confidence", 0)
    source_file = metadata.get("source_file", "Document")
    response_time = metadata.get("response_time", 0)
    is_comparison = metadata.get("is_comparison", False)
    doc_a = metadata.get("doc_a", "Document A")
    doc_b = metadata.get("doc_b", "Document B")
    
    if is_comparison:
        st.markdown(
            f"<div class='chat-meta'>Comparison Mode · Sources: {doc_a} & {doc_b} · Response time: {response_time:.2f}s</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='chat-meta'>Confidence: {confidence:.0%} · Source: {source_file} · Response time: {response_time:.2f}s</div>",
            unsafe_allow_html=True,
        )
    
    child_chunks = metadata.get("child_chunks", [])
    parent_chunks = metadata.get("parent_chunks", [])
    source_files = metadata.get("source_files", [source_file] * len(child_chunks))
    
    if child_chunks:
        with st.expander("🔍 View Retrieved Source Citations"):
            if is_comparison:
                tab_a, tab_b = st.tabs([f"📄 {doc_a} Sources", f"📄 {doc_b} Sources"])
                
                with tab_a:
                    count_a = 1
                    for child, parent, sfile in zip(child_chunks, parent_chunks, source_files):
                        if sfile == doc_a:
                            st.markdown(f"**Source Snippet {count_a}:**")
                            c_tab, p_tab = st.tabs(["Snippet", "Full Context"])
                            with c_tab:
                                st.info(child)
                            with p_tab:
                                st.caption(parent)
                            count_a += 1
                    if count_a == 1:
                        st.write("No matching citations retrieved for this document.")
                        
                with tab_b:
                    count_b = 1
                    for child, parent, sfile in zip(child_chunks, parent_chunks, source_files):
                        if sfile == doc_b:
                            st.markdown(f"**Source Snippet {count_b}:**")
                            c_tab, p_tab = st.tabs(["Snippet", "Full Context"])
                            with c_tab:
                                st.info(child)
                            with p_tab:
                                st.caption(parent)
                            count_b += 1
                    if count_b == 1:
                        st.write("No matching citations retrieved for this document.")
            else:
                for i, (child, parent, sfile) in enumerate(zip(child_chunks, parent_chunks, source_files), 1):
                    st.markdown(f"**Source {i}:** `{sfile}`")
                    c_tab, p_tab = st.tabs(["Child Snippet (Matched)", "Parent Context (LLM Input)"])
                    with c_tab:
                        st.info(child)
                    with p_tab:
                        st.caption(parent)

def render_chat_interface():
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>Chat workspace</h2>
            <span>Streaming responses with hierarchical parent context</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workspace Mode Toggle
    app_mode = st.radio(
        "Workspace Mode Selection",
        ["Standard Q&A Mode", "Document Comparison Mode"],
        horizontal=True,
        index=0 if st.session_state.get("app_mode", "Standard Q&A") == "Standard Q&A" else 1,
        label_visibility="collapsed"
    )
    
    mapped_mode = "Standard Q&A" if app_mode == "Standard Q&A Mode" else "Document Comparison"
    if mapped_mode != st.session_state.app_mode:
        st.session_state.app_mode = mapped_mode
        st.rerun()

    # Document Selectors for Comparison Mode
    if st.session_state.app_mode == "Document Comparison":
        if not st.session_state.uploaded_files or len(st.session_state.uploaded_files) < 2:
            st.warning("⚠️ Please upload at least 2 documents to unlock Document Comparison Mode.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
            
        st.markdown(
            """
            <div style="margin-bottom:1rem; padding:0.8rem 1rem; border-radius:14px; background: rgba(180,83,9,0.08); border: 1px solid rgba(180,83,9,0.18); font-size: 0.9rem; color: var(--text);">
                📊 <strong>Document Comparison Mode Active</strong>: Choose Document A and Document B below. Questions asked will compare their sections side-by-side.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        doc_names = [f["name"] for f in st.session_state.uploaded_files]
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("Document A", doc_names, index=0, key="comp_select_doc_a")
        with col2:
            default_b_idx = 1 if len(doc_names) > 1 else 0
            doc_b = st.selectbox("Document B", doc_names, index=default_b_idx, key="comp_select_doc_b")
            
        st.session_state.comparison_doc_a = doc_a
        st.session_state.comparison_doc_b = doc_b
        
        if doc_a == doc_b:
            st.error("❌ Document A and Document B must be different files. Please select distinct documents.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # 1. Render historical messages of current thread
    for question, answer, _, metadata in st.session_state.history:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧭"):
            st.markdown(answer)
            render_citations(metadata)

    # 2. Render live streaming output if active
    if "streaming_question" in st.session_state and st.session_state.streaming_question:
        question = st.session_state.streaming_question
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)
            
        with st.chat_message("assistant", avatar="🧭"):
            start_time = time.time()
            is_chitchat = is_conversational_query(question)
            is_comparison = st.session_state.app_mode == "Document Comparison"
            
            if is_chitchat:
                search_results = {
                    "chunks": [""],
                    "child_chunks": [],
                    "scores": [],
                    "metadata": [],
                    "source_files": []
                }
            elif is_comparison:
                doc_a = st.session_state.comparison_doc_a
                doc_b = st.session_state.comparison_doc_b
                search_results = st.session_state.vs.search_comparison(
                    question,
                    doc_a,
                    doc_b,
                    top_k=3
                )
            else:
                search_results = st.session_state.vs.search(
                    question,
                    top_k=3,
                    allowed_documents=st.session_state.allowed_documents
                )
            
            if not is_chitchat and not search_results["chunks"]:
                st.error("No relevant content found in the selected documents.")
                st.session_state.streaming_question = None
            else:
                context = "\n".join(search_results["chunks"])
                
                response_placeholder = st.empty()
                full_response = ""
                
                llm_settings = st.session_state.llm_settings.copy()
                llm_settings["is_comparison"] = is_comparison if not is_chitchat else False
                llm_settings["is_chitchat"] = is_chitchat
                
                try:
                    stream = generate_answer_stream(
                        context,
                        question,
                        llm_settings,
                        st.session_state.history
                    )
                    
                    for token in stream:
                        full_response += token
                        response_placeholder.markdown(full_response + " ▌")
                    
                    response_placeholder.markdown(full_response)
                    response_time = time.time() - start_time
                    
                    confidence = (
                        sum(search_results["scores"]) / len(search_results["scores"])
                        if search_results["scores"]
                        else 0
                    )
                    
                    metadata = {
                        "confidence": 1.0 if is_chitchat else confidence,
                        "source_file": "General Chat" if is_chitchat else (f"{st.session_state.get('comparison_doc_a', 'Doc A')} & {st.session_state.get('comparison_doc_b', 'Doc B')}" if is_comparison else search_results.get("source_files", ["Unknown"])[0]),
                        "source_files": search_results.get("source_files", ["Unknown"]),
                        "response_time": response_time,
                        "timestamp": datetime.now().isoformat(),
                        "child_chunks": search_results.get("child_chunks", []),
                        "parent_chunks": search_results.get("parent_chunks", search_results.get("chunks", [])),
                        "is_comparison": is_comparison if not is_chitchat else False,
                        "doc_a": st.session_state.get("comparison_doc_a") if (is_comparison and not is_chitchat) else None,
                        "doc_b": st.session_state.get("comparison_doc_b") if (is_comparison and not is_chitchat) else None,
                        "is_chitchat": is_chitchat
                    }
                    
                    # Save to active history state
                    st.session_state.history.append((question, full_response, context, metadata))
                    st.session_state.threads[st.session_state.current_thread_id]["history"] = st.session_state.history
                    
                    # Reset streaming state and refresh UI to fully render
                    st.session_state.streaming_question = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during response stream: {str(e)}")
                    st.session_state.streaming_question = None
    elif not st.session_state.history:
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="margin:0 0 0.5rem; font-family:'Fraunces', serif; font-size: 1.8rem;">Your chat starts here</h3>
                <p style="margin:0; color:#66727f; line-height:1.7;">
                    Upload documents to build the knowledge base, then ask questions. The interface streams responses.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat Input Box
    prompt = st.chat_input("Ask a question about your documents", key="main_chat_input")
    
    # Quick prompt actions
    if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
        prompt = st.session_state.pop("quick_prompt")
        
    if prompt:
        process_question(prompt)

    st.markdown("</div>", unsafe_allow_html=True)

def process_question(question):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    if not st.session_state.vs:
        st.error("Upload at least one document before asking a question.")
        return

    is_chitchat = is_conversational_query(question)
    is_comparison = st.session_state.get("app_mode", "Standard Q&A") == "Document Comparison"
    
    if not is_chitchat:
        if is_comparison:
            doc_a = st.session_state.get("comparison_doc_a")
            doc_b = st.session_state.get("comparison_doc_b")
            if not doc_a or not doc_b:
                st.error("Please upload and select Document A and Document B to run comparison queries.")
                return
            if doc_a == doc_b:
                st.error("Document A and Document B must be different files. Please select distinct documents.")
                return
        else:
            if st.session_state.allowed_documents is not None and len(st.session_state.allowed_documents) == 0:
                st.error("Please select at least one document in the sidebar to search.")
                return

    # Trigger streaming workflow by setting state and calling rerun
    st.session_state.streaming_question = question
    st.rerun()

def render_analytics_dashboard():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>Document Analytics & Insights</h2>
            <span>Detailed language and structural metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if not st.session_state.uploaded_files:
        st.info("No documents uploaded yet. Insights will appear here once files are parsed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    for index, file_info in enumerate(st.session_state.uploaded_files, 1):
        name = file_info["name"]
        insights = file_info.get("insights", {})
        
        st.markdown(f"### {index}. {name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", f"{insights.get('word_count', 0):,}")
        with col2:
            st.metric("Estimated Pages", f"{insights.get('estimated_pages', 0.0):.1f}")
        with col3:
            st.metric("Complexity Index", f"{insights.get('complexity_score', 0.0):.1f}/100")
        with col4:
            st.metric("Language", str(insights.get("language", "Unknown")).upper())
            
        st.markdown(f"**Reading Time estimate:** {insights.get('reading_time', 'N/A')}")
        
        keywords = insights.get("keywords", [])
        if keywords:
            st.markdown(f"**Extracted Keywords:** " + ", ".join([f"`{kw}`" for kw in keywords]))
        st.markdown("---")
        
    st.markdown("</div>", unsafe_allow_html=True)

def render_configuration_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>System Configuration</h2>
            <span>Manage LLM connectivity parameters and exports</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.subheader("LLM Provider Settings")
    provider = st.selectbox(
        "Provider",
        ["Mistral", "Ollama"],
        index=0 if st.session_state.llm_settings["provider"] == "Mistral" else 1
    )
    
    model_info = get_model_info()
    
    if provider == "Mistral":
        model_options = model_info["mistral_models"]
        current_model = st.session_state.llm_settings["model"]
        if current_model not in model_options:
            model_options = [current_model] + model_options
            
        model = st.selectbox("Mistral Model", model_options, index=model_options.index(current_model))
        api_key = st.text_input("Custom API Key (Optional, overrides .env key)", value=st.session_state.llm_settings["api_key"], type="password")
        ollama_url = st.session_state.llm_settings["ollama_url"]
    else: # Ollama
        ollama_url = st.text_input("Ollama Endpoint URL", value=st.session_state.llm_settings["ollama_url"])
        model = st.text_input("Ollama Model Name (e.g. llama3, mistral, gemma, phi3)", value=st.session_state.llm_settings["model"])
        api_key = st.session_state.llm_settings["api_key"]
        
    st.subheader("Model Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.llm_settings["temperature"], step=0.05)
    with col2:
        top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=st.session_state.llm_settings["top_p"], step=0.05)
        
    max_tokens = st.slider("Max Prediction Tokens", min_value=100, max_value=3000, value=st.session_state.llm_settings["max_tokens"], step=50)
    
    # Save settings
    if st.button("Apply Parameters Settings"):
        st.session_state.llm_settings = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "ollama_url": ollama_url
        }
        st.success("✅ Applied model settings parameters!")
        
    st.markdown("---")
    st.subheader("Export Chat Logs")
    if not st.session_state.history:
        st.caption("No chat history available in current thread to export.")
    else:
        col_json, col_md, col_pdf = st.columns(3)
        
        with col_json:
            json_data = export_chat_history(st.session_state.history, "json")
            st.download_button(
                "📥 Download JSON Logs",
                data=json_data,
                file_name=f"chat_history_{st.session_state.current_thread_id}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
            
        with col_md:
            md_data = export_chat_history(st.session_state.history, "md")
            st.download_button(
                "📥 Download Markdown Logs",
                data=md_data,
                file_name=f"chat_history_{st.session_state.current_thread_id}_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        with col_pdf:
            pdf_data = export_chat_history(st.session_state.history, "pdf")
            if pdf_data:
                st.download_button(
                    "📥 Download PDF Logs",
                    data=pdf_data,
                    file_name=f"chat_history_{st.session_state.current_thread_id}_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("Download PDF (Error compiling)", disabled=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

def main():
    initialize_session_state()
    inject_theme_styles(st.session_state.theme)
    render_sidebar()
    render_header()
    st.write("")
    render_metrics_row()
    st.write("")
    
    # Tab Layout Reorganization
    tab_chat, tab_analytics, tab_config = st.tabs([
        "💬 Chat Workspace", 
        "📊 Document Analytics", 
        "⚙️ System Configuration"
    ])
    
    with tab_chat:
        render_upload_panel()
        st.write("")
        render_chat_interface()
        
    with tab_analytics:
        render_analytics_dashboard()
        
    with tab_config:
        render_configuration_panel()

if __name__ == "__main__":
    main()