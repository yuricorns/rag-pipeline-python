import streamlit as st
from docx import Document
import chromadb
from groq import Groq
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: '', 'Inter' ,s new roman, serif !important;
}

/* Background */
.stApp {
    background:040f0f; ;
    color: #d1d5db;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f620, #3b82f620);
    border: 1px solid #7c3aed40;
    color: #a78bfa;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 4 rem;
    font-weight: 300;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 1rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
}
.hero p {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* Upload zone */
.upload-zone {
    background: #0f0f1a;
    border: 1.5px dashed #1e1e2e;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin: 1.5rem 0;
}
.upload-zone:hover {
    border-color: #7c3aed40;
    background: #0f0f20;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin: 1rem 0;
}
.status-ready {
    background: #052e16;
    border: 1px solid #166534;
    color: #4ade80;
}
.status-loading {
    background: #1c1917;
    border: 1px solid #44403c;
    color: #a8a29e;
}

/* Chat messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin: 1.5rem 0;
}
.message-user {
    display: flex;
    justify-content: flex-end;
}
.message-ai {
    display: flex;
    justify-content: flex-start;
}
.bubble-user {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 4px 20px #7c3aed30;
}
.bubble-ai {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    color: #d1d5db;
    padding: 16px 20px;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.7;
}
.bubble-ai strong {
    color: #a78bfa;
}

/* Avatar */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 4px;
}
.avatar-ai {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
}
.avatar-user {
    background: #1e1e2e;
}
.msg-wrapper {
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-chip {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #6b7280;
}
.stat-chip span {
    color: #a78bfa;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e2e, transparent);
    margin: 2rem 0;
}

/* Source chunks */
.chunk-card {
    background: #0a0a12;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c3aed;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}

/* Input styling */
.stTextInput input {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
}
.stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px #7c3aed20 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border-radius: 12px;
    border: 1px dashed #1e1e2e;
    padding: 1rem;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Sidebar content */
.sidebar-section {
    margin-bottom: 2rem;
}
.sidebar-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4b5563;
    margin-bottom: 12px;
}
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #6b7280;
    background: #0a0a12;
    border: 1px solid #1e1e2e;
}
.sidebar-item .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7c3aed;
    flex-shrink: 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #374151;
}
.empty-state .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-state p {
    font-size: 14px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div style='font-size: 32px; font-weight: 600; color: #ffffff; margin-bottom: 4px;'>🧠 DocMind</div>
        <div style='font-size: 18px; color: #4b5563;'>RAG-powered document intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )
    
    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="70b is more accurate, 8b is faster"
    )
    
    top_k = st.slider("Chunks to retrieve", 1, 5, 3, help="More chunks = more context")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">How it works</div>', unsafe_allow_html=True)
    
    steps = [
        ("1", "Upload your document"),
        ("2", "Text split into chunks"),
        ("3", "Chunks stored in ChromaDB"),
        ("4", "Your question gets embedded"),
        ("5", "Similar chunks retrieved"),
        ("6", "LLM answers with context"),
    ]
    for num, step in steps:
        st.markdown(f"""
        <div class="sidebar-item">
            <div class="dot"></div>
            <span style="color: #4b5563; font-size: 11px; font-weight: 600; min-width: 14px;">{num}</span>
            <span>{step}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">Retrieval Augmented Generation</div>
    <h1>Ask anything about<br><span>your documents</span></h1>
    <p>Upload a Word document and get precise, grounded answers powered by Groq LLM and ChromaDB vector search.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Upload section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["docx"],
        help="Word documents (.docx) supported",
        label_visibility="collapsed"
    )

    if uploaded_file and api_key:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                try:
                    # Read document
                    doc = Document(uploaded_file)
                    text = ""
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            text += paragraph.text + ". "

                    # Chunk
                    chunks = []
                    for sentence in text.split("."):
                        sentence = sentence.strip()
                        if len(sentence) > 20:
                            chunks.append(sentence)

                    # Store in ChromaDB
                    chroma_client = chromadb.Client()
                    collection = chroma_client.create_collection(
                        f"doc_{uploaded_file.name.replace(' ', '_')[:20]}"
                    )
                    collection.add(
                        documents=chunks,
                        ids=[str(i) for i in range(len(chunks))]
                    )

                    st.session_state.collection = collection
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.messages = []

                except Exception as e:
                    st.error(f"Error processing document: {e}")

    if st.session_state.doc_name:
        st.markdown(f"""
        <div class="status-pill status-ready">
            ✓ &nbsp; {st.session_state.doc_name} ready
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-chip">Chunks <span>{st.session_state.chunk_count}</span></div>
            <div class="stat-chip">Model <span>llama-3.3-70b</span></div>
            <div class="stat-chip">DB <span>ChromaDB</span></div>
            <div class="stat-chip">Status <span>Ready</span></div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 2rem; margin-bottom: 0.8rem;">📄</div>
            <div style="font-size: 14px; color: #4b5563; margin-bottom: 0.4rem;">Drop your .docx file here</div>
            <div style="font-size: 12px; color: #374151;">Resume, report, manual — anything</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Chat section ──────────────────────────────────────────────────────────────
if not api_key:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔑</div>
        <p>Enter your Groq API key in the sidebar<br>to get started. It's free at console.groq.com</p>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.collection:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📂</div>
        <p>Upload a Word document above<br>to start asking questions about it</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Display messages
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="message-user">
                    <div class="msg-wrapper">
                        <div class="bubble-user">{msg["content"]}</div>
                        <div class="avatar avatar-user">👤</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-ai">
                    <div class="msg-wrapper">
                        <div class="avatar avatar-ai">🧠</div>
                        <div class="bubble-ai">{msg["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Question input
    
        question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            try:
                groq_client = Groq(api_key=api_key)

                # Retrieve
                results = st.session_state.collection.query(
                    query_texts=[question],
                    n_results=top_k
                )
                relevant_chunks = results["documents"][0]
                context = "\n".join(relevant_chunks)

                # Prompt
                prompt = f"""You are a helpful document assistant. Answer the question using only the context below.
If the answer is not found in the context, say "I don't have that information in the document."
Be concise and precise.

Context:
{context}

Question: {question}
Answer:"""

                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Show sources
                with st.expander("📎 Source chunks retrieved", expanded=False):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f'<div class="chunk-card">#{i+1} — {chunk}</div>', unsafe_allow_html=True)

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
