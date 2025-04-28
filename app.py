import streamlit as st
from pathlib import Path
import import_ipynb
from advanced_rag import initialize_rag, create_rag_chain

st.set_page_config(page_title="RAG Chat Assistant", page_icon="🧠", layout="wide")

# --- Session Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_links" not in st.session_state:
    st.session_state.pdf_links = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9195/9195256.png", width=200)
    st.markdown("### ⚙️ RAG Configuration")

    query_rewriting = st.checkbox("Rewrite query", value=True)
    use_reranker = st.checkbox("Use Reranker", value=True)

    prompt_type = st.selectbox(
        "Prompt Strategy",
        options=["zero_shot", "cot", "react", "explain_like_5", "elaborate", "meta"],
    )

    st.markdown("---")
    st.markdown("### 📄 Add PDF from URL")
    user_pdf_url = st.text_input("Paste a PDF link")
    load_pdf = st.button("📥 Load PDF")

    if load_pdf and user_pdf_url:
        if user_pdf_url not in st.session_state.pdf_links:
            with st.spinner("📄 Indexing your PDF... Please wait."):
                st.session_state.pdf_links.append(user_pdf_url)
                st.cache_resource.clear()
                st.session_state.db, st.session_state.llm = initialize_rag(new_docs=st.session_state.pdf_links)
            st.success("✅ PDF added and indexed!")
            st.rerun()
        else:
            st.info("This PDF is already loaded.")

    if st.session_state.pdf_links:
        st.markdown("### ✅ Indexed PDFs")
        for link in st.session_state.pdf_links:
            st.markdown(f"- [{Path(link).name}]({link})")

# --- Main UI ---
st.markdown("""
    <div class="title">🧠 RAG Chat Assistant</div>
    <div class="subtitle">Ask questions grounded in your indexed documents</div>
""", unsafe_allow_html=True)

# --- Load RAG on first run ---
if "db" not in st.session_state or "llm" not in st.session_state:
    with st.spinner("🔄 Initializing RAG system... Please wait."):
        st.session_state.db, st.session_state.llm = initialize_rag(new_docs=st.session_state.pdf_links)

# --- Chat Interaction ---
query = st.chat_input("Ask me anything about your documents...")

if query:
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message-user'>🗣️{query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            rag_chain, rag_memory = create_rag_chain(
                vectorstore=st.session_state.db,
                llm=st.session_state.llm,
                prompt_type=prompt_type,
                use_memory=True,
                retriever_k=3,
                use_reranking=use_reranker,
                rewrite=query_rewriting,
                top_k_chunks=10
            )
            result = rag_chain({"question": query})
            answer = result.get("answer", "🤔 I couldn't find a good answer.")
            st.markdown(f"<div class='chat-message-assistant'>🤖 {answer}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append((query, answer))
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --- Chat History ---
with st.expander("📜 Chat History", expanded=False):
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"<div class='chat-message-user'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message-assistant'><b>A{i+1}:</b> {a}</div>", unsafe_allow_html=True)
        st.divider()