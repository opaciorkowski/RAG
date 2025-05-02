import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from backend.rag_pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🧠", layout="wide")

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_links" not in st.session_state:
        st.session_state.pdf_links =[
        'https://assets.pokemon.com/assets/cms2/pdf/trading-card-game/rulebook/sm7_rulebook_en.pdf',
        'https://media.wizards.com/images/magic/tcg/resources/rules/MagicCompRules_21031101.pdf',
        'https://cdn.1j1ju.com/medias/d3/22/83-monopoly-rulebook.pdf',
        'https://fgbradleys.com/wp-content/uploads/rules/Monopoly_Rules.pdf?srsltid=AfmBOorDaiGKyaEWIQFd-au0rl8-tKoqedlzy_6r4EETpj_ZMIUYsNMQ'
        ]
    if "pipeline" not in st.session_state:
        with st.spinner("🔄 Initializing RAG system... Please wait, this might take up to few minutes :)"):
            st.session_state.pipeline = RAGPipeline()
            st.session_state.pipeline.load_documents(st.session_state.pdf_links)
            st.session_state.pipeline.set_memory(True)
            st.session_state.pipeline.set_prompt_type("zero_shot", "Answer only with the content of the documents - otherwise say 'Answer not included in the documents'")
            st.session_state.pipeline.set_query_rewriting(False)
            st.session_state.pipeline.set_use_reranker(False)
            st.session_state.chain = st.session_state.pipeline.get_chain()

def render_sidebar():
    st.image("https://cdn-icons-png.flaticon.com/512/9195/9195256.png", width=200)
    st.markdown("### ⚙️ RAG Configuration")

    query_rewriting = st.checkbox("Rewrite query", value=False)
    prompt_type = st.selectbox(
        "Prompt Strategy",
        options=["zero_shot", "cot", "react", "explain_like_5", "elaborate", "meta"],
    )
    use_reranker = st.checkbox("Use Reranker", value=False)

    st.session_state.pipeline.set_prompt_type(prompt_type, "Answer only with the content of the documents - otherwise say 'Answer not included in the documents'")
    st.session_state.pipeline.set_query_rewriting(query_rewriting)
    st.session_state.pipeline.set_use_reranker(use_reranker)
    st.session_state.chain = st.session_state.pipeline.get_chain()
    
    st.markdown("---")
    st.markdown("### 📄 Add PDF from URL")
    user_pdf_url = st.text_input("Paste a PDF link")
    load_pdf = st.button("👅 Load PDF")

    if load_pdf and user_pdf_url:
        if user_pdf_url not in st.session_state.pdf_links:
            with st.spinner("📄 Indexing your PDF... Please wait."):
                st.session_state.pdf_links.append(user_pdf_url)
                st.cache_resource.clear()
                st.session_state.pipeline.load_documents(st.session_state.pdf_links)
            st.success("✅ PDF added and indexed!")
            st.rerun()
        else:
            st.info("This PDF is already loaded.")

    if st.session_state.pdf_links:
        st.markdown("### ✅ Indexed PDFs")
        for link in st.session_state.pdf_links:
            st.markdown(f"- [{Path(link).name}]({link})")

def render_main_ui():
    st.markdown("""
        <div class="title">🧠 RAG Chat Assistant</div>
        <div class="subtitle">Ask questions grounded in your indexed documents</div>
    """, unsafe_allow_html=True)

def handle_query(query):
    if query:
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message-user'>🗣️{query}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant"):
            chain = st.session_state.chain 
            result = chain({"question": query})
            answer = result.get("answer", "🤔 I couldn't find a good answer.")
            st.markdown(f"<div class='chat-message-assistant'>🤖 {answer}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append((query, answer))
            with st.expander("Sources", expanded=False):
                if "source_documents" in result:
                    seen = set()
                    for i, doc in enumerate(result["source_documents"]):
                        chunk_id = doc.metadata.get("chunk_id") or doc.page_content[:100]
                        if chunk_id in seen:
                            continue
                        seen.add(chunk_id)
                        page = doc.metadata.get("page", "N/A")
                        src = doc.metadata.get("source", doc.metadata.get("parent_source", "unknown"))
                        st.markdown(f"**Source {i+1} — {src}, page {page}**")
                        st.code(doc.page_content.strip()[:500] + ("..." if len(doc.page_content) > 500 else ""), language="text")

def render_chat_history():
    with st.expander("📜 Chat History", expanded=False):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='chat-message-user'><b>Q{i+1}:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message-assistant'><b>A{i+1}:</b> {a}</div>", unsafe_allow_html=True)
            st.divider()

def main():
    init_session_state()
    with st.sidebar:
        render_sidebar()
    render_main_ui()
    query = st.chat_input("Ask me anything about your documents...")
    handle_query(query)
    render_chat_history()

if __name__ == "__main__":
    main()
