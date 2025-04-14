import streamlit as st
from pathlib import Path
import import_ipynb
from advanced_rag import initialize_rag, create_rag_chain, print_result_summary

st.set_page_config(page_title="RAG Chat Assistant", page_icon="🧠", layout="wide")

st.title("🤖 RAG Chat Assistant")

with st.sidebar:
    st.header("⚙️ Configuration")
    use_reranker = st.checkbox("Use Reranker", value=True)
    prompt_type = st.selectbox(
        "Prompt Strategy",
        options=["zero_shot", "cot", "react"],
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask me anything about your documents...")

# Initialize RAG only once and cache it
@st.cache_resource(show_spinner="Loading RAG components...")
def load_rag():
    return initialize_rag()

if "db" not in st.session_state or "llm" not in st.session_state:
    st.session_state.db, st.session_state.llm = load_rag()

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            rag_chain, rag_memory = create_rag_chain(
                vectorstore=st.session_state.db,
                llm=st.session_state.llm,
                prompt_type=prompt_type,
                use_memory=True,
                retriever_k=3,
                use_reranking=use_reranker,
                top_k_chunks=10
            )
            result = rag_chain({"question": query})
            answer = result.get("answer", "🤔 I couldn't find a good answer.")
            st.markdown(answer)
            st.session_state.chat_history.append((query, answer))
        except Exception as e:
            st.error(f"Error: {str(e)}")

with st.expander("📜 Chat History"):
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
