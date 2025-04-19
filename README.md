# RAG Chat Assistant

A fully interactive, dark-themed Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, powered by LangChain, Gemini / GPT, and FAISS. Paste PDF links and chat with your documents in real time.

![UI](images/ui.jpg)

---

## Features

- Sleek, dark UI with custom chat bubbles  
- Paste public PDF URLs to index and query instantly  
- Multi-turn memory-powered conversations  
- Prompt strategies: `zero_shot`, `cot`, `react`, `elaborate`, `explain_like_5`, `meta`  
- FAISS vector search + optional CrossEncoder reranking  
- Duplicate document skipping  
- Gemini 2.0 Flash or GPT-3.5 support  
- Expandable chat history  

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

For Gemini users:
```bash
pip install langchain-google-genai
```

### 2. Configure environment variables

Create a `.env` file:
```env
GEMINI_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## How to Use

- Paste a public PDF link into the sidebar  
- The app will download, chunk, and index it  
- Start chatting naturally!

Example queries:
- `What are the win conditions in Monopoly?`
- `Explain Pokémon energy cards like I'm five`
- `What is the stack rule in Magic the Gathering?`

---

## Sidebar Controls

- Rewrite query — optimize user input before retrieval  
- Use reranker — reorders context chunks by relevance  
- Prompt style — control reasoning: step-by-step, CoT, ReAct  

---

## Tech Stack

| Component        | Tool                          |
|------------------|-------------------------------|
| LLM              | Gemini / OpenAI GPT           |
| Embedding Model  | HuggingFace Transformers      |
| Vector DB        | FAISS                         |
| Reranker         | Sentence Transformers         |
| UI               | Streamlit                     |
| Framework        | LangChain                     |

---

## Preloaded Docs

- Pokémon TCG Rulebook  
- Monopoly (Classic)  
- Magic the Gathering Comprehensive Rules  
 
 ---


