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
### 0. Setup Venv
```bash
python -m venv rag_venv
rag_venv\Scripts\activate.bat
```
### 1. Install dependencies
This might take a while
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file (only one api-key is needed):
```env
GEMINI_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
STREAMLIT_WATCH_USE_POLLING=true
```

### 3. Run the app
```bash
streamlit run streamlit-app/app.py
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
- Check logs — see what is going on inside the app
- Import documents — test RAG on your own PDFs

---

## Tech Stack

| Component        | Tool                          |
|------------------|-------------------------------|
| LLM              | Gemini 2.0 Flash / gpt-4.1-mini|
| Embedding Model  | text-embedding-ada-002        |
| Vector DB        | FAISS                         |
| UI               | Streamlit                     |
| Framework        | LangChain                     |

---

## Preloaded Docs

- Pokémon TCG Rulebook  
- Monopoly (Classic)  
- Magic the Gathering Comprehensive Rules  
 
 ---

## RAGAS Evaluation 
You can evaluate your RAG pipeline using the included notebook `ragas_evaluation.ipynb` located in the `notebooks` folder.
### What it does:
- Runs quality checks on your RAG answers
- Scores them for relevance, correctness, and source grounding
- Helps you improve prompt choice, chunking, and retrievers

### How to use:
- Open ragas_evaluation.ipynb in Jupyter or VS Code
- Fill in test questions and expected answers
- Run the notebook to get evaluation scores

Sample questions with answers are provided in `notebooks/qa_eval.json`