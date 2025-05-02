import streamlit as st
from pathlib import Path

st.set_page_config(page_title="📜 RAG Logs", page_icon="📄", layout="wide")

st.title("📜 Application Logs")
log_file = Path("logs\\app.log")

if log_file.exists():
    logs = log_file.read_text(encoding="utf-8", errors="replace")
    marker = "STARTING RAG PIPELINE "
    if marker in logs:
        logs = logs.split(marker)[-1]
        logs = marker + logs
    st.text_area("Log Output", logs.strip(), height=600, key="log_output", disabled=True)
    st.download_button("📥 Download Log File", logs, file_name="app.log")
else:
    st.warning(f"⚠️ Log file not found at: {log_file}")
