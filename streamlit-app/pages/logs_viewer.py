import streamlit as st
from pathlib import Path

st.set_page_config(page_title="📜 RAG Logs", page_icon="📄", layout="wide")

st.title("📜 Application Logs")
log_file = Path("logs\\app.log")
marker = "STARTING RAG PIPELINE"

if log_file.exists():
    logs = log_file.read_text(encoding="utf-8", errors="replace")
    last_index = logs.rfind(marker)
    if last_index != -1:
        logs = logs[last_index:]
    else:
        logs = "⚠️ Marker not found in logs."

    st.text_area("Log Output (from last pipeline start)", logs.strip(), height=600, key="log_output", disabled=True)
    st.download_button("📥 Download Log File", logs, file_name="app.log")
else:
    st.warning(f"⚠️ Log file not found at: {log_file}")