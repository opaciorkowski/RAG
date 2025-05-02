import os
from pathlib import Path
from typing import List, Union
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from backend.logging import get_logger

class VectorstoreManager:
    def __init__(self, embedding_model, persist_directory: str = "faiss_index"):
        self.logger = get_logger(self.__class__.__name__)
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        
    def store_documents(self, chunks: List[Document]) -> FAISS:
        self.logger.info("Starting document storage process.")

        existing_ids = set()
        vectorstore = None

        if Path(self.persist_directory).exists():
            vectorstore = FAISS.load_local(self.persist_directory, self.embedding_model, allow_dangerous_deserialization=True)
            self.logger.info("Loaded existing vectorstore from faiss_index.")
            existing_ids = {
                doc.metadata.get("parent_id")
                for doc in vectorstore.docstore._dict.values()
                if "parent_id" in doc.metadata
            }

        new_chunks = [c for c in chunks if c.metadata.get("parent_id") not in existing_ids]

        if not new_chunks:
            self.logger.info("No new documents to embed. Skipping storage.")
            return vectorstore

        if vectorstore:
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, self.embedding_model)

        vectorstore.save_local(self.persist_directory)
        self.logger.info(f"Stored {len(new_chunks)} new document chunks in vectorstore.")
        return vectorstore
