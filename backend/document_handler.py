
import os
import requests
from typing import List, Union
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.logging import get_logger

class DocumentHandler:
    def __init__(self, folder: str = "documents"):
        self.logger = get_logger(self.__class__.__name__)
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.logger.info(f"DocumentHandler initialized with folder: {self.folder}")

    def _download_pdf(self, url: str) -> tuple[str, bool]:
        filename = os.path.basename(url.split('?')[0])
        filepath = os.path.join(self.folder, filename)
        if os.path.exists(filepath):
            self.logger.info(f"PDF already exists: {filepath}, skipping download.")
            return filepath, False
        self.logger.info(f"Downloading PDF from {url} to {filepath}.")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        self.logger.info(f"Downloaded PDF to {filepath}.")
        return filepath, True

    def load_and_chunk_pdfs(self, documents: Union[List[str], List[Document]], chunk_size: int = 800, chunk_overlap: int = 80) -> List[Document]:
            self.logger.info("Starting PDF loading and chunking.")
            if isinstance(chunk_size, str):
                chunk_size = int(chunk_size)
            if isinstance(chunk_overlap, str):
                chunk_overlap = int(chunk_overlap)

            if documents and isinstance(documents[0], str):
                loaded_docs = []
                for url in documents:
                    try:
                        pdf_path, downloaded = self._download_pdf(url)
                        if downloaded:
                            pdf_docs = PyPDFLoader(pdf_path).load()
                            loaded_docs.extend(pdf_docs)
                    except Exception as e:
                        self.logger.warning(f"Skipping document due to error: {e}")
                documents = loaded_docs

            def get_filename(path):
                if not path or path == "unknown":
                    return "unknown_document"
                return os.path.basename(path).split('.')[0]

            source_groups = {}
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                source_groups.setdefault(source, []).append(doc)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            all_chunks = []
            for source, docs in source_groups.items():
                docs.sort(key=lambda x: x.metadata.get("page", 0))
                parent_id = get_filename(source)
                chunks = splitter.split_documents(docs)

                for i, chunk in enumerate(chunks):
                    page_num = chunk.metadata.get("page", 0)
                    chunk_id = f"{parent_id}_p{page_num}_c{i}"
                    chunk.metadata.update({
                        "parent_id": parent_id,
                        "parent_source": source,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    all_chunks.append(chunk)
            self.logger.info(f"Generated {len(all_chunks)} chunks from documents.")
            return all_chunks
