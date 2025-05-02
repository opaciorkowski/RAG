from typing import List, Any
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever

class LLMRerankerRetriever(BaseRetriever):
    def __init__(self, vectorstore: Any, top_k_chunks=20, top_k_parents=4):
        self.top_k_chunks = top_k_chunks
        self.top_k_parents = top_k_parents
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query: str) -> List[Document]:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        relevant_chunks_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k_chunks)
        chunks = [doc for doc, _ in relevant_chunks_with_scores]
        scores = [score for _, score in relevant_chunks_with_scores]

        parent_docs = {}
        for chunk, score in zip(chunks, scores):
            parent_id = chunk.metadata.get("parent_id", f"doc_{len(parent_docs)}")
            if parent_id not in parent_docs:
                parent_docs[parent_id] = {
                    "chunks": [],
                    "scores": [],
                    "source": chunk.metadata.get("parent_source", "unknown")
                }
            parent_docs[parent_id]["chunks"].append(chunk)
            parent_docs[parent_id]["scores"].append(score)

        reranked_parents = []
        for parent_id, parent in parent_docs.items():
            full_text = "\n".join([c.page_content for c in parent["chunks"]])
            try:
                rerank_score = float(self.cross_encoder.predict([(query, full_text)])[0])
            except Exception:
                rerank_score = 0.0
            for c in parent["chunks"]:
                c.metadata["rerank_score"] = rerank_score
            reranked_parents.append({
                "chunks": parent["chunks"],
                "rerank_score": rerank_score
            })

        reranked_parents.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_docs = []
        for parent in reranked_parents[:self.top_k_parents]:
            top_docs.extend(parent["chunks"])

        return top_docs
