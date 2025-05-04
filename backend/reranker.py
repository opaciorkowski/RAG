from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from backend.logging import get_logger 

def create_parent_document_llm_reranker(vectorstore, top_k_chunks=20, top_k_parents=4):
    class LLMRerankerRetriever(BaseRetriever):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "logger", get_logger(self.__class__.__name__))
            self.logger.info("LLMRerankerRetriever initialized.")

        def _get_relevant_documents(self, query: str) -> List[Document]:
            self.logger.info(f"Starting reranked retrieval for query: {query}")
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            results = vectorstore.similarity_search_with_score(query, k=top_k_chunks)
            self.logger.info(f"Retrieved {len(results)} chunks from vectorstore.")

            chunks = [doc for doc, _ in results]
            scores = [score for _, score in results]

            parent_docs = {}
            for chunk, score in zip(chunks, scores):
                parent_id = chunk.metadata.get("parent_id") or chunk.metadata.get("doc_id", f"doc_{len(parent_docs)}")
                if parent_id not in parent_docs:
                    parent_docs[parent_id] = {
                        "chunks": [],
                        "scores": [],
                        "source": chunk.metadata.get("parent_source", "unknown")
                    }
                parent_docs[parent_id]["chunks"].append(chunk)
                parent_docs[parent_id]["scores"].append(score)

            self.logger.info(f"Grouped chunks into {len(parent_docs)} parent documents.")

            reranked = []
            for parent_id, parent in parent_docs.items():
                parent["chunks"].sort(key=lambda c: (c.metadata.get("page", 0), c.metadata.get("chunk_index", 0)))
                full_text = "\n".join(chunk.page_content for chunk in parent["chunks"])

                try:
                    rerank_score = float(cross_encoder.predict([(query, full_text)])[0])
                except Exception as e:
                    self.logger.warning(f"CrossEncoder prediction failed for parent {parent_id}: {e}")
                    rerank_score = 0.0

                self.logger.info(f"Rerank score for parent {parent_id}: {rerank_score:.4f}")

                for chunk in parent["chunks"]:
                    chunk.metadata["rerank_score"] = rerank_score

                reranked.append({
                    "id": parent_id,
                    "chunks": parent["chunks"],
                    "rerank_score": rerank_score,
                    "source": parent["source"]
                })

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

            top_docs = []
            for i, parent in enumerate(reranked[:top_k_parents]):
                self.logger.info(f"Selected parent {i+1}: {parent['id']} (Score: {parent['rerank_score']:.4f})")
                top_docs.extend(parent["chunks"])

            self.logger.info(f"Returning {len(top_docs)} top-ranked chunks.")
            return top_docs

        async def _aget_relevant_documents(self, query: str):
            raise NotImplementedError("Async version is not implemented.")

    return LLMRerankerRetriever()
