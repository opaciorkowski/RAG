import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from backend.document_handler import DocumentHandler
from backend.vectorstore_manager import VectorstoreManager
from prompts.prompt_manager import PromptManager
from backend.reranker import create_parent_document_llm_reranker
from backend.logging import get_logger
from sentence_transformers import CrossEncoder
from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("================================================================")
        self.logger.info("STARTING RAG PIPELINE")
        self.logger.info("================================================================")
        self.logger.info("Initializing RAGPipeline...")
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = self._load_llm()
        self.vectorstore = None
        self.doc_handler = DocumentHandler()
        self.prompt_manager = PromptManager()
        self.retriever = None
        self.memory = None
        self.rewrite = False
        self.prompt_type = "zero_shot"
        self.additional_prompt_instruction = None
        self.use_reranking = False
        self.retriever_k = 4
        self.top_k_chunks = 20
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def _load_llm(self):
        self.logger.info("Loading LLM...")
        if os.getenv("GEMINI_API_KEY"):
            self.logger.info("Using Gemini model.")
            return GoogleGenerativeAI(api_key=os.getenv("GEMINI_API_KEY"), model="gemini-2.0-flash")
        elif os.getenv("OPENAI_API_KEY"):
            self.logger.info("Using OpenAI model.")
            return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0.0)
        raise ValueError("Invalid LLM configuration. Please set OPENAI_API_KEY or GEMINI_API_KEY.")

    def _update_retriever(self):
        if self.vectorstore is None:
            self.logger.error("Vectorstore is not initialized.")
            raise ValueError("Vectorstore is not initialized. Please load documents first.")

        if self.use_reranking:
            self.logger.info("Using reranker retriever.")
            self.retriever = create_parent_document_llm_reranker(
                vectorstore=self.vectorstore,
                top_k_chunks=self.top_k_chunks,
                top_k_parents=self.retriever_k
            )
        else:
            self.logger.info(f"Using standard retriever with k={self.retriever_k}.")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retriever_k}
            )

    def load_documents(self, urls: str):
        self.logger.info(f"Loading and chunking documents from {len(urls)} URLs...")
        chunks = self.doc_handler.load_and_chunk_pdfs(urls)
        self.vectorstore = VectorstoreManager(self.embedding_model).store_documents(chunks)
        self.logger.info("Documents stored in vectorstore.")
        self._update_retriever()

    def set_prompt_type(self, prompt_type: str, additional_instruction: str = None):
        self.logger.info(f"Setting prompt type: {prompt_type}")
        self.prompt_type = prompt_type
        self.additional_prompt_instruction = additional_instruction

    def set_use_reranker(self, enabled: bool):
        self.logger.info(f"Setting reranker: {enabled}")
        self.use_reranking = enabled
        self._update_retriever()

    def set_memory(self, enabled: bool):
        self.logger.info(f"Setting memory: {enabled}")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        ) if enabled else None

    def set_query_rewriting(self, enabled: bool):
        self.logger.info(f"Setting query rewriting: {enabled}")
        self.rewrite = enabled

    def get_chain(self):
        self.logger.info(f"Building RAG chain with prompt: {self.prompt_type}")
        template = self.prompt_manager.get_prompt(self.prompt_type).template
        if self.additional_prompt_instruction:
            template += f"\n\nInstruction: {self.additional_prompt_instruction}"
        prompt = PromptTemplate.from_template(template)

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            memory=self.memory,
            return_source_documents=True
        )

        if self.rewrite:
            self.logger.info("Wrapping chain with query rewriting.")
            def wrapped(inputs):
                inputs["question"] = self.prompt_manager.rewrite_query(inputs["question"], self.llm, self.retriever)
                self.logger.info(f"Rewritten query: {inputs['question']}")
                return chain.invoke(inputs)
            return wrapped

        self.logger.info("Returning standard chain.")
        return chain