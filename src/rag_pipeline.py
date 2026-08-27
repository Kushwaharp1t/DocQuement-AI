"""
End-to-End RAG Pipeline module orchestrating PDF parsing, chunking, vector indexing,
retrieval, prompt construction, LLM completion, and citation tracking using local Ollama (llama3.2).
"""

from typing import List, Dict, Any, Union, BinaryIO, Tuple
import time

from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.prompts import build_rag_prompt, INSUFFICIENT_INFO_PHRASE
from src.llm import OllamaLLM
from src.utils import format_source_citations


class RAGPipeline:
    """Core RAG application orchestrator powered by local Ollama."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
        ollama_base_url: str = None,
        ollama_model: str = None
    ):
        """
        Initialize RAG Pipeline components.

        Args:
            chunk_size (int): Character size per chunk (default 800 for full paragraph context).
            chunk_overlap (int): Overlap character count (default 150).
            top_k (int): Top matching passages to retrieve (default 5).
            ollama_base_url (str): Optional Ollama URL (default http://localhost:11434).
            ollama_model (str): Optional Ollama model tag (default llama3.2).
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISSVectorStore(dimension=384)
        self.retriever = Retriever(self.embedding_generator, self.vector_store)
        self.llm = OllamaLLM(base_url=ollama_base_url, model_name=ollama_model)
        self.top_k = top_k
        self.processed_files: List[str] = []

    def update_config(self, chunk_size: int, chunk_overlap: int, top_k: int) -> None:
        """Dynamically update chunker and retriever parameters."""
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.top_k = top_k

    def process_pdf_inputs(self, pdf_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes uploaded PDFs: extracts text, chunks content, generates embeddings,
        and indexes them into the FAISS vector database.

        Args:
            pdf_inputs: List of dicts e.g. [{"input": file_path_or_bytes, "filename": "doc.pdf"}]

        Returns:
            Dict[str, Any]: Indexing summary metrics.
        """
        start_time = time.perf_counter()

        # Step 1: Extract text page-by-page
        extracted_pages = PDFLoader.load_multiple_pdfs(pdf_inputs)

        if not extracted_pages:
            return {
                "total_pdfs": 0,
                "total_pages": 0,
                "total_chunks": 0,
                "processing_time_sec": round(time.perf_counter() - start_time, 2)
            }

        # Step 2: Intelligent Chunking
        chunks = self.chunker.chunk_pages(extracted_pages)

        # Step 3: Embedding Generation
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

        # Step 4: Index into FAISS Vector Store
        self.vector_store.add_chunks(embeddings, chunks)

        # Record file metadata
        for inp in pdf_inputs:
            fname = inp.get("filename", "uploaded_doc.pdf")
            if fname not in self.processed_files:
                self.processed_files.append(fname)

        elapsed = round(time.perf_counter() - start_time, 2)

        return {
            "total_pdfs": len(pdf_inputs),
            "total_pages": len(extracted_pages),
            "total_chunks": len(chunks),
            "total_indexed_chunks": self.vector_store.total_chunks(),
            "processing_time_sec": elapsed
        }

    def clear_database(self) -> None:
        """Clears the vector store index and resets file tracking."""
        self.vector_store.clear()
        self.processed_files = []

    def answer_question(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        top_k_override: int = None
    ) -> Dict[str, Any]:
        """
        Executes full RAG query cycle: retrieval -> prompt building -> LLM completion -> citation mapping.

        Args:
            query (str): User question.
            chat_history (List[Dict[str, str]]): Conversational history list.
            top_k_override (int): Optional override for top-k retrieval.

        Returns:
            Dict[str, Any]: Dict containing answer, citations, retrieved_chunks, retrieval_time, generation_time.
        """
        k = top_k_override or self.top_k

        # 1. Vector Search & Retrieval
        retrieval_res = self.retriever.retrieve(query=query, top_k=k)
        retrieved_chunks = retrieval_res["results"]
        retrieval_time_ms = retrieval_res["retrieval_time_ms"]

        # If no documents uploaded or no matches found
        if not retrieved_chunks:
            return {
                "answer": INSUFFICIENT_INFO_PHRASE,
                "citations": [],
                "retrieved_chunks": [],
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": 0.0,
                "is_sufficient": False
            }

        # 2. Construct Strict Context Prompt
        prompt = build_rag_prompt(query=query, retrieved_chunks=retrieved_chunks, chat_history=chat_history)

        # 3. LLM Completion Generation via local Ollama
        gen_start = time.perf_counter()
        llm_answer = self.llm.generate_response(prompt)
        generation_time_ms = round((time.perf_counter() - gen_start) * 1000, 2)

        # 4. Format Citations
        citations = format_source_citations(retrieved_chunks)

        # 5. Hallucination Safeguard Verification
        clean_answer = llm_answer.strip()
        is_sufficient = not clean_answer.startswith(INSUFFICIENT_INFO_PHRASE) and len(clean_answer) > 0

        return {
            "answer": llm_answer,
            "citations": citations if is_sufficient else [],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "is_sufficient": is_sufficient
        }
