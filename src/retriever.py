"""
Retriever module bridging embedding generator and vector store for dense similarity search.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore


class Retriever:
    """Retrieves context chunks relevant to user natural language queries."""

    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: FAISSVectorStore):
        """
        Initialize Retriever.

        Args:
            embedding_generator (EmbeddingGenerator): Generator for query embeddings.
            vector_store (FAISSVectorStore): FAISS vector store.
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.last_query_vector: Optional[np.ndarray] = None
        self.last_query_text: Optional[str] = None
        self.last_retrieved_chunk_ids: List[str] = []

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Executes query retrieval pipeline.

        Args:
            query (str): User prompt/question string.
            top_k (int): Number of top matches to retrieve.
            score_threshold (float): Minimum similarity score filter (0.0 to 1.0).

        Returns:
            Dict[str, Any]: Dict containing 'results', 'query_vector', and 'retrieval_time_ms'.
        """
        start_time = time.perf_counter()

        if not query.strip() or self.vector_store.total_chunks() == 0:
            return {
                "results": [],
                "query_vector": None,
                "retrieval_time_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        query_vector = self.embedding_generator.generate_query_embedding(query)
        self.last_query_vector = query_vector
        self.last_query_text = query

        matches = self.vector_store.search(query_vector, top_k=top_k)

        filtered_matches = []
        retrieved_ids = []
        for chunk, score in matches:
            if score >= score_threshold:
                chunk_meta = chunk.get("metadata", {})
                chunk_id = chunk_meta.get("chunk_id", chunk.get("chunk_id", ""))
                retrieved_ids.append(chunk_id)
                filtered_matches.append({
                    "chunk": chunk,
                    "score": round(score, 4),
                    "score_percentage": f"{max(0.0, min(100.0, score * 100)):.1f}%"
                })

        self.last_retrieved_chunk_ids = retrieved_ids
        end_time = time.perf_counter()

        return {
            "results": filtered_matches,
            "query_vector": query_vector,
            "retrieval_time_ms": round((end_time - start_time) * 1000, 2)
        }
