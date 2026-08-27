"""
Vector Store module managing FAISS index operations, document metadata indexing,
similarity queries, and vector database management.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
import os
import pickle


class FAISSVectorStore:
    """FAISS-backed vector store using Inner Product (Cosine Similarity on normalized vectors)."""

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index.

        Args:
            dimension (int): Vector embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        # IndexFlatIP calculates inner product, which equals Cosine Similarity when vectors are L2-normalized
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Adds vector embeddings and corresponding chunk metadata into the index.

        Args:
            embeddings (np.ndarray): 2D float32 array of shape (N, dimension).
            chunks (List[Dict[str, Any]]): List of chunk metadata objects.
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch between embeddings count ({embeddings.shape[0]}) and chunks count ({len(chunks)}).")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}.")

        # Add to FAISS index
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the vector database for top-k similar chunks.

        Args:
            query_embedding (np.ndarray): 2D float32 array of shape (1, dimension).
            top_k (int): Number of top matches to retrieve.

        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (chunk_dict, similarity_score) tuples.
        """
        if self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            score = float(scores[0][i])
            if idx != -1 and idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx]
                results.append((chunk, score))

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """
        Extracts all stored embedding vectors directly from the FAISS index
        without recomputing or generating duplicate embeddings.

        Returns:
            np.ndarray: 2D float32 array of shape (ntotal, dimension).
        """
        if self.index.ntotal == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            return self.index.reconstruct_n(0, self.index.ntotal).astype(np.float32)
        except Exception:
            return np.empty((0, self.dimension), dtype=np.float32)

    def clear(self) -> None:
        """Clears all vectors and metadata stored in the vector database."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []

    def total_chunks(self) -> int:
        """Returns total number of chunks indexed in the database."""
        return self.index.ntotal

    def save(self, directory: str) -> None:
        """
        Saves the FAISS index and metadata to disk.

        Args:
            directory (str): Directory path to persist vector database files.
        """
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, directory: str) -> bool:
        """
        Loads FAISS index and metadata from disk.

        Args:
            directory (str): Directory path to load from.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.chunks_metadata = pickle.load(f)
            return True
        return False
