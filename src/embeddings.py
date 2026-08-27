"""
Embedding Generator module using Sentence Transformers (all-MiniLM-L6-v2).
Provides dense vector representations with L2 normalization for vector similarity search.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Encapsulates HuggingFace SentenceTransformer embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name (str): SentenceTransformers model identifier.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model to save startup resources."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates L2-normalized float32 numpy array embeddings for a list of text strings.

        Args:
            texts (List[str]): Input texts.

        Returns:
            np.ndarray: 2D float32 numpy array of shape (N, embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Generate embeddings as float32 numpy array
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Ensure L2 normalization for Cosine Similarity
        )

        return embeddings.astype(np.float32)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates L2-normalized float32 embedding vector for a single query.

        Args:
            query (str): Input query text.

        Returns:
            np.ndarray: 2D numpy array of shape (1, embedding_dim).
        """
        embeddings = self.generate_embeddings([query])
        return embeddings
