"""
Unit tests for embedding generator module.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import EmbeddingGenerator


def test_embedding_generation_shape():
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    texts = [
        "Software Engineering Job Description",
        "Minimum CGPA eligibility requirement is 7.5"
    ]
    embeddings = generator.generate_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization (norm should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert pytest.approx(norm, 0.01) == 1.0


def test_query_embedding_shape():
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    query_vec = generator.generate_query_embedding("What is the CGPA criteria?")
    assert query_vec.shape == (1, 384)
