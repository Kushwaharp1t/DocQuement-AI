"""
Unit tests for FAISS vector store module.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import FAISSVectorStore


def test_vector_store_add_and_search():
    store = FAISSVectorStore(dimension=4)

    # Unit vectors
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    chunks = [
        {"chunk_id": "c1", "text": "First chunk text", "metadata": {"filename": "doc1.pdf", "page_number": 1}},
        {"chunk_id": "c2", "text": "Second chunk text", "metadata": {"filename": "doc2.pdf", "page_number": 2}}
    ]

    embeddings = np.vstack([v1, v2])
    store.add_chunks(embeddings, chunks)

    assert store.total_chunks() == 2

    # Query with vector matching v1
    query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    # L2 normalize query
    query = query / np.linalg.norm(query)

    results = store.search(query, top_k=2)

    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == "c1"
    assert score > 0.8


def test_vector_store_clear():
    store = FAISSVectorStore(dimension=4)
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store.add_chunks(v1, [{"chunk_id": "c1", "text": "sample"}])

    assert store.total_chunks() == 1
    store.clear()
    assert store.total_chunks() == 0
