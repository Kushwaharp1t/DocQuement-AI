"""
Unit tests for vector_visualization module.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_visualization import render_vector_visualization, VISUALIZATION_EXPLANATION


def test_visualization_explanation_string():
    assert "PCA projects" in VISUALIZATION_EXPLANATION
    assert "high-dimensional" in VISUALIZATION_EXPLANATION


def test_visualization_empty_embeddings_handling():
    # Verify handles empty arrays gracefully without crashing
    render_vector_visualization(
        embeddings=np.empty((0, 384), dtype=np.float32),
        chunks_metadata=[]
    )


def test_visualization_single_chunk_edge_case():
    single_emb = np.random.rand(1, 384).astype(np.float32)
    single_meta = [{"chunk_id": "c1", "text": "Sample text", "metadata": {"filename": "doc.pdf", "page_number": 1}}]
    
    # Should warn about requiring at least 2 samples for PCA projection without erroring
    render_vector_visualization(
        embeddings=single_emb,
        chunks_metadata=single_meta
    )
