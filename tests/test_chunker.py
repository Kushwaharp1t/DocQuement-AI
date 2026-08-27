"""
Unit tests for chunker module.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunker import TextChunker


def test_chunker_basic_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "Students must maintain a minimum CGPA of 7.5 to be eligible for campus drives. "
        "No active backlogs are allowed at the time of recruitment. "
        "Dress code for interviews is strictly formal."
    )
    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # Max length bound considering sentence integrity


def test_chunker_metadata_preservation():
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    pages_data = [
        {
            "text": "Placement policy rule 1: Minimum CGPA is 8.0. Rule 2: Attendance must be 85%.",
            "metadata": {"filename": "Placement_Policy.pdf", "page_number": 3}
        }
    ]
    chunk_records = chunker.chunk_pages(pages_data)

    assert len(chunk_records) >= 1
    first_chunk = chunk_records[0]
    assert first_chunk["metadata"]["filename"] == "Placement_Policy.pdf"
    assert first_chunk["metadata"]["page_number"] == 3
    assert "chunk_id" in first_chunk["metadata"]
    assert first_chunk["metadata"]["chunk_id"].startswith("Placement_Policy.pdf_p3")


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
