"""
Integration tests for rag_pipeline module.
"""

import pytest
import io
import fitz
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
from src.prompts import INSUFFICIENT_INFO_PHRASE


def create_sample_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_rag_pipeline_end_to_end_insufficient_info(monkeypatch):
    pipeline = RAGPipeline(chunk_size=300, chunk_overlap=50, top_k=2)

    # Process sample PDF
    pdf_text = "Google Recruitment Notice: Software Engineer role requiring B.Tech CS degree."
    pdf_bytes = create_sample_pdf_bytes(pdf_text)

    summary = pipeline.process_pdf_inputs([
        {"input": io.BytesIO(pdf_bytes), "filename": "Google_Notice.pdf"}
    ])

    assert summary["total_pdfs"] == 1
    assert summary["total_chunks"] >= 1

    # Mock LLM completion to simulate Gemini API returning insufficient info phrase
    monkeypatch.setattr(pipeline.llm, "generate_response", lambda prompt, temperature=0.1: INSUFFICIENT_INFO_PHRASE)

    # Ask unanswerable question to test hallucination safeguard
    result = pipeline.answer_question("What is the dress code for Microsoft interviews?")

    assert result["answer"] == INSUFFICIENT_INFO_PHRASE
    assert result["is_sufficient"] is False


def test_rag_pipeline_clear_db():
    pipeline = RAGPipeline()
    pdf_bytes = create_sample_pdf_bytes("Placement Policy Page")
    pipeline.process_pdf_inputs([{"input": io.BytesIO(pdf_bytes), "filename": "Policy.pdf"}])

    assert pipeline.vector_store.total_chunks() > 0
    pipeline.clear_database()
    assert pipeline.vector_store.total_chunks() == 0
