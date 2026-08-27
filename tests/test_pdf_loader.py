"""
Unit tests for pdf_loader module.
"""

import pytest
import fitz
import io
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_loader import PDFLoader


def create_dummy_pdf_bytes(text_per_page: list) -> bytes:
    """Helper to generate a multi-page PDF in memory."""
    doc = fitz.open()
    for text in text_per_page:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_text_extraction():
    sample_pages = [
        "Company A Placement Policy. Minimum CGPA required is 7.5.",
        "Selection Process: Round 1 Online Test, Round 2 Tech Interview, Round 3 HR."
    ]
    pdf_bytes = create_dummy_pdf_bytes(sample_pages)
    stream = io.BytesIO(pdf_bytes)

    extracted = PDFLoader.extract_text_from_file(stream, filename="Company_A_Policy.pdf")

    assert len(extracted) == 2
    assert extracted[0]["metadata"]["filename"] == "Company_A_Policy.pdf"
    assert extracted[0]["metadata"]["page_number"] == 1
    assert "CGPA required is 7.5" in extracted[0]["text"]

    assert extracted[1]["metadata"]["page_number"] == 2
    assert "Selection Process" in extracted[1]["text"]


def test_load_multiple_pdfs():
    pdf1 = create_dummy_pdf_bytes(["Page 1 of Doc 1"])
    pdf2 = create_dummy_pdf_bytes(["Page 1 of Doc 2"])

    pdf_inputs = [
        {"input": io.BytesIO(pdf1), "filename": "Doc1.pdf"},
        {"input": io.BytesIO(pdf2), "filename": "Doc2.pdf"}
    ]

    all_pages = PDFLoader.load_multiple_pdfs(pdf_inputs)
    assert len(all_pages) == 2
    filenames = [p["metadata"]["filename"] for p in all_pages]
    assert "Doc1.pdf" in filenames
    assert "Doc2.pdf" in filenames
