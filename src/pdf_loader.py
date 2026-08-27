"""
PDF Loader module for extracting page-wise text from PDF documents using PyMuPDF (fitz).
Preserves metadata such as filename and 1-based page numbers.
"""

from typing import List, Dict, Any, Union, BinaryIO
import fitz  # PyMuPDF
import os
import io


class PDFLoader:
    """Handles loading and extracting page-level content from PDF files."""

    @staticmethod
    def extract_text_from_file(file_input: Union[str, BinaryIO], filename: str = "") -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF file path or file-like binary stream page by page.

        Args:
            file_input (Union[str, BinaryIO]): File path string or binary file stream.
            filename (str): Name of the file for metadata tagging.

        Returns:
            List[Dict[str, Any]]: List of page dictionaries containing 'text', 'filename', and 'page_number'.
        """
        pages_data = []

        try:
            if isinstance(file_input, str):
                if not os.path.exists(file_input):
                    raise FileNotFoundError(f"PDF file not found: {file_input}")
                doc = fitz.open(file_input)
                if not filename:
                    filename = os.path.basename(file_input)
            elif hasattr(file_input, "read"):
                # Handle bytes / BytesIO / UploadedFile
                stream_bytes = file_input.read()
                # Reset stream pointer if applicable
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                doc = fitz.open(stream=stream_bytes, filetype="pdf")
                if not filename and hasattr(file_input, "name"):
                    filename = getattr(file_input, "name")
            else:
                raise ValueError("Unsupported file input format. Expected file path or binary stream.")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                
                # Only include non-empty pages
                if text:
                    pages_data.append({
                        "text": text,
                        "metadata": {
                            "filename": filename or "document.pdf",
                            "page_number": page_num + 1  # 1-based page numbering
                        }
                    })

            doc.close()

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF '{filename}': {str(e)}") from e

        return pages_data

    @classmethod
    def load_multiple_pdfs(cls, pdf_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Loads multiple PDF inputs and combines page level data.

        Args:
            pdf_inputs: List of dicts e.g. [{"input": file_or_path, "filename": "doc1.pdf"}, ...]

        Returns:
            List[Dict[str, Any]]: Combined list of page records across all files.
        """
        all_pages = []
        for item in pdf_inputs:
            file_inp = item.get("input")
            fname = item.get("filename", "")
            pages = cls.extract_text_from_file(file_inp, filename=fname)
            all_pages.extend(pages)
        return all_pages
