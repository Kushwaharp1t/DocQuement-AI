"""
Intelligent Document Chunker module for splitting text into overlapping chunks
while preserving page and document metadata.
"""

from typing import List, Dict, Any
import re


class TextChunker:
    """Splits page text into configurable chunks with sentence boundary awareness."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the TextChunker.

        Args:
            chunk_size (int): Target character count per chunk.
            chunk_overlap (int): Overlap character count between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks prioritizing sentence boundaries.

        Args:
            text (str): Source text string.

        Returns:
            List[str]: List of string chunks.
        """
        if not text or not text.strip():
            return []

        # Split into sentences or lines
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Handle case where a single sentence exceeds chunk size
            if sentence_len > self.chunk_size:
                # If current chunk has content, commit it first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0

                # Force split long sentence by character sliding window
                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < sentence_len:
                    end = start + self.chunk_size
                    chunks.append(sentence[start:end])
                    start += step
                continue

            if current_len + sentence_len + (1 if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(sentence)
                current_len += sentence_len + (1 if current_chunk else 0)
            else:
                chunks.append(" ".join(current_chunk))

                # Handle overlap by retaining recent sentences from current_chunk
                overlap_chunk = []
                overlap_len = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_len + len(prev_sentence) + 1 <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_len += len(prev_sentence) + 1
                    else:
                        break

                current_chunk = overlap_chunk + [sentence]
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks page data records while keeping rich metadata.

        Args:
            pages_data (List[Dict[str, Any]]): Extracted pages from PDFLoader.

        Returns:
            List[Dict[str, Any]]: List of chunk dicts containing metadata and text.
        """
        all_chunks = []
        global_chunk_idx = 0

        for page in pages_data:
            text = page.get("text", "")
            metadata = page.get("metadata", {})
            filename = metadata.get("filename", "unknown.pdf")
            page_number = metadata.get("page_number", 1)

            str_chunks = self.split_text(text)

            for local_idx, chunk_text in enumerate(str_chunks):
                global_chunk_idx += 1
                chunk_id = f"{filename}_p{page_number}_c{local_idx + 1}"
                
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "global_idx": global_chunk_idx,
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "page_number": page_number,
                        "local_chunk_idx": local_idx + 1
                    }
                })

        return all_chunks
