# DocQuement AI

An enterprise-ready, local Retrieval-Augmented Generation (RAG) system for analyzing documents with page-level citations, strict hallucination prevention, local Ollama (Llama 3.2) inference, FAISS dense vector search, and interactive 2D PCA vector space visualization.

---

## Overview

DocQuement AI allows users to upload PDF documents (such as placement notices, company job descriptions, and eligibility policies) and query them using natural language. The architecture is engineered to run **100% locally**, combining local LLM completion via Ollama with HuggingFace embeddings and FAISS vector indexing.

Key capabilities include:
- **Verifiable Source Attribution**: Direct document name and page number citations for every generated answer.
- **Hallucination Prevention**: Grounding directives that instruct the LLM to return a strict fallback response when context is absent.
- **2D Embedding Space Visualization**: Real-time Principal Component Analysis (PCA) projection of 384-dimensional chunk embeddings into an interactive Plotly map.

---

## RAG Architecture & System Design

```
       [ Upload PDFs ]
              │
     [ PyMuPDF Text Extractor ]
              │
  [ Sentence-Aware Sliding Chunker ]
              │
   [ HuggingFace MiniLM-L6-v2 ]
              │
     [ FAISS Vector Store Index ] ───→ [ Extract Vectors (reconstruct_n) ]
              │                                      │
    [ Natural Language Query ]            [ PCA 2D Dimensionality Reduction ]
              │                                      │
  [ Convert Query to Vector Embedding ]   [ Plotly 2D Interactive Scatter Map ]
              │
  [ Top-K Cosine Similarity Search ]
              │
 [ Prompt Assembly + Context Snippets ]
              │
  [ Local Ollama (Llama 3.2) Engine ]
              │
 [ Answer + Page-Level Source Citations ]
```

### Component Breakdown

1. **Document Ingestion (`src/pdf_loader.py`)**: Parses PDF files page-by-page using PyMuPDF (`fitz`), capturing document basenames and 1-based page numbers.
2. **Text Chunking (`src/chunker.py`)**: Splits document text using a sentence-boundary sliding window (default `chunk_size=800`, `chunk_overlap=150`) to preserve context integrity.
3. **Vector Embeddings (`src/embeddings.py`)**: Generates 384-dimensional dense float32 vectors using HuggingFace `sentence-transformers/all-MiniLM-L6-v2` with L2 normalization.
4. **Dense Vector Store (`src/vector_store.py`)**: Indexes embeddings using `faiss.IndexFlatIP` for inner product (cosine similarity) calculation.
5. **Context Retriever (`src/retriever.py`)**: Transforms queries into dense vectors and queries FAISS for top-$k$ nearest context passages.
6. **Local LLM Engine (`src/llm.py`)**: Interfaces with a local Ollama server running `llama3.2` at `http://localhost:11434`.
7. **2D Vector Space Map (`src/vector_visualization.py`)**: Reuses indexed vectors directly from FAISS and applies PCA to project high-dimensional embeddings into a 2D Plotly map.

---

## Project Structure

```
DocQuement-AI/
├── app.py                     # Streamlit frontend application
├── requirements.txt           # Dependency specifications
├── .env.example               # Environment variables configuration
├── README.md                  # System architecture documentation
│
├── data/
│   └── uploaded_pdfs/         # Directory for document uploads
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py          # PDF text extraction
│   ├── chunker.py             # Sentence-aware text chunking
│   ├── embeddings.py          # Sentence Transformers embedding engine
│   ├── vector_store.py        # FAISS vector store manager
│   ├── retriever.py           # Vector similarity retriever
│   ├── rag_pipeline.py        # End-to-end RAG orchestrator
│   ├── llm.py                 # Ollama API client wrapper
│   ├── prompts.py             # Context prompt formatting
│   ├── vector_visualization.py# PCA 2D Plotly scatter visualization
│   └── utils.py               # Citation formatting & JSON exporters
│
└── tests/
    ├── test_pdf_loader.py
    ├── test_chunker.py
    ├── test_embeddings.py
    ├── test_vector_store.py
    ├── test_rag_pipeline.py
    └── test_vector_visualization.py
```

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Runtime environment |
| **Interface** | Streamlit | Web frontend |
| **Extraction** | PyMuPDF (`fitz`) | PDF text parsing |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` dense vector model |
| **Vector DB** | FAISS (`faiss-cpu`) | Similarity indexing |
| **Visualization** | `plotly`, `scikit-learn` | 2D PCA projection |
| **LLM Inference** | Ollama (`llama3.2`) | Local LLM completion |
| **Utilities** | `numpy`, `pandas`, `requests` | Data processing & HTTP requests |

---

## Quick Start Guide

### Prerequisites
- Python 3.11 or higher
- [Ollama](https://ollama.com/) installed on local host

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kushwaharp1t/DocQuement-AI.git
   cd DocQuement-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama model:
   ```bash
   ollama run llama3.2
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```

The application will be accessible at `http://localhost:8501`.

---

## Verification & Testing

Execute the test suite using pytest:
```bash
python -m pytest tests/ -v
```

Output:
```
14 passed in 42.83s
```
