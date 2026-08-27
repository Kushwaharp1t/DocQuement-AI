"""
DocQuement AI
Streamlit Web Application Entrypoint (100% Pure Local Ollama llama3.2 RAG + Vector Visualization).
"""

import streamlit as st
import os
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline
from src.utils import export_chat_history_json, export_chat_history_markdown
from src.prompts import INSUFFICIENT_INFO_PHRASE
from src.vector_visualization import render_vector_visualization

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="DocQuement AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "rag_pipeline" not in st.session_state or not hasattr(st.session_state.rag_pipeline.llm, "is_available"):
    st.session_state.rag_pipeline = RAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_analytics" not in st.session_state:
    st.session_state.last_analytics = {
        "total_pdfs": 0,
        "total_chunks": 0,
        "retrieval_time_ms": 0.0,
        "generation_time_ms": 0.0
    }

if "processed_files_list" not in st.session_state:
    st.session_state.processed_files_list = []


# ==============================================================================
# SIDEBAR CONTROLS & ANALYTICS
# ==============================================================================
with st.sidebar:
    st.title("Control Panel")

    # Local Ollama Setup
    st.subheader("Local Ollama Setup")
    
    ollama_url = st.text_input(
        "Ollama Server URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Local URL of your running Ollama instance."
    )
    ollama_model = st.text_input(
        "Ollama Model Tag",
        value=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Target local model e.g. llama3.2, mistral, etc."
    )

    # Sync configuration
    st.session_state.rag_pipeline.llm.base_url = ollama_url.rstrip('/')
    st.session_state.rag_pipeline.llm.model_name = ollama_model
    st.session_state.rag_pipeline.llm._init_client()

    # Connection Status Check
    is_connected = st.session_state.rag_pipeline.llm.is_available()
    if is_connected:
        st.success(f"Ollama Online ({ollama_model})")
    else:
        st.error(
            f"Ollama Disconnected at `{ollama_url}`\n\n"
            f"1. Install Ollama from [https://ollama.com](https://ollama.com)\n"
            f"2. Run `ollama run {ollama_model}` in your terminal."
        )

    st.divider()

    # Document Upload Section
    st.subheader("Upload Placement Documents")
    uploaded_files = st.file_uploader(
        "Upload Job Descriptions, Policies, Notices (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to build the retrieval index."
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files_list]
        if new_files:
            with st.spinner("Extracting text, chunking, and embedding into FAISS..."):
                pdf_inputs = [{"input": f, "filename": f.name} for f in uploaded_files]
                summary = st.session_state.rag_pipeline.process_pdf_inputs(pdf_inputs)

                # Update session state metrics
                st.session_state.processed_files_list = st.session_state.rag_pipeline.processed_files
                st.session_state.last_analytics["total_pdfs"] = len(st.session_state.processed_files_list)
                st.session_state.last_analytics["total_chunks"] = summary["total_indexed_chunks"]

                st.success(f"Indexed {summary['total_chunks']} chunks from {summary['total_pdfs']} PDF(s)!")
        elif st.button("Re-index Documents", use_container_width=True):
            with st.spinner("Re-indexing documents..."):
                st.session_state.rag_pipeline.clear_database()
                pdf_inputs = [{"input": f, "filename": f.name} for f in uploaded_files]
                summary = st.session_state.rag_pipeline.process_pdf_inputs(pdf_inputs)
                st.session_state.processed_files_list = st.session_state.rag_pipeline.processed_files
                st.session_state.last_analytics["total_pdfs"] = len(st.session_state.processed_files_list)
                st.session_state.last_analytics["total_chunks"] = summary["total_indexed_chunks"]
                st.success(f"Indexed {summary['total_chunks']} chunks!")

    # Indexed Documents List
    if st.session_state.processed_files_list:
        st.markdown("**Indexed Files:**")
        for fname in st.session_state.processed_files_list:
            st.markdown(f"- `{fname}`")
    else:
        st.info("No documents indexed yet. Upload PDFs above.")

    st.divider()

    # Hyperparameter Settings
    st.subheader("RAG Configuration")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1200, value=800, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=150, step=10)
    top_k = st.slider("Top-K Retrieved Passages", min_value=1, max_value=10, value=5, step=1)

    # Update pipeline hyperparams
    st.session_state.rag_pipeline.update_config(chunk_size, chunk_overlap, top_k)

    st.divider()

    # Analytics Dashboard Panel
    st.subheader("Performance Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Total PDFs", st.session_state.last_analytics["total_pdfs"])
    m2.metric("Total Chunks", st.session_state.last_analytics["total_chunks"])

    m3, m4 = st.columns(2)
    m3.metric("Retrieval Time", f"{st.session_state.last_analytics['retrieval_time_ms']:.1f} ms")
    m4.metric("Ollama Latency", f"{st.session_state.last_analytics['generation_time_ms']:.1f} ms")

    st.divider()

    # Actions Section: Clear DB & Download Transcript
    st.subheader("Actions")
    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("Clear DB", use_container_width=True, help="Wipe FAISS vector database and clear file cache."):
            st.session_state.rag_pipeline.clear_database()
            st.session_state.processed_files_list = []
            st.session_state.last_analytics["total_pdfs"] = 0
            st.session_state.last_analytics["total_chunks"] = 0
            st.success("Database cleared!")
            st.rerun()

    with col_export:
        if st.session_state.chat_history:
            json_export = export_chat_history_json(st.session_state.chat_history)
            st.download_button(
                label="Export Chat",
                data=json_export,
                file_name="placement_chat_history.json",
                mime="application/json",
                use_container_width=True
            )


# ==============================================================================
# MAIN DISPLAY & TABS INTERFACE
# ==============================================================================

# Header Section
st.title("DocQuement AI")
st.caption("100% Local RAG Architecture • **Ollama (llama3.2)** + Sentence Transformers + FAISS.")

# Tabbed Interface: Chat Assistant & Vector Visualization
tab_chat, tab_viz = st.tabs(["Chat Assistant", "Vector Visualization"])

# ------------------------------------------------------------------------------
# TAB 1: CHAT ASSISTANT
# ------------------------------------------------------------------------------
with tab_chat:
    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=None):
            st.markdown(message["content"])

            # Render citations if present
            if "citations" in message and message["citations"]:
                with st.expander("View Source Citations & Scores", expanded=False):
                    for cite in message["citations"]:
                        st.markdown(f"**Document:** `{cite['filename']}` | **Page:** {cite['page_number']} | **Relevance Match:** {cite.get('score_percentage', 'N/A')}")
                        st.caption(f"*\"{cite['snippet']}\"*")
                        st.divider()

            # Render hallucination warning if unanswerable
            if message.get("is_insufficient"):
                st.warning("Hallucination Safeguard Triggered: The uploaded documents do not contain sufficient verified context to answer this specific query.")

    # Query Input Box
    if query := st.chat_input("Ask a question about placement rules, eligibility, CGPA criteria, or company JDs..."):
        # Ensure uploaded files are indexed before answering
        if uploaded_files and st.session_state.rag_pipeline.vector_store.total_chunks() == 0:
            with st.spinner("Indexing uploaded documents into FAISS..."):
                pdf_inputs = [{"input": f, "filename": f.name} for f in uploaded_files]
                summary = st.session_state.rag_pipeline.process_pdf_inputs(pdf_inputs)
                st.session_state.processed_files_list = st.session_state.rag_pipeline.processed_files
                st.session_state.last_analytics["total_pdfs"] = len(st.session_state.processed_files_list)
                st.session_state.last_analytics["total_chunks"] = summary["total_indexed_chunks"]

        # Append user question to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user", avatar=None):
            st.markdown(query)

        # Process AI RAG completion
        with st.chat_message("assistant", avatar=None):
            with st.spinner("Retrieving relevant passages & generating answer via local Ollama (llama3.2)..."):
                rag_result = st.session_state.rag_pipeline.answer_question(
                    query=query,
                    chat_history=st.session_state.chat_history[:-1]
                )

                answer = rag_result["answer"]
                citations = rag_result["citations"]
                is_sufficient = rag_result["is_sufficient"]

                # Update analytics metrics
                st.session_state.last_analytics["retrieval_time_ms"] = rag_result["retrieval_time_ms"]
                st.session_state.last_analytics["generation_time_ms"] = rag_result["generation_time_ms"]

                # Output answer text
                st.markdown(answer)

                # Output citations
                if citations:
                    with st.expander("View Source Citations & Similarity Scores", expanded=True):
                        for cite in citations:
                            st.markdown(f"**Document:** `{cite['filename']}` | **Page:** {cite['page_number']} | **Relevance Match:** {cite.get('score_percentage', 'N/A')}")
                            st.caption(f"*\"{cite['snippet']}\"*")

                if not is_sufficient:
                    st.warning("Hallucination Safeguard Triggered: The uploaded documents do not contain sufficient verified context to answer this specific query.")

                # Append assistant message to session state
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "is_insufficient": not is_sufficient
                })
                st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: VECTOR VISUALIZATION
# ------------------------------------------------------------------------------
with tab_viz:
    st.subheader("2D Vector Space Map")
    
    # Retrieve existing stored embeddings from FAISS without recomputing
    embeddings = st.session_state.rag_pipeline.vector_store.get_all_embeddings()
    chunks_meta = st.session_state.rag_pipeline.vector_store.chunks_metadata
    retrieved_ids = st.session_state.rag_pipeline.retriever.last_retrieved_chunk_ids
    query_vec = st.session_state.rag_pipeline.retriever.last_query_vector
    query_txt = st.session_state.rag_pipeline.retriever.last_query_text

    render_vector_visualization(
        embeddings=embeddings,
        chunks_metadata=chunks_meta,
        retrieved_chunk_ids=retrieved_ids,
        query_vector=query_vec,
        query_text=query_txt
    )
