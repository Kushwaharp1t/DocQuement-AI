"""
Vector Visualization module for projecting high-dimensional document chunk embeddings
into an interactive 2D Plotly scatter plot using Principal Component Analysis (PCA).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import streamlit as st


VISUALIZATION_EXPLANATION = (
    "Each point represents a document chunk embedding. PCA projects the "
    "high-dimensional embeddings into two dimensions for visualization. "
    "Points that are closer in this projection may have similar semantic "
    "representations, although the 2D projection does not preserve all "
    "high-dimensional relationships."
)


def render_vector_visualization(
    embeddings: np.ndarray,
    chunks_metadata: List[Dict[str, Any]],
    retrieved_chunk_ids: Optional[List[str]] = None,
    query_vector: Optional[np.ndarray] = None,
    query_text: Optional[str] = None
) -> None:
    """
    Renders an interactive 2D Plotly scatter plot visualization of chunk embeddings.

    Args:
        embeddings (np.ndarray): 2D float32 numpy array of shape (N, dimension).
        chunks_metadata (List[Dict[str, Any]]): Metadata objects for each chunk.
        retrieved_chunk_ids (Optional[List[str]]): List of chunk IDs retrieved in last query.
        query_vector (Optional[np.ndarray]): 2D array of shape (1, dimension) for user query.
        query_text (Optional[str]): Text of the last user query.
    """
    st.markdown(f"*{VISUALIZATION_EXPLANATION}*")
    st.markdown("")

    # Edge Case 1: No documents indexed
    if embeddings is None or len(embeddings) == 0 or len(chunks_metadata) == 0:
        st.info("No document chunks indexed in vector store yet. Upload PDFs to view vector embeddings.")
        return

    # Edge Case 2: Inconsistent number of embeddings and metadata entries
    num_items = min(len(embeddings), len(chunks_metadata))
    if num_items == 0:
        st.info("No document chunks indexed in vector store yet. Upload PDFs to view vector embeddings.")
        return

    embeddings = embeddings[:num_items]
    chunks_metadata = chunks_metadata[:num_items]
    retrieved_set = set(retrieved_chunk_ids or [])

    # Edge Case 3: Fewer than 2 chunks (PCA requires at least 2 samples for 2D components)
    if num_items < 2 and (query_vector is None or query_vector.size == 0):
        st.warning("At least 2 document chunks are required for 2D PCA projection.")
        return

    # Combine chunk embeddings and optional query vector for unified PCA space
    combined_vectors = [embeddings]
    if query_vector is not None and query_vector.size > 0:
        combined_vectors.append(query_vector.reshape(1, -1))

    all_matrix = np.vstack(combined_vectors)

    if all_matrix.shape[0] < 2:
        st.warning("At least 2 vector samples are required for 2D PCA projection.")
        return

    # Execute PCA 2D Dimensionality Reduction
    try:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(all_matrix)
    except Exception as e:
        st.error(f"PCA Dimensionality Reduction failed: {str(e)}")
        return

    chunk_coords = coords_2d[:num_items]
    has_query = (query_vector is not None and query_vector.size > 0 and len(coords_2d) > num_items)
    query_coord = coords_2d[num_items] if has_query else None

    # Separate points by category for clean visual distinction
    normal_x, normal_y, normal_hover = [], [], []
    retrieved_x, retrieved_y, retrieved_hover = [], [], []

    for i in range(num_items):
        meta = chunks_metadata[i].get("metadata", {})
        filename = meta.get("filename", "document.pdf")
        page_num = meta.get("page_number", 1)
        chunk_id = meta.get("chunk_id", chunks_metadata[i].get("chunk_id", f"chunk_{i}"))
        text = chunks_metadata[i].get("text", "")
        preview = (text[:120] + "...") if len(text) > 120 else text
        # Escape HTML break lines for hover
        hover_str = (
            f"<b>Document:</b> {filename}<br>"
            f"<b>Page:</b> {page_num}<br>"
            f"<b>Chunk ID:</b> {chunk_id}<br>"
            f"<b>Preview:</b> <i>{preview}</i>"
        )

        x_val, y_val = float(chunk_coords[i, 0]), float(chunk_coords[i, 1])

        if chunk_id in retrieved_set:
            retrieved_x.append(x_val)
            retrieved_y.append(y_val)
            retrieved_hover.append(hover_str)
        else:
            normal_x.append(x_val)
            normal_y.append(y_val)
            normal_hover.append(hover_str)

    # Build Interactive Plotly Figure
    fig = go.Figure()

    # Trace 1: Normal Document Chunks
    if normal_x:
        fig.add_trace(go.Scatter(
            x=normal_x,
            y=normal_y,
            mode='markers',
            name='Document Chunk',
            hoverinfo='text',
            hovertext=normal_hover,
            marker=dict(
                size=10,
                color='#38bdf8',  # Sky blue
                opacity=0.85,
                line=dict(width=1, color='#0284c7')
            )
        ))

    # Trace 2: Retrieved Chunks (Highlighted for latest query)
    if retrieved_x:
        fig.add_trace(go.Scatter(
            x=retrieved_x,
            y=retrieved_y,
            mode='markers',
            name='Retrieved Chunk',
            hoverinfo='text',
            hovertext=retrieved_hover,
            marker=dict(
                size=14,
                color='#f59e0b',  # Highlight Amber / Orange
                symbol='diamond',
                opacity=0.95,
                line=dict(width=2, color='#b45309')
            )
        ))

    # Trace 3: User Query Vector Point
    if has_query and query_coord is not None:
        q_preview = (query_text[:100] + "...") if query_text and len(query_text) > 100 else (query_text or "User Question")
        query_hover = f"<b>User Query:</b> {q_preview}<br><i>(Highlighted Query Vector)</i>"

        fig.add_trace(go.Scatter(
            x=[float(query_coord[0])],
            y=[float(query_coord[1])],
            mode='markers',
            name='User Query',
            hoverinfo='text',
            hovertext=[query_hover],
            marker=dict(
                size=18,
                color='#ef4444',  # Red Star
                symbol='star',
                line=dict(width=2, color='#991b1b')
            )
        ))

    # Customize Chart Layout
    fig.update_layout(
        title=f"2D PCA Embedding Map ({num_items} Indexed Chunks)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        template="plotly_dark",
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)
