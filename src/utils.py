"""
Utility helper functions for citation formatting, document analytics,
chat export, and temporary file management.
"""

from typing import List, Dict, Any
import json
import time


def format_source_citations(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates and formats source document citations from retrieved chunks.

    Args:
        retrieved_chunks (List[Dict[str, Any]]): Retrieved matches list.

    Returns:
        List[Dict[str, Any]]: List of citation dicts containing document name, page, score.
    """
    seen = set()
    citations = []

    for match in retrieved_chunks:
        chunk = match.get("chunk", {})
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "document.pdf")
        page_number = meta.get("page_number", 1)
        score_pct = match.get("score_percentage", "N/A")
        raw_score = match.get("score", 0.0)

        key = (filename, page_number)
        if key not in seen:
            seen.add(key)
            citations.append({
                "filename": filename,
                "page_number": page_number,
                "score_percentage": score_pct,
                "score": raw_score,
                "snippet": chunk.get("text", "")[:150] + "..." if len(chunk.get("text", "")) > 150 else chunk.get("text", "")
            })

    return citations


def export_chat_history_json(chat_history: List[Dict[str, Any]]) -> str:
    """
    Converts chat history list into a formatted JSON string for download.

    Args:
        chat_history (List[Dict[str, Any]]): List of chat messages.

    Returns:
        str: Pretty formatted JSON.
    """
    export_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_messages": len(chat_history),
        "history": chat_history
    }
    return json.dumps(export_payload, indent=2)


def export_chat_history_markdown(chat_history: List[Dict[str, Any]]) -> str:
    """
    Converts chat history into a Markdown transcript string.

    Args:
        chat_history (List[Dict[str, Any]]): List of chat messages.

    Returns:
        str: Markdown document string.
    """
    lines = [
        "# AI Placement Assistant - Chat Transcript",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        ""
    ]

    for msg in chat_history:
        role = "Student" if msg.get("role") == "user" else "AI Assistant"
        content = msg.get("content", "")
        lines.append(f"### {role}")
        lines.append(content)

        if "citations" in msg and msg["citations"]:
            lines.append("\n**Sources Cited:**")
            for cite in msg["citations"]:
                lines.append(f"- `{cite['filename']}` (Page {cite['page_number']}) - Match: {cite.get('score_percentage', 'N/A')}")
        
        lines.append("\n---\n")

    return "\n".join(lines)
