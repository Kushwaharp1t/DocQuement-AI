"""
Prompt templates and formatting utilities for RAG query construction using local Ollama (llama3.2).
"""

from typing import List, Dict, Any

# Mandatory phrase required when context is completely insufficient
INSUFFICIENT_INFO_PHRASE = "I could not find sufficient information in the uploaded documents."

SYSTEM_PROMPT = f"""You are an intelligent AI Document Assistant. Your task is to answer the user's question accurately using ONLY the provided Context snippets extracted from uploaded PDF documents.

GUIDELINES:
1. Carefully read all provided Context snippets below.
2. Provide a clear, direct, and complete answer based on facts found in the Context.
3. If the provided Context contains no relevant information to answer the question at all, respond EXACTLY with:
   "{INSUFFICIENT_INFO_PHRASE}"
4. Do not invent facts or use external knowledge not present in the Context.
"""


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
    """
    Constructs a structured prompt for Ollama llama3.2 containing system rules,
    chat history, and numbered context passages with document name and page number references.

    Args:
        query (str): User question.
        retrieved_chunks (List[Dict[str, Any]]): Retrieved matching chunk records.
        chat_history (List[Dict[str, str]]): List of previous turn dicts.

    Returns:
        str: Fully formatted prompt ready for LLM completion.
    """
    context_blocks = []
    for idx, match in enumerate(retrieved_chunks, 1):
        chunk = match.get("chunk", {})
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "document.pdf")
        page_num = meta.get("page_number", "1")
        text = chunk.get("text", "")
        score = match.get("score_percentage", "N/A")

        context_blocks.append(
            f"--- Snippet [{idx}] (Document: {filename}, Page: {page_num}, Relevance: {score}) ---\n"
            f"{text}\n"
        )

    formatted_context = "\n".join(context_blocks) if context_blocks else "NO CONTEXT AVAILABLE."

    history_str = ""
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-4:]  # Keep last 2 turns
        history_lines = []
        for msg in recent_history:
            role = "Student" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_str = "Recent Chat History:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""{SYSTEM_PROMPT}

{history_str}==================== CONTEXT SNIPPETS ====================
{formatted_context}
==========================================================

Question: {query}

Answer:"""

    return prompt
