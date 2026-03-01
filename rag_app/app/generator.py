def generate_answer(query: str, docs: list):
    """
    Return retrieved context directly.
    This simulates RAG without external LLM.
    """
    if not docs:
        return "No relevant documents found."

    context = "\n\n".join([doc["text"] for doc in docs])

    return f"Retrieved Context:\n\n{context}"