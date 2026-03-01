"""
Endee Adapter Layer

This module simulates interaction with Endee vector database.
In production, embedding vectors would be inserted and searched
via Endee's vector indexing engine.
"""
import numpy as np
from app.embeddings import generate_embedding

vector_store = []

def store_document(doc_id: str, text: str):
    embedding = generate_embedding(text)

    vector_store.append({
        "id": doc_id,
        "text": text,
        "embedding": np.array(embedding)
    })

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query: str, top_k=3):
    if not vector_store:
        return []

    query_embedding = np.array(generate_embedding(query))

    similarities = []

    for doc in vector_store:
        score = cosine_similarity(query_embedding, doc["embedding"])
        similarities.append((score, doc))

    similarities.sort(reverse=True, key=lambda x: x[0])

    top_docs = [doc for _, doc in similarities[:top_k]]

    return top_docs