from fastapi import FastAPI
import os
from app.retriever import store_document, search
from app.generator import generate_answer

# Create FastAPI app FIRST
app = FastAPI()


@app.post("/load")
def load_docs():
    """
    Load sample document into vector store
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "sample_docs", "ai_notes.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        store_document("doc1", content)

    return {"status": "Documents loaded successfully"}


@app.post("/query")
def query(question: str):
    """
    Query the RAG system
    """
    docs = search(question)
    answer = generate_answer(question, docs)

    return {"response": answer}