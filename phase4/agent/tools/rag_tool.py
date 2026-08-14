import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_FILE = ROOT / "rag" / "vectorstore" / "documents.json"
VECTORIZER_FILE = ROOT / "rag" / "embeddings" / "tfidf_vectorizer.joblib"
EMBEDDING_FILE = ROOT / "rag" / "embeddings" / "chunk_embeddings.json"


def load_documents():

    with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"]


def load_embeddings():

    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["embeddings"]


def retrieve_knowledge(query, top_k=3):

    documents = load_documents()
    embeddings = load_embeddings()

    vectorizer = joblib.load(VECTORIZER_FILE)

    query_vector = vectorizer.transform([query])

    vectors = [
        embedding["vector"]
        for embedding in embeddings
    ]

    matrix = np.asarray(vectors, dtype=float)

    similarities = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        index = int(index)

        embedding = embeddings[index]

        matching_document = None

        for document in documents:

            if (
                document.get("source") == embedding.get("source")
                and document.get("chunk_id") == embedding.get("chunk_id")
            ):
                matching_document = document
                break

        if matching_document is None:
            continue

        results.append({
            "source": embedding["source"],
            "chunk_id": embedding["chunk_id"],
            "similarity": float(similarities[index]),
            "text": matching_document.get("text", "")
        })

    return results


if __name__ == "__main__":

    print("=" * 50)
    print("       PHASE 4 AGENT RAG TOOL")
    print("=" * 50)

    query = (
        "What should an investigator do when "
        "insurance claim evidence contains contradictions?"
    )

    print()
    print("Query:")
    print(query)

    results = retrieve_knowledge(query)

    print()
    print("Retrieved chunks:", len(results))

    for number, result in enumerate(results, start=1):

        print()
        print(f"--- RESULT {number} ---")
        print("Source:", result["source"])
        print("Chunk:", result["chunk_id"])
        print("Similarity:", round(result["similarity"], 4))
        print("Text:", result["text"][:500])

    print()
    print("RAG knowledge tool ready.")

    print("=" * 50)
