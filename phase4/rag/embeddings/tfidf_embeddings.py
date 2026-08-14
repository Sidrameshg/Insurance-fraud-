from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "phase4"
    / "rag"
    / "vectorstore"
    / "documents.json"
)

OUTPUT_DIR = (
    ROOT
    / "phase4"
    / "rag"
    / "embeddings"
)

VECTORIZER_FILE = (
    OUTPUT_DIR
    / "tfidf_vectorizer.joblib"
)

EMBEDDINGS_FILE = (
    OUTPUT_DIR
    / "chunk_embeddings.json"
)


# =================================================
# LOAD CHUNKS
# =================================================

def load_chunks():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"RAG document store not found: "
            f"{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["chunks"]


# =================================================
# CREATE EMBEDDINGS
# =================================================

def create_embeddings(
    chunks
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    return (
        vectorizer,
        matrix
    )


# =================================================
# SAVE EMBEDDINGS
# =================================================

def save_embeddings(
    chunks,
    vectorizer,
    matrix
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    embedding_records = []

    for index, chunk in enumerate(
        chunks
    ):

        embedding_records.append({

            "document_id":
                chunk["document_id"],

            "source":
                chunk["source"],

            "chunk_id":
                chunk["chunk_id"],

            "vector":
                matrix[index].toarray()[0].tolist()
        })

    output = {

        "phase":
            "phase4",

        "component":
            "rag",

        "stage":
            "tfidf_embeddings",

        "embedding_type":
            "TF-IDF",

        "chunk_count":
            len(chunks),

        "vector_dimensions":
            matrix.shape[1],

        "embeddings":
            embedding_records
    }

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


# =================================================
# MAIN
# =================================================

def main():

    print()
    print("==========================================")
    print("       PHASE 4 RAG EMBEDDING")
    print("==========================================")

    chunks = load_chunks()

    print()
    print(
        "Chunks loaded:",
        len(chunks)
    )

    vectorizer, matrix = (
        create_embeddings(
            chunks
        )
    )

    print(
        "Embedding type: TF-IDF"
    )

    print(
        "Vector dimensions:",
        matrix.shape[1]
    )

    save_embeddings(
        chunks,
        vectorizer,
        matrix
    )

    print()
    print(
        "Vectorizer:"
    )

    print(
        VECTORIZER_FILE
    )

    print()
    print(
        "Embeddings:"
    )

    print(
        EMBEDDINGS_FILE
    )

    print()
    print("==========================================")
    print("       RAG EMBEDDING COMPLETE")
    print("==========================================")


if __name__ == "__main__":

    main()
