from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]


# =================================================
# PATHS
# =================================================

VECTORSTORE_FILE = (
    ROOT
    / "phase4"
    / "rag"
    / "vectorstore"
    / "documents.json"
)

VECTORIZER_FILE = (
    ROOT
    / "phase4"
    / "rag"
    / "embeddings"
    / "tfidf_vectorizer.joblib"
)

EMBEDDINGS_FILE = (
    ROOT
    / "phase4"
    / "rag"
    / "embeddings"
    / "chunk_embeddings.json"
)


# =================================================
# RETRIEVER
# =================================================

class RAGRetriever:

    def __init__(
        self,
        top_k=3
    ):

        self.top_k = top_k

        self.vectorizer = (
            joblib.load(
                VECTORIZER_FILE
            )
        )

        with open(
            VECTORSTORE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            document_data = json.load(
                file
            )

        self.chunks = (
            document_data[
                "chunks"
            ]
        )

        with open(
            EMBEDDINGS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            embedding_data = json.load(
                file
            )

        self.embeddings = np.array(
            [
                item["vector"]
                for item
                in embedding_data[
                    "embeddings"
                ]
            ]
        )

    def retrieve(
        self,
        query
    ):

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.embeddings
        )[0]

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[
            :self.top_k
        ]:

            chunk = self.chunks[
                index
            ]

            results.append({

                "document_id":
                    chunk[
                        "document_id"
                    ],

                "source":
                    chunk[
                        "source"
                    ],

                "chunk_id":
                    chunk[
                        "chunk_id"
                    ],

                "score":
                    float(
                        scores[index]
                    ),

                "text":
                    chunk[
                        "text"
                    ]
            })

        return results


# =================================================
# TEST
# =================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("          PHASE 4 RAG RETRIEVER")
    print("==========================================")

    retriever = RAGRetriever(
        top_k=3
    )

    query = (
        "What should an investigator do "
        "when traditional and ML damage "
        "detectors disagree?"
    )

    print()
    print(
        "Query:",
        query
    )

    results = retriever.retrieve(
        query
    )

    print()
    print(
        "Retrieved chunks:",
        len(results)
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"--- RESULT {index} ---"
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "Chunk:",
            result["chunk_id"]
        )

        print(
            "Similarity:",
            round(
                result["score"],
                4
            )
        )

        print(
            "Text:",
            result["text"]
        )

    print()
    print("==========================================")
    print("          RAG RETRIEVAL COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    pass
