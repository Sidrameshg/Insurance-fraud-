from pathlib import Path
import sys


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from phase4.rag.retrieval.rag_retriever import (
    RAGRetriever
)


# =================================================
# CONTEXT BUILDER
# =================================================

class RAGContextBuilder:

    def __init__(
        self,
        top_k=3
    ):

        self.retriever = RAGRetriever(
            top_k=top_k
        )

    def build_context(
        self,
        query
    ):

        results = (
            self.retriever.retrieve(
                query
            )
        )

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {index}
Document: {result["source"]}
Chunk: {result["chunk_id"]}
Similarity: {result["score"]:.4f}

{result["text"]}
""".strip()
            )

        context = "\n\n".join(
            context_parts
        )

        return {
            "query": query,
            "results": results,
            "context": context
        }


# =================================================
# TEST
# =================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("       PHASE 4 RAG CONTEXT BUILDER")
    print("==========================================")

    query = (
        "What should an investigator do "
        "when traditional and ML damage "
        "detectors disagree?"
    )

    builder = RAGContextBuilder(
        top_k=3
    )

    result = builder.build_context(
        query
    )

    print()
    print(
        "Query:",
        query
    )

    print()
    print("===== GROUNDED CONTEXT =====")
    print()

    print(
        result["context"]
    )

    print()
    print("==========================================")
    print("       CONTEXT BUILD COMPLETE")
    print("==========================================")
