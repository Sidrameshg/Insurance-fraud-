from pathlib import Path
import sys
import json


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =================================================
# PHASE 4 MODULES
# =================================================

from phase4.rag.retrieval.context_builder import (
    RAGContextBuilder
)

from phase4.llm.client.llm_client import (
    LLMClient
)

from phase4.llm.prompts.investigation_prompt import (
    SYSTEM_PROMPT
)


# =================================================
# RAG + LLM PIPELINE
# =================================================

def build_rag_prompt(
    question,
    context
):

    return f"""
You are investigating an insurance claim.

Use ONLY the supplied knowledge context to support
general investigation guidance.

USER QUESTION:
{question}

RETRIEVED KNOWLEDGE:

{context}

INSTRUCTIONS:

1. Answer the user's question.
2. Use the retrieved knowledge as supporting context.
3. Do not invent facts.
4. Clearly distinguish general guidance from
   claim-specific evidence.
5. If the retrieved knowledge is insufficient,
   explicitly say so.

Provide a concise investigation-oriented answer.
"""


# =================================================
# MAIN
# =================================================

def main():

    print()
    print("==========================================")
    print("          PHASE 4 RAG + LLM")
    print("==========================================")

    question = (
        "What should an investigator do when "
        "traditional and ML damage detectors disagree?"
    )

    # ---------------------------------------------
    # Retrieval
    # ---------------------------------------------

    print()
    print(
        "[1/3] Retrieving knowledge..."
    )

    context_builder = (
        RAGContextBuilder(
            top_k=3
        )
    )

    retrieval_result = (
        context_builder.build_context(
            question
        )
    )

    print(
        "      Retrieved:",
        len(
            retrieval_result[
                "results"
            ]
        ),
        "chunks"
    )

    # ---------------------------------------------
    # Build prompt
    # ---------------------------------------------

    print()
    print(
        "[2/3] Building grounded LLM prompt..."
    )

    user_prompt = build_rag_prompt(
        question,
        retrieval_result[
            "context"
        ]
    )

    print(
        "      Prompt ready."
    )

    # ---------------------------------------------
    # LLM
    # ---------------------------------------------

    print()
    print(
        "[3/3] Running LLM..."
    )

    llm = LLMClient()

    llm_result = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    # ---------------------------------------------
    # Output
    # ---------------------------------------------

    result = {

        "phase":
            "phase4",

        "component":
            "rag_llm",

        "question":
            question,

        "retrieval": {

            "result_count":
                len(
                    retrieval_result[
                        "results"
                    ]
                ),

            "results":
                retrieval_result[
                    "results"
                ]
        },

        "llm": {

            "provider":
                llm_result[
                    "provider"
                ],

            "model":
                llm_result[
                    "model"
                ],

            "response":
                llm_result[
                    "response"
                ]
        }
    }

    output_file = (
        ROOT
        / "phase4"
        / "integration"
        / "rag_llm_result.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("==========================================")
    print("          RAG + LLM COMPLETE")
    print("==========================================")

    print()
    print(
        "Provider:",
        llm_result[
            "provider"
        ]
    )

    print(
        "Model:",
        llm_result[
            "model"
        ]
    )

    print()
    print(
        "Response:"
    )

    print(
        llm_result[
            "response"
        ]
    )

    print()
    print(
        "Output:"
    )

    print(
        output_file
    )

    print()
    print("==========================================")


if __name__ == "__main__":

    main()
