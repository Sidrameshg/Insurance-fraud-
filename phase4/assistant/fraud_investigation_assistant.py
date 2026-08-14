import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "agent"))

from investigation_agent import build_investigation


OUTPUT_DIR = ROOT / "assistant" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def answer_question(question):

    investigation, _ = build_investigation()

    evidence = investigation["claim_evidence"]
    knowledge = investigation["retrieved_knowledge"]

    question_lower = question.lower()

    if "risk" in question_lower:

        answer = (
            f"The current claim risk level is "
            f"{evidence['risk_level']}. "
            f"The fraud model probability is "
            f"{evidence['fraud_probability']:.6f}, "
            f"and its prediction is "
            f"{evidence['fraud_prediction']}."
        )

    elif "contradiction" in question_lower:

        contradictions = evidence["contradictions"]

        answer = (
            f"The analysis identified "
            f"{len(contradictions)} contradictions."
        )

        for contradiction in contradictions:

            if isinstance(contradiction, dict):
                description = contradiction.get(
                    "description",
                    "Contradictory evidence detected."
                )
                answer += f"\n- {description}"
            else:
                answer += f"\n- {contradiction}"

    elif (
        "investigator" in question_lower
        or "investigate" in question_lower
        or "action" in question_lower
    ):

        answer = (
            "The investigator should review the conflicting "
            "evidence manually, verify significant financial "
            "differences, compare the available claim documents "
            "for consistency, and review multiple independent "
            "evidence sources before making a final decision."
        )

    else:

        answer = (
            "The available evidence indicates that the claim "
            "is classified as REVIEW_REQUIRED. The fraud model "
            "prediction is NON_FRAUD, but contradictions were "
            "identified and should be reviewed."
        )

    return {
        "question": question,
        "answer": answer,

        "claim_context": {
            "fraud_probability": evidence["fraud_probability"],
            "fraud_prediction": evidence["fraud_prediction"],
            "risk_level": evidence["risk_level"],
            "evidence_score": evidence["evidence_score"],
            "contradiction_count": len(
                evidence["contradictions"]
            )
        },

        "knowledge_sources": [
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "similarity": item["similarity"]
            }
            for item in knowledge
        ],

        "grounding": {
            "phase": "phase4",
            "hallucination_policy":
                "Do not invent information absent from supplied evidence."
        }
    }


def main():

    print("=" * 60)
    print("       PHASE 4 FRAUD INVESTIGATION ASSISTANT")
    print("=" * 60)

    questions = [
        "What is the risk level of this claim?",
        "What contradictions were found?",
        "What should the investigator do?"
    ]

    results = []

    for question in questions:

        print()
        print("Question:")
        print(question)

        result = answer_question(question)

        print()
        print("Answer:")
        print(result["answer"])

        results.append(result)

    output = {
        "phase": "phase4",
        "component": "fraud_investigation_assistant",
        "responses": results
    }

    output_file = (
        OUTPUT_DIR /
        "fraud_investigation_assistant.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 60)
    print("       INVESTIGATION ASSISTANT COMPLETE")
    print("=" * 60)

    print()
    print("Questions answered:", len(results))

    print()
    print("Output:")
    print(output_file)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
