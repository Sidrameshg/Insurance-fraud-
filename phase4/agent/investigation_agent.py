import json
from pathlib import Path

from tools.evidence_tool import analyze_evidence
from tools.rag_tool import retrieve_knowledge


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "agent" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_investigation():

    evidence = analyze_evidence()

    query = (
        "How should an investigator handle this insurance claim "
        "when there are contradictions or conflicting evidence?"
    )

    knowledge = retrieve_knowledge(
        query,
        top_k=3
    )

    investigation = {
        "phase": "phase4",
        "component": "investigation_agent",

        "claim_evidence": {
            "fraud_probability": evidence.get("fraud_probability"),
            "fraud_prediction": evidence.get("fraud_prediction"),
            "risk_level": evidence.get("risk_level"),
            "evidence_score": evidence.get("evidence_score"),
            "contradictions": evidence.get("contradictions")
        },

        "retrieved_knowledge": knowledge,

        "investigation": {
            "claim_summary": (
                "The claim requires investigation based on "
                "the available fraud and evidence signals."
            ),

            "important_evidence": evidence.get(
                "evidence",
                []
            ),

            "contradictions": evidence.get(
                "contradictions",
                []
            ),

            "fraud_risk_interpretation": (
                "The fraud model output should be treated as "
                "a risk signal rather than definitive proof. "
                "Conflicting evidence should be reviewed "
                "before making a final decision."
            ),

            "recommended_actions": [
                "Review the conflicting evidence manually.",
                "Verify significant financial differences.",
                "Compare the available claim documents for consistency.",
                "Review multiple independent evidence sources before making a final decision.",
                "Escalate conflicting evidence for human review."
            ]
        }
    }

    output_file = (
        OUTPUT_DIR
        / "investigation_agent_result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            investigation,
            f,
            indent=2
        )

    return investigation, output_file


if __name__ == "__main__":

    print("=" * 55)
    print("       PHASE 4 INVESTIGATION AGENT")
    print("=" * 55)

    print()
    print("[1/3] Loading claim evidence...")

    result, output = build_investigation()

    print("      Evidence loaded.")

    print()
    print("[2/3] Retrieving investigation knowledge...")

    print(
        "      Knowledge retrieved:",
        len(result["retrieved_knowledge"])
    )

    print()
    print("[3/3] Building investigation result...")

    print()
    print("=" * 55)
    print("       INVESTIGATION AGENT COMPLETE")
    print("=" * 55)

    print()
    print("Fraud probability:",
          result["claim_evidence"]["fraud_probability"])

    print("Fraud prediction:",
          result["claim_evidence"]["fraud_prediction"])

    print("Risk level:",
          result["claim_evidence"]["risk_level"])

    print(
        "Contradictions:",
        len(result["claim_evidence"]["contradictions"])
    )

    print()
    print("Recommended actions:")

    for action in result["investigation"]["recommended_actions"]:
        print("-", action)

    print()
    print("Output:")
    print(output)

    print()
    print("=" * 55)
