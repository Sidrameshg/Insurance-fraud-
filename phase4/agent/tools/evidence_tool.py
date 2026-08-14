import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CONSISTENCY_FILE = (
    ROOT
    / "multi_document"
    / "consistency"
    / "claim_consistency.json"
)

EXPLANATION_FILE = (
    ROOT
    / "multi_document"
    / "explanation"
    / "claim_investigation_explanation.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_evidence():

    consistency = load_json(CONSISTENCY_FILE)
    explanation = load_json(EXPLANATION_FILE)

    analysis = consistency.get("analysis", {})
    risk = consistency.get("phase3_risk_summary", {})

    return {
        "fraud_probability": risk.get("fraud_model", {}).get("probability"),
        "fraud_prediction": risk.get("fraud_model", {}).get("prediction"),
        "risk_level": risk.get("final_risk", {}).get("level"),
        "evidence_score": risk.get("evidence", {}).get("score"),
        "contradiction_count": analysis.get("contradiction_count", 0),
        "contradictions": analysis.get("contradictions", []),
        "consistent_evidence": analysis.get("consistent_evidence", []),
        "observations": analysis.get("observations", []),
        "recommended_actions": explanation.get(
            "recommended_actions", []
        )
    }


if __name__ == "__main__":

    print("=" * 50)
    print("       PHASE 4 AGENT EVIDENCE TOOL")
    print("=" * 50)

    result = analyze_evidence()

    print()
    print("Fraud probability:", result["fraud_probability"])
    print("Fraud prediction :", result["fraud_prediction"])
    print("Risk level       :", result["risk_level"])
    print("Evidence score   :", result["evidence_score"])
    print("Contradictions   :", result["contradiction_count"])

    print()
    print("Evidence analysis tool ready.")

    print("=" * 50)
