import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONSISTENCY_FILE = (
    ROOT
    / "multi_document"
    / "consistency"
    / "claim_consistency.json"
)

OUTPUT_FILE = (
    ROOT
    / "multi_document"
    / "explanation"
    / "claim_investigation_explanation.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_explanation(data):

    analysis = data.get("analysis", {})

    contradictions = analysis.get("contradictions", [])
    consistent_evidence = analysis.get("consistent_evidence", [])
    observations = analysis.get("observations", [])

    explanation = {
        "engine": "Phase 4 Investigation Explanation",
        "source": "Multi-Document Consistency Analysis",

        "contradiction_count": len(contradictions),

        "contradictions": contradictions,

        "consistent_evidence": consistent_evidence,

        "observations": observations,

        "investigation_interpretation": [],

        "recommended_actions": []
    }

    for item in contradictions:

        contradiction_type = item.get("type", "")
        description = item.get("description", "")

        if contradiction_type == "financial_mismatch":

            claim_amount = item.get("claim_repair_amount")
            invoice_amount = item.get("invoice_amount")
            difference = item.get("difference")

            explanation["investigation_interpretation"].append(
                f"The claimed repair amount ({claim_amount}) "
                f"differs from the invoice amount ({invoice_amount}) "
                f"by {difference}."
            )

            explanation["recommended_actions"].append(
                "Review and verify the claimed repair amount against the submitted invoice."
            )

        elif contradiction_type == "damage_model_disagreement":

            traditional_damage = item.get("traditional_damage")
            ml_damage = item.get("ml_damage")
            ml_class = item.get("ml_class")
            ml_confidence = item.get("ml_confidence")

            explanation["investigation_interpretation"].append(
                f"Traditional damage analysis returned {traditional_damage}, "
                f"while the ML classifier returned {ml_damage} "
                f"with class {ml_class} and confidence {ml_confidence}."
            )

            explanation["recommended_actions"].append(
                "Review the vehicle damage evidence manually because the traditional and ML analyses disagree."
            )

    explanation["grounding_rule"] = (
        "This explanation uses only the evidence contained in the "
        "Phase 4 multi-document consistency analysis. "
        "No additional claim facts are invented."
    )

    return explanation


def main():

    print("=" * 50)
    print("   PHASE 4 INVESTIGATION EXPLANATION")
    print("=" * 50)

    print()
    print("[1/3] Loading consistency analysis...")

    if not CONSISTENCY_FILE.exists():
        raise FileNotFoundError(
            f"Consistency file not found: {CONSISTENCY_FILE}"
        )

    data = load_json(CONSISTENCY_FILE)

    print("      Consistency analysis loaded.")

    print()
    print("[2/3] Building investigation explanation...")

    result = build_explanation(data)

    print("      Explanation generated.")

    print()
    print("[3/3] Saving explanation...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 50)
    print("   INVESTIGATION EXPLANATION COMPLETE")
    print("=" * 50)

    print()
    print("Contradictions:", result["contradiction_count"])

    for item in result["contradictions"]:
        print("-", item.get("description", "Unknown contradiction"))

    print()
    print("Consistent evidence:")

    for item in result["consistent_evidence"]:
        if isinstance(item, str):
            print("-", item)
        else:
            print("-", item.get("description", str(item)))

    print()
    print("Recommended actions:")

    for action in result["recommended_actions"]:
        print("-", action)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()
