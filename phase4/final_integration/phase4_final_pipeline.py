import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Allow Phase 4 modules to be imported
sys.path.insert(0, str(ROOT / "agent"))


from investigation_agent import build_investigation


OUTPUT_DIR = ROOT / "final_integration" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("          PHASE 4 FINAL INTEGRATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load Phase 3 result
    # ---------------------------------------------------------

    print()
    print("[1/6] Loading Phase 3 evidence...")

    phase3_file = (
        ROOT.parent
        / "phase3"
        / "integration"
        / "phase3_risk_fusion_dynamic.json"
    )

    if phase3_file.exists():

        phase3_data = load_json(phase3_file)

        print("      Phase 3 loaded successfully.")

    else:

        print("      Phase 3 risk file not found.")
        print("      Continuing with Investigation Agent data.")

        phase3_data = None

    # ---------------------------------------------------------
    # 2. Run Investigation Agent
    # ---------------------------------------------------------

    print()
    print("[2/6] Running Investigation Agent...")

    investigation, _ = build_investigation()

    print("      Investigation agent completed.")

    claim_evidence = investigation["claim_evidence"]
    retrieved_knowledge = investigation["retrieved_knowledge"]

    # ---------------------------------------------------------
    # 3. Multi-document analysis
    # ---------------------------------------------------------

    print()
    print("[3/6] Loading multi-document consistency analysis...")

    consistency_file = (
        ROOT
        / "multi_document"
        / "consistency"
        / "claim_consistency.json"
    )

    if consistency_file.exists():

        consistency_data = load_json(consistency_file)

        print("      Consistency analysis loaded.")

    else:

        consistency_data = {}

        print("      Consistency analysis file not found.")

    # ---------------------------------------------------------
    # 4. Investigation explanation
    # ---------------------------------------------------------

    print()
    print("[4/6] Loading investigation explanation...")

    explanation_file = (
        ROOT
        / "multi_document"
        / "explanation"
        / "claim_investigation_explanation.json"
    )

    if explanation_file.exists():

        explanation_data = load_json(explanation_file)

        print("      Investigation explanation loaded.")

    else:

        explanation_data = {}

        print("      Investigation explanation not found.")

    # ---------------------------------------------------------
    # 5. Build final grounded investigation report
    # ---------------------------------------------------------

    print()
    print("[5/6] Building final investigation report...")

    contradictions = consistency_data.get(
        "analysis",
        {}
    ).get(
        "contradictions",
        []
    )

    consistent_evidence = consistency_data.get(
        "analysis",
        {}
    ).get(
        "consistent_evidence",
        []
    )

    observations = consistency_data.get(
        "analysis",
        {}
    ).get(
        "observations",
        []
    )

    recommended_actions = explanation_data.get(
        "recommended_actions",
        []
    )

    if not recommended_actions:

        recommended_actions = [
            "Review the conflicting evidence manually.",
            "Verify significant financial differences.",
            "Compare available claim documents for consistency.",
            "Review multiple independent evidence sources before making a final decision."
        ]

    final_report = {

        "phase": "phase4",

        "component": "final_investigation_pipeline",

        "status": "COMPLETE",

        "claim_summary": {

            "fraud_probability":
                claim_evidence.get(
                    "fraud_probability"
                ),

            "fraud_prediction":
                claim_evidence.get(
                    "fraud_prediction"
                ),

            "risk_level":
                claim_evidence.get(
                    "risk_level"
                ),

            "evidence_score":
                claim_evidence.get(
                    "evidence_score"
                )
        },

        "important_evidence": {

            "evidence_score":
                claim_evidence.get(
                    "evidence_score"
                ),

            "observations":
                observations,

            "consistent_evidence":
                consistent_evidence
        },

        "contradictions": contradictions,

        "fraud_risk_interpretation": (
            "The fraud model prediction is a risk signal "
            "and should not be treated as definitive proof "
            "of fraud. The claim requires review because "
            "contradictory evidence was identified."
        ),

        "recommended_investigation_actions":
            recommended_actions,

        "retrieved_knowledge": [

            {
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "similarity": item.get("similarity")
            }

            for item in retrieved_knowledge
        ],

        "uncertainty": {

            "principle":
                "Conflicting evidence should be reviewed "
                "by a human investigator.",

            "hallucination_policy":
                "No facts should be invented beyond the "
                "supplied claim evidence and retrieved knowledge."
        },

        "source_components": {

            "phase3_result_loaded":
                phase3_data is not None,

            "investigation_agent":
                True,

            "rag_retrieval":
                len(retrieved_knowledge),

            "multi_document_analysis":
                bool(consistency_data),

            "investigation_explanation":
                bool(explanation_data)
        }
    }

    # ---------------------------------------------------------
    # 6. Save final output
    # ---------------------------------------------------------

    print()
    print("[6/6] Saving final Phase 4 report...")

    output_file = (
        OUTPUT_DIR
        / "phase4_final_investigation_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("          PHASE 4 FINAL INTEGRATION COMPLETE")
    print("=" * 70)

    print()
    print("Fraud probability:",
          final_report["claim_summary"]["fraud_probability"])

    print("Fraud prediction:",
          final_report["claim_summary"]["fraud_prediction"])

    print("Risk level:",
          final_report["claim_summary"]["risk_level"])

    print("Evidence score:",
          final_report["claim_summary"]["evidence_score"])

    print("Contradictions:",
          len(final_report["contradictions"]))

    print("Retrieved knowledge:",
          len(final_report["retrieved_knowledge"]))

    print()
    print("Recommended actions:")

    for action in final_report[
        "recommended_investigation_actions"
    ]:

        print("-", action)

    print()
    print("Output:")
    print(output_file)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
