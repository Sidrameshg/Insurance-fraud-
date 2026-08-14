from pathlib import Path
import sys
import json


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =================================================
# INPUT
# =================================================

PHASE3_FEATURES = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features_final.json"
)

PHASE3_RISK = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_risk_fusion_dynamic.json"
)


# =================================================
# OUTPUT
# =================================================

OUTPUT_DIR = (
    ROOT
    / "phase4"
    / "multi_document"
    / "consistency"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "claim_consistency.json"
)


# =================================================
# LOAD JSON
# =================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required Phase 3 file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =================================================
# ANALYSIS
# =================================================

def analyze_consistency(
    features
):

    contradictions = []
    consistent_evidence = []
    observations = []

    # ---------------------------------------------
    # Financial comparison
    # ---------------------------------------------

    claim_amount = float(
        features.get(
            "claim_repair_amount",
            0
        )
    )

    invoice_amount = float(
        features.get(
            "invoice_total",
            0
        )
    )

    difference = float(
        features.get(
            "claim_invoice_amount_difference",
            0
        )
    )

    ratio = float(
        features.get(
            "invoice_to_claim_amount_ratio",
            0
        )
    )

    if (
        claim_amount > 0
        and invoice_amount > 0
        and difference != 0
    ):

        contradictions.append({

            "type":
                "financial_mismatch",

            "claim_repair_amount":
                claim_amount,

            "invoice_amount":
                invoice_amount,

            "difference":
                difference,

            "invoice_to_claim_ratio":
                ratio,

            "description":
                "Claimed repair amount and invoice amount differ."
        })

    else:

        consistent_evidence.append(
            "Claim repair amount and invoice amount are consistent."
        )

    # ---------------------------------------------
    # Invoice validation
    # ---------------------------------------------

    invoice_valid = int(
        features.get(
            "invoice_total_valid",
            0
        )
    )

    if invoice_valid == 1:

        consistent_evidence.append(
            "Invoice total validation passed."
        )

    else:

        contradictions.append({

            "type":
                "invoice_validation",

            "description":
                "Invoice total validation failed."
        })

    # ---------------------------------------------
    # Signature
    # ---------------------------------------------

    signature_match = int(
        features.get(
            "signature_match",
            0
        )
    )

    signature_similarity = float(
        features.get(
            "signature_similarity",
            0
        )
    )

    if signature_match == 1:

        consistent_evidence.append({

            "type":
                "signature",

            "similarity":
                signature_similarity,

            "description":
                "Signature verification returned MATCH."
        })

    else:

        contradictions.append({

            "type":
                "signature",

            "similarity":
                signature_similarity,

            "description":
                "Signature verification did not return MATCH."
        })

    # ---------------------------------------------
    # Traditional vs ML damage
    # ---------------------------------------------

    traditional_damage = int(
        features.get(
            "damage_detected",
            0
        )
    )

    ml_damage = int(
        features.get(
            "damage_ml_is_damage",
            0
        )
    )

    ml_normal = int(
        features.get(
            "damage_ml_is_normal",
            0
        )
    )

    ml_class = features.get(
        "damage_ml_class",
        "UNKNOWN"
    )

    ml_confidence = float(
        features.get(
            "damage_ml_confidence",
            0
        )
    )

    if traditional_damage == 1:

        observations.append(
            "Traditional damage detector found damage."
        )

    else:

        observations.append(
            "Traditional damage detector did not find damage."
        )

    if ml_damage == 1:

        observations.append(
            f"ML damage classifier detected damage "
            f"({ml_class}, confidence={ml_confidence:.4f})."
        )

    elif ml_normal == 1:

        observations.append(
            f"ML damage classifier classified the vehicle "
            f"as normal ({ml_class}, confidence={ml_confidence:.4f})."
        )

    # ---------------------------------------------
    # Damage contradiction
    # ---------------------------------------------

    if (
        traditional_damage == 1
        and ml_damage == 0
    ):

        contradictions.append({

            "type":
                "damage_model_disagreement",

            "traditional_damage":
                traditional_damage,

            "ml_damage":
                ml_damage,

            "ml_class":
                ml_class,

            "ml_confidence":
                ml_confidence,

            "description":
                "Traditional and ML damage analysis disagree."
        })

    elif (
        traditional_damage == 0
        and ml_damage == 1
    ):

        contradictions.append({

            "type":
                "damage_model_disagreement",

            "traditional_damage":
                traditional_damage,

            "ml_damage":
                ml_damage,

            "ml_class":
                ml_class,

            "ml_confidence":
                ml_confidence,

            "description":
                "Traditional and ML damage analysis disagree."
        })

    else:

        consistent_evidence.append(
            "Traditional and ML damage analysis are directionally consistent."
        )

    # ---------------------------------------------
    # Damage detector metrics
    # ---------------------------------------------

    observations.append({

        "damage_ratio":
            float(
                features.get(
                    "damage_ratio",
                    0
                )
            ),

        "damage_mean_pixel_difference":
            float(
                features.get(
                    "damage_mean_pixel_difference",
                    0
                )
            )
    })

    return {

        "contradiction_count":
            len(contradictions),

        "contradictions":
            contradictions,

        "consistent_evidence":
            consistent_evidence,

        "observations":
            observations
    }


# =================================================
# MAIN
# =================================================

def main():

    print()
    print("==========================================")
    print("       PHASE 4 MULTI-DOCUMENT ANALYSIS")
    print("==========================================")

    # ---------------------------------------------
    # Load Phase 3 feature vector
    # ---------------------------------------------

    print()
    print(
        "[1/3] Loading Phase 3 features..."
    )

    feature_data = load_json(
        PHASE3_FEATURES
    )

    features = feature_data.get(
        "features",
        {}
    )

    print(
        "      Features loaded:",
        len(features)
    )

    # ---------------------------------------------
    # Load risk output
    # ---------------------------------------------

    print()
    print(
        "[2/3] Loading Phase 3 risk result..."
    )

    risk_data = load_json(
        PHASE3_RISK
    )

    print(
        "      Risk result loaded."
    )

    # ---------------------------------------------
    # Analyze
    # ---------------------------------------------

    print()
    print(
        "[3/3] Comparing claim evidence..."
    )

    analysis = analyze_consistency(
        features
    )

    # ---------------------------------------------
    # Build result
    # ---------------------------------------------

    result = {

        "phase":
            "phase4",

        "component":
            "multi_document_analysis",

        "source": {

            "phase3_features":
                str(PHASE3_FEATURES),

            "phase3_risk":
                str(PHASE3_RISK)
        },

        "phase3_risk_summary":
            risk_data,

        "analysis":
            analysis
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------
    # Display
    # ---------------------------------------------

    print()
    print("==========================================")
    print("       CONSISTENCY ANALYSIS")
    print("==========================================")

    print()
    print(
        "Contradictions:",
        analysis[
            "contradiction_count"
        ]
    )

    for item in analysis[
        "contradictions"
    ]:

        print(
            "-",
            item["description"]
        )

    print()
    print(
        "Consistent evidence:"
    )

    for item in analysis[
        "consistent_evidence"
    ]:

        if isinstance(
            item,
            dict
        ):

            print(
                "-",
                item["description"]
            )

        else:

            print(
                "-",
                item
            )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("==========================================")
    print("       MULTI-DOCUMENT ANALYSIS COMPLETE")
    print("==========================================")


if __name__ == "__main__":

    main()
