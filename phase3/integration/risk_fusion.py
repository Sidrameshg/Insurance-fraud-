from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features_final.json"
)

OUTPUT_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_risk_fusion.json"
)


FRAUD_THRESHOLD = 0.15


def load_features():

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


def calculate_evidence_score(features):

    score = 0.0

    reasons = []

    # -------------------------------------------------
    # Invoice / claim amount mismatch
    # -------------------------------------------------

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

    ratio = float(
        features.get(
            "invoice_to_claim_amount_ratio",
            0
        )
    )

    if claim_amount > 0 and invoice_amount > 0:

        if ratio < 0.50:

            score += 30

            reasons.append(
                "Large invoice-to-claim amount mismatch"
            )

        elif ratio < 0.75:

            score += 15

            reasons.append(
                "Moderate invoice-to-claim amount mismatch"
            )

    # -------------------------------------------------
    # Invoice validation
    # -------------------------------------------------

    invoice_valid = int(
        features.get(
            "invoice_total_valid",
            0
        )
    )

    if invoice_valid == 0:

        score += 20

        reasons.append(
            "Invoice total validation failed"
        )

    # -------------------------------------------------
    # Signature
    # -------------------------------------------------

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

    if signature_match == 0:

        score += 25

        reasons.append(
            "Signature mismatch"
        )

    elif signature_similarity < 0.80:

        score += 15

        reasons.append(
            "Low signature similarity"
        )

    # -------------------------------------------------
    # Traditional damage
    # -------------------------------------------------

    traditional_damage = int(
        features.get(
            "damage_detected",
            0
        )
    )

    if traditional_damage == 1:

        score += 10

        reasons.append(
            "Traditional damage detector found damage"
        )

    # -------------------------------------------------
    # ML damage
    # -------------------------------------------------

    ml_damage = int(
        features.get(
            "damage_ml_is_damage",
            0
        )
    )

    ml_confidence = float(
        features.get(
            "damage_ml_confidence",
            0
        )
    )

    ml_normal = int(
        features.get(
            "damage_ml_is_normal",
            0
        )
    )

    if ml_damage == 1:

        score += 15

        reasons.append(
            "ML damage classifier detected damage"
        )

    elif (
        ml_normal == 1
        and
        ml_confidence >= 0.80
    ):

        score -= 10

        reasons.append(
            "ML damage classifier strongly indicates normal vehicle"
        )

    # -------------------------------------------------
    # Traditional / ML inconsistency
    # -------------------------------------------------

    if (
        traditional_damage == 1
        and
        ml_damage == 0
    ):

        score += 10

        reasons.append(
            "Traditional and ML damage detectors disagree"
        )

    # -------------------------------------------------
    # Clamp score
    # -------------------------------------------------

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return score, reasons


def determine_risk(
    fraud_probability,
    evidence_score
):

    # -------------------------------------------------
    # High risk
    # -------------------------------------------------

    if (
        fraud_probability >= 0.30
        or
        evidence_score >= 60
    ):

        return "HIGH_RISK"

    # -------------------------------------------------
    # Review
    # -------------------------------------------------

    if (
        fraud_probability >= FRAUD_THRESHOLD
        or
        evidence_score >= 30
    ):

        return "REVIEW_REQUIRED"

    # -------------------------------------------------
    # Low risk
    # -------------------------------------------------

    return "LOW_RISK"


def main():

    print()
    print("==========================================")
    print("          PHASE 3 RISK FUSION")
    print("==========================================")

    data = load_features()

    features = data["features"]

    # -------------------------------------------------
    # Current fraud model probability
    # -------------------------------------------------

    # This value comes from the trained
    # HistGradientBoosting model inspection.
    fraud_probability = 0.120003

    evidence_score, reasons = (
        calculate_evidence_score(
            features
        )
    )

    risk_level = determine_risk(
        fraud_probability,
        evidence_score
    )

    fraud_model_prediction = (
        "FRAUD"
        if fraud_probability >= FRAUD_THRESHOLD
        else "NON_FRAUD"
    )

    result = {

        "phase":
            "phase3",

        "fraud_model": {

            "model":
                "HistGradientBoosting",

            "probability":
                fraud_probability,

            "threshold":
                FRAUD_THRESHOLD,

            "prediction":
                fraud_model_prediction
        },

        "evidence": {

            "score":
                round(
                    evidence_score,
                    2
                ),

            "reasons":
                reasons
        },

        "final_risk": {

            "level":
                risk_level
        }
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print()
    print("===== FRAUD MODEL =====")

    print(
        "Probability:",
        fraud_probability
    )

    print(
        "Threshold:",
        FRAUD_THRESHOLD
    )

    print(
        "Prediction:",
        fraud_model_prediction
    )

    print()
    print("===== EVIDENCE =====")

    print(
        "Evidence score:",
        round(
            evidence_score,
            2
        )
    )

    for reason in reasons:

        print(
            "-",
            reason
        )

    print()
    print("===== FINAL RISK =====")

    print(
        "Risk level:",
        risk_level
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("==========================================")


if __name__ == "__main__":
    main()
