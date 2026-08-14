from pathlib import Path
import json
import sys
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from phase3.ml.fraud.inference.fraud_predictor import predict_fraud

FEATURE_FILE = ROOT / "phase3" / "integration" / "phase3_features_final.json"

FRAUD_TEST_FILE = (
    ROOT / "phase3" / "ml" / "fraud" / "datasets" / "test.csv"
)

OUTPUT_DIR = ROOT / "phase3" / "integration"

FRAUD_THRESHOLD = 0.15


def load_features():
    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_claim(row_number):
    data = pd.read_csv(FRAUD_TEST_FILE)

    if row_number < 0 or row_number >= len(data):
        raise ValueError(
            f"Row {row_number} is outside the test dataset. "
            f"Valid range: 0-{len(data)-1}"
        )

    actual_label = int(
        data.iloc[row_number]["FraudFound_P"]
    )

    claim = data.drop(
        columns=["FraudFound_P"]
    ).iloc[[row_number]]

    return claim, actual_label


def calculate_evidence_score(features):

    score = 0.0
    reasons = []

    claim_amount = float(
        features.get("claim_repair_amount", 0)
    )

    invoice_amount = float(
        features.get("invoice_total", 0)
    )

    ratio = float(
        features.get("invoice_to_claim_amount_ratio", 0)
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

    if int(
        features.get("invoice_total_valid", 0)
    ) == 0:

        score += 20
        reasons.append(
            "Invoice total validation failed"
        )

    signature_match = int(
        features.get("signature_match", 0)
    )

    signature_similarity = float(
        features.get("signature_similarity", 0)
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

    traditional_damage = int(
        features.get("damage_detected", 0)
    )

    if traditional_damage == 1:

        score += 10
        reasons.append(
            "Traditional damage detector found damage"
        )

    ml_damage = int(
        features.get("damage_ml_is_damage", 0)
    )

    ml_confidence = float(
        features.get("damage_ml_confidence", 0)
    )

    ml_normal = int(
        features.get("damage_ml_is_normal", 0)
    )

    if ml_damage == 1:

        score += 15
        reasons.append(
            "ML damage classifier detected damage"
        )

    elif ml_normal == 1 and ml_confidence >= 0.80:

        score -= 10
        reasons.append(
            "ML damage classifier strongly indicates normal vehicle"
        )

    if traditional_damage == 1 and ml_damage == 0:

        score += 10
        reasons.append(
            "Traditional and ML damage detectors disagree"
        )

    score = max(0.0, min(100.0, score))

    return score, reasons


def determine_risk(
    fraud_probability,
    evidence_score
):

    if fraud_probability >= 0.30 or evidence_score >= 60:
        return "HIGH_RISK"

    if fraud_probability >= FRAUD_THRESHOLD or evidence_score >= 30:
        return "REVIEW_REQUIRED"

    return "LOW_RISK"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Test dataset row number"
    )

    args = parser.parse_args()

    print()
    print("==========================================")
    print("      DYNAMIC PHASE 3 RISK FUSION")
    print("==========================================")

    print()
    print("Selected test row:", args.row)

    phase3_data = load_features()
    features = phase3_data["features"]

    claim, actual_label = load_claim(
        args.row
    )

    print(
        "Historical fraud features:",
        len(claim.columns)
    )

    fraud_result = predict_fraud(
        claim
    )

    fraud_probability = fraud_result[
        "fraud_probability"
    ]

    fraud_prediction = (
        "FRAUD"
        if fraud_probability >= FRAUD_THRESHOLD
        else "NON_FRAUD"
    )

    evidence_score, reasons = (
        calculate_evidence_score(
            features
        )
    )

    risk_level = determine_risk(
        fraud_probability,
        evidence_score
    )

    print()
    print("===== FRAUD MODEL =====")

    print(
        "Actual label:",
        actual_label
    )

    print(
        "Fraud probability:",
        round(fraud_probability, 6)
    )

    print(
        "Threshold:",
        FRAUD_THRESHOLD
    )

    print(
        "Prediction:",
        fraud_prediction
    )

    print()
    print("===== EVIDENCE =====")

    print(
        "Evidence score:",
        round(evidence_score, 2)
    )

    for reason in reasons:
        print("-", reason)

    print()
    print("===== FINAL RISK =====")

    print(
        "Risk level:",
        risk_level
    )

    result = {
        "phase": "phase3",
        "test_row": args.row,
        "actual_fraud_label": actual_label,

        "fraud_model": {
            "model": "HistGradientBoosting",
            "probability": fraud_probability,
            "threshold": FRAUD_THRESHOLD,
            "prediction": fraud_prediction
        },

        "evidence": {
            "feature_count": len(features),
            "score": round(evidence_score, 2),
            "reasons": reasons
        },

        "final_risk": {
            "level": risk_level
        }
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"phase3_risk_fusion_row_{args.row}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print()
    print("Output:")
    print(output_file)

    print()
    print("==========================================")


if __name__ == "__main__":

    main()
