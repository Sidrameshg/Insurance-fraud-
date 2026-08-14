from pathlib import Path
import json
import sys
import argparse

import pandas as pd


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =================================================
# PHASE 3 MODULES
# =================================================

from phase3.ml.fraud.inference.claim_adapter import (
    predict_claim
)


# =================================================
# PATHS
# =================================================

FRAUD_TEST_FILE = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "datasets"
    / "test.csv"
)

EVIDENCE_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features_final.json"
)

OUTPUT_DIR = (
    ROOT
    / "phase3"
    / "integration"
    / "final"
)


# =================================================
# CONFIGURATION
# =================================================

FRAUD_THRESHOLD = 0.15


# =================================================
# LOAD EVIDENCE
# =================================================

def load_evidence():

    with open(
        EVIDENCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# =================================================
# LOAD FRAUD CLAIM
# =================================================

def load_fraud_claim(row_number):

    data = pd.read_csv(
        FRAUD_TEST_FILE
    )

    if row_number < 0 or row_number >= len(data):

        raise ValueError(
            f"Invalid row {row_number}. "
            f"Valid range: 0-{len(data)-1}"
        )

    row = data.iloc[
        row_number
    ]

    actual_label = int(
        row["FraudFound_P"]
    )

    claim = row.drop(
        labels=["FraudFound_P"]
    ).to_dict()

    return (
        claim,
        actual_label
    )


# =================================================
# EVIDENCE SCORE
# =================================================

def calculate_evidence_score(
    features
):

    score = 0.0

    reasons = []

    # ---------------------------------------------
    # Invoice mismatch
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

    ratio = float(
        features.get(
            "invoice_to_claim_amount_ratio",
            0
        )
    )

    if (
        claim_amount > 0
        and invoice_amount > 0
    ):

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

    # ---------------------------------------------
    # Invoice validation
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Traditional damage
    # ---------------------------------------------

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

    # ---------------------------------------------
    # ML damage
    # ---------------------------------------------

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

    ml_confidence = float(
        features.get(
            "damage_ml_confidence",
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

    # ---------------------------------------------
    # Traditional / ML disagreement
    # ---------------------------------------------

    if (
        traditional_damage == 1
        and
        ml_damage == 0
    ):

        score += 10

        reasons.append(
            "Traditional and ML damage detectors disagree"
        )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return (
        score,
        reasons
    )


# =================================================
# FINAL RISK DECISION
# =================================================

def determine_risk(
    fraud_probability,
    evidence_score
):

    if (
        fraud_probability >= 0.30
        or
        evidence_score >= 60
    ):

        return "HIGH_RISK"

    if (
        fraud_probability >= FRAUD_THRESHOLD
        or
        evidence_score >= 30
    ):

        return "REVIEW_REQUIRED"

    return "LOW_RISK"


# =================================================
# MAIN
# =================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--row",
        type=int,
        default=50
    )

    args = parser.parse_args()

    print()
    print("==========================================")
    print("       PHASE 3 FINAL CLAIM FUSION")
    print("==========================================")

    # ---------------------------------------------
    # Load Phase 3 evidence
    # ---------------------------------------------

    print()
    print("[1/4] Loading Phase 3 evidence...")

    evidence_data = load_evidence()

    evidence_features = (
        evidence_data[
            "features"
        ]
    )

    print(
        "      Evidence features:",
        len(evidence_features)
    )

    # ---------------------------------------------
    # Load fraud claim
    # ---------------------------------------------

    print()
    print(
        "[2/4] Loading fraud claim row:",
        args.row
    )

    claim, actual_label = (
        load_fraud_claim(
            args.row
        )
    )

    print(
        "      Fraud features:",
        len(claim)
    )

    print(
        "      Actual label:",
        actual_label
    )

    # ---------------------------------------------
    # Fraud prediction
    # ---------------------------------------------

    print()
    print(
        "[3/4] Running HistGradientBoosting..."
    )

    fraud_result = predict_claim(
        claim
    )

    fraud_probability = (
        fraud_result[
            "fraud_probability"
        ]
    )

    fraud_prediction = (
        "FRAUD"
        if fraud_probability >= FRAUD_THRESHOLD
        else "NON_FRAUD"
    )

    print(
        "      Probability:",
        round(
            fraud_probability,
            6
        )
    )

    print(
        "      Threshold:",
        FRAUD_THRESHOLD
    )

    print(
        "      Prediction:",
        fraud_prediction
    )

    # ---------------------------------------------
    # Evidence fusion
    # ---------------------------------------------

    print()
    print(
        "[4/4] Calculating evidence risk..."
    )

    evidence_score, reasons = (
        calculate_evidence_score(
            evidence_features
        )
    )

    final_risk = determine_risk(
        fraud_probability,
        evidence_score
    )

    print(
        "      Evidence score:",
        round(
            evidence_score,
            2
        )
    )

    print()
    print("==========================================")
    print("             FINAL DECISION")
    print("==========================================")

    print(
        "Fraud probability:",
        round(
            fraud_probability,
            6
        )
    )

    print(
        "Evidence score:",
        round(
            evidence_score,
            2
        )
    )

    print(
        "Fraud prediction:",
        fraud_prediction
    )

    print(
        "Final risk:",
        final_risk
    )

    print()
    print("Risk reasons:")

    for reason in reasons:

        print(
            "-",
            reason
        )

    # ---------------------------------------------
    # Output
    # ---------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"phase3_final_claim_fusion_row_{args.row}.json"
    )

    result = {

        "phase":
            "phase3",

        "integration_type":
            "claim_fusion_test",

        "test_row":
            args.row,

        "actual_fraud_label":
            actual_label,

        "fraud_model": {

            "model":
                fraud_result[
                    "model"
                ],

            "feature_count":
                fraud_result[
                    "feature_count"
                ],

            "probability":
                fraud_probability,

            "threshold":
                FRAUD_THRESHOLD,

            "prediction":
                fraud_prediction
        },

        "phase3_evidence": {

            "feature_count":
                len(
                    evidence_features
                ),

            "evidence_score":
                evidence_score,

            "reasons":
                reasons
        },

        "final_risk": {

            "level":
                final_risk
        }
    }

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
    print("Output:")
    print(output_file)

    print()
    print("==========================================")
    print("       PHASE 3 FUSION COMPLETE")
    print("==========================================")


if __name__ == "__main__":

    main()
