from pathlib import Path
import json

import pandas as pd

from phase3.ml.fraud.inference.fraud_predictor import (
    predict_fraud
)


ROOT = Path(__file__).resolve().parents[2]

CLAIMS_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "fraud_oracle_clean.csv"
)


FRAUD_THRESHOLD = 0.15


def find_claim_row(claim_id: str):

    if not CLAIMS_DATASET.exists():

        raise FileNotFoundError(
            "Processed claims dataset not found: "
            + str(CLAIMS_DATASET)
        )

    data = pd.read_csv(
        CLAIMS_DATASET
    )

    matches = data[
        data["PolicyNumber"].astype(str).str.strip()
        ==
        str(claim_id).strip()
    ]

    if matches.empty:

        return None, None

    row_index = matches.index[0]

    return (
        data.loc[row_index],
        int(row_index)
    )


def prepare_claim_for_model(row):

    claim = row.to_dict()

    return pd.DataFrame(
        [claim]
    )


def load_phase3_fusion(row_number):

    candidate_files = [

        (
            ROOT
            / "phase3"
            / "integration"
            / "final"
            / f"phase3_final_claim_fusion_row_{row_number}.json"
        ),

        (
            ROOT
            / "phase3"
            / "integration"
            / f"phase3_risk_fusion_row_{row_number}.json"
        ),
    ]

    for fusion_file in candidate_files:

        if fusion_file.exists():

            with open(
                fusion_file,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

    return None


def analyze_claim(claim_id: str):

    row, row_number = find_claim_row(
        claim_id
    )

    if row is None:

        return None

    claim_dataframe = (
        prepare_claim_for_model(
            row
        )
    )

    fraud_result = predict_fraud(
        claim_dataframe
    )

    fraud_probability = float(
        fraud_result[
            "fraud_probability"
        ]
    )

    fraud_prediction = (
        "FRAUD"
        if fraud_probability >= FRAUD_THRESHOLD
        else "NON_FRAUD"
    )

    phase3_result = (
        load_phase3_fusion(
            row_number
        )
    )

    evidence_score = None
    reasons = []

    if phase3_result:

        phase3_evidence = phase3_result.get(
            "phase3_evidence",
            {}
        )

        evidence_score = phase3_evidence.get(
            "evidence_score"
        )

        reasons = phase3_evidence.get(
            "reasons",
            []
        )

    # ========================================================
    # FINAL RISK DECISION
    # ========================================================

    if fraud_probability >= FRAUD_THRESHOLD:

        risk_level = "HIGH_RISK"

    elif (
        evidence_score is not None
        and evidence_score >= 50
    ):

        risk_level = "REVIEW_REQUIRED"

    else:

        risk_level = "LOW_RISK"

    # ========================================================
    # UNCERTAINTY / HUMAN REVIEW
    # ========================================================

    uncertainty_level = "LOW"

    human_review_required = False

    human_review_reason = None

    threshold_distance = abs(
        fraud_probability - FRAUD_THRESHOLD
    )

    if threshold_distance <= 0.05:

        uncertainty_level = "HIGH"

        human_review_required = True

        human_review_reason = (
            "Fraud probability is close to "
            "the decision threshold."
        )

    elif threshold_distance <= 0.10:

        uncertainty_level = "MEDIUM"

    return {

        "claim_id":
            str(claim_id),

        "dataset_row":
            row_number,

        "fraud_probability":
            fraud_probability,

        "fraud_prediction":
            fraud_prediction,

        "fraud_threshold":
            FRAUD_THRESHOLD,

        "evidence_score":
            evidence_score,

        "evidence_reasons":
            reasons,

        "risk_level":
            risk_level,

        "uncertainty_level":
            uncertainty_level,

        "human_review_required":
            human_review_required,

        "human_review_reason":
            human_review_reason,

        "phase3_fusion_available":
            phase3_result is not None,

        "status":
            "ANALYSIS_COMPLETE"
    }


if __name__ == "__main__":

    result = analyze_claim("1")

    print(json.dumps(
        result,
        indent=2
    ))
