from pathlib import Path

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
HIGH_FRAUD_THRESHOLD = 0.30

REVIEW_EVIDENCE_THRESHOLD = 30
HIGH_EVIDENCE_THRESHOLD = 60


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


def determine_risk(
    fraud_probability,
    evidence_score
):

    if (
        fraud_probability >= HIGH_FRAUD_THRESHOLD
        or
        (
            evidence_score is not None
            and evidence_score >= HIGH_EVIDENCE_THRESHOLD
        )
    ):

        return "HIGH_RISK"

    if (
        fraud_probability >= FRAUD_THRESHOLD
        or
        (
            evidence_score is not None
            and evidence_score >= REVIEW_EVIDENCE_THRESHOLD
        )
    ):

        return "REVIEW_REQUIRED"

    return "LOW_RISK"


def determine_uncertainty(
    fraud_probability
):

    threshold_distance = abs(
        fraud_probability - FRAUD_THRESHOLD
    )

    if threshold_distance <= 0.05:

        return (
            "HIGH",
            True,
            "Fraud probability is close to "
            "the decision threshold."
        )

    if threshold_distance <= 0.10:

        return (
            "MEDIUM",
            False,
            None
        )

    return (
        "LOW",
        False,
        None
    )


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

    # ========================================================
    # FRAUD MODEL
    # ========================================================

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

    # ========================================================
    # EVIDENCE
    #
    # Phase 5 does not invent evidence.
    # It only exposes model output here.
    # ========================================================

    evidence_score = None
    reasons = []

    # ========================================================
    # FINAL RISK
    # ========================================================

    risk_level = determine_risk(
        fraud_probability,
        evidence_score
    )

    # ========================================================
    # UNCERTAINTY
    # ========================================================

    (
        uncertainty_level,
        human_review_required,
        human_review_reason
    ) = determine_uncertainty(
        fraud_probability
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

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

        "status":
            "ANALYSIS_COMPLETE"
    }