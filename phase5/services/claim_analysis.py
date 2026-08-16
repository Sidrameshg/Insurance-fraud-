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


PHASE3_FINAL_DIR = (
    ROOT
    / "phase3"
    / "integration"
    / "final"
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

    fusion_file = (
        PHASE3_FINAL_DIR
        / f"phase3_final_claim_fusion_row_{row_number}.json"
    )

    if not fusion_file.exists():

        return None

    with open(
        fusion_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


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
    risk_level = None
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

        risk_level = (
            phase3_result
            .get(
                "final_risk",
                {}
            )
            .get(
                "level"
            )
        )

    else:

        if fraud_probability >= 0.30:

            risk_level = "HIGH_RISK"

        elif fraud_probability >= FRAUD_THRESHOLD:

            risk_level = "REVIEW_REQUIRED"

        else:

            risk_level = "LOW_RISK"

    return {

        "claim_id": str(claim_id),

        "dataset_row": row_number,

        "fraud_probability":
            fraud_probability,

        "fraud_prediction":
            fraud_prediction,

        "evidence_score":
            evidence_score,

        "risk_level":
            risk_level,

        "evidence_reasons":
            reasons,

        "phase3_fusion_available":
            phase3_result is not None,

        "status":
            "ANALYSIS_COMPLETE"
    }
