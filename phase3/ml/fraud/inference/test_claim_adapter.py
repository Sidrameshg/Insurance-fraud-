from pathlib import Path
import sys

import pandas as pd


# -------------------------------------------------
# Project root
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------------------------------------------------
# Phase 3 fraud modules
# -------------------------------------------------

from phase3.ml.fraud.inference.claim_adapter import (
    predict_claim
)


# -------------------------------------------------
# Test dataset
# -------------------------------------------------

TEST_FILE = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "datasets"
    / "test.csv"
)


ROW_NUMBER = 50


def main():

    print()
    print("==========================================")
    print("       CLAIM ADAPTER END-TO-END TEST")
    print("==========================================")

    # ---------------------------------------------
    # Load dataset
    # ---------------------------------------------

    data = pd.read_csv(
        TEST_FILE
    )

    row = data.iloc[
        ROW_NUMBER
    ]

    actual_label = int(
        row["FraudFound_P"]
    )

    claim = row.drop(
        labels=["FraudFound_P"]
    ).to_dict()

    print()
    print(
        "Test row:",
        ROW_NUMBER
    )

    print(
        "Actual fraud label:",
        actual_label
    )

    print(
        "Claim features:",
        len(claim)
    )

    # ---------------------------------------------
    # Dynamic prediction through adapter
    # ---------------------------------------------

    result = predict_claim(
        claim
    )

    probability = result[
        "fraud_probability"
    ]

    print()
    print("===== ADAPTER PREDICTION =====")

    print(
        "Model:",
        result["model"]
    )

    print(
        "Feature count:",
        result["feature_count"]
    )

    print(
        "Fraud probability:",
        round(
            probability,
            6
        )
    )

    # ---------------------------------------------
    # Compare with previous result
    # ---------------------------------------------

    expected_probability = 0.014126

    print()
    print("===== EXPECTED =====")

    print(
        "Previous row 50 probability:",
        expected_probability
    )

    difference = abs(
        probability
        -
        expected_probability
    )

    print()
    print("===== RESULT =====")

    print(
        "Probability difference:",
        difference
    )

    if difference < 0.000001:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: CHECK"
        )

    print()
    print("==========================================")


if __name__ == "__main__":

    main()
