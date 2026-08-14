from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(
    0,
    str(ROOT)
)


from phase3.ml.fraud.inference.fraud_predictor import (
    get_required_features,
    predict_fraud
)


NUMERIC_FEATURES = [
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Age",
    "RepNumber",
    "Deductible",
    "DriverRating",
    "Year",
    "AgeOfPolicyHolder_Clean",
    "NumberOfCars_Clean"
]


CATEGORICAL_FEATURES = [
    "Month",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "BasePolicy"
]


def validate_claim(claim):

    required_features = (
        get_required_features()
    )

    missing = [
        feature
        for feature in required_features
        if feature not in claim
    ]

    if missing:

        raise ValueError(
            "Missing claim features: "
            + str(missing)
        )

    return True


def build_claim_dataframe(claim):

    validate_claim(
        claim
    )

    data = {}

    for feature in NUMERIC_FEATURES:

        value = claim[feature]

        if value is None or value == "":

            data[feature] = None

        else:

            data[feature] = float(
                value
            )

    for feature in CATEGORICAL_FEATURES:

        value = claim[feature]

        if value is None:

            data[feature] = None

        else:

            data[feature] = str(
                value
            )

    return pd.DataFrame(
        [data]
    )


def predict_claim(claim):

    claim_dataframe = (
        build_claim_dataframe(
            claim
        )
    )

    return predict_fraud(
        claim_dataframe
    )


if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )
    print(
        "       CLAIM ADAPTER READY"
    )
    print(
        "=========================================="
    )

    print()
    print(
        "Numeric features:",
        len(NUMERIC_FEATURES)
    )

    print(
        "Categorical features:",
        len(CATEGORICAL_FEATURES)
    )

    print(
        "Total features:",
        len(
            NUMERIC_FEATURES
            +
            CATEGORICAL_FEATURES
        )
    )

    print()
    print(
        "Required model features:",
        len(
            get_required_features()
        )
    )

    print()
    print(
        "Claim adapter validation ready."
    )

    print(
        "=========================================="
    )
