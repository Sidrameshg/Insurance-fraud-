# ============================================================
# PHASE 2 - CATBOOST MODEL SERVICE
# Insurance Claim Fraud Detection
# ============================================================

import os
import joblib
import pandas as pd

from catboost import CatBoostClassifier


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "phase2",
    "catboost_fraud_model.cbm"
)

PREPROCESSOR_PATH = os.path.join(
    PROJECT_DIR,
    "phase2",
    "models",
    "preprocessor_final.pkl"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "phase2",
    "catboost_threshold.pkl"
)


# ============================================================
# CHECK ARTIFACTS
# ============================================================

print("Checking CatBoost artifacts...")

print("Model path:", MODEL_PATH)
print("Preprocessor path:", PREPROCESSOR_PATH)
print("Threshold path:", THRESHOLD_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CatBoost model not found: {MODEL_PATH}"
    )

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(
        f"Threshold not found: {THRESHOLD_PATH}"
    )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = CatBoostClassifier()

model.load_model(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

threshold = joblib.load(
    THRESHOLD_PATH
)

print()
print("CatBoost model service loaded successfully!")
print("Model:", type(model).__name__)
print("Preprocessor:", type(preprocessor).__name__)
print("Threshold:", threshold)


# ============================================================
# REQUIRED CLAIM FIELDS
# ============================================================

required_columns = [
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "WeekOfMonthClaimed",
    "Sex",
    "MaritalStatus",
    "Age",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "PolicyNumber",
    "RepNumber",
    "Deductible",
    "DriverRating",
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
    "Year",
    "BasePolicy",
    "AgeOfPolicyHolder_Clean",
    "NumberOfCars_Clean"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_claim(claim_data):

    if not claim_data:
        raise ValueError(
            "Claim data cannot be empty."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in claim_data
    ]

    if missing_columns:
        raise ValueError(
            "Missing required claim fields: "
            + ", ".join(missing_columns)
        )

    claim_df = pd.DataFrame(
        [claim_data]
    )

    encoded_data = preprocessor.transform(
        claim_df
    )

    probability = float(
        model.predict_proba(
            encoded_data
        )[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    label = (
        "FRAUD"
        if prediction == 1
        else "LEGITIMATE"
    )

    return {
        "fraud_probability": probability,
        "fraud_probability_percent": round(
            probability * 100,
            2
        ),
        "prediction": prediction,
        "label": label,
        "threshold": float(threshold),
        "model": "CatBoost"
    }
