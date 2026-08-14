from pathlib import Path
import joblib
from catboost import CatBoostClassifier

# ============================================================
# CATBOOST MODEL SERVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "phase2"
    / "catboost_fraud_model.cbm"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "phase2"
    / "catboost_threshold.pkl"
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)

# ------------------------------------------------------------
# Load threshold
# ------------------------------------------------------------

threshold = joblib.load(
    THRESHOLD_PATH
)

print("CatBoost model service loaded successfully!")
print("Model:", type(model).__name__)
print("Threshold:", threshold)


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_processed_claim(X):
    """
    Predict fraud probability for already-preprocessed data.
    """

    probability = float(
        model.predict_proba(X)[0][1]
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
        "threshold": float(threshold)
    }
