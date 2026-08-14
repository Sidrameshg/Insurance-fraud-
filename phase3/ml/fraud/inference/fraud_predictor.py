from pathlib import Path
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "models"
    / "hist_gradient_boosting_fraud.joblib"
)


def load_model():

    return joblib.load(
        MODEL_PATH
    )


def get_required_features():

    model = load_model()

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    numeric_features = (
        preprocessor.transformers_[0][2]
    )

    categorical_features = (
        preprocessor.transformers_[1][2]
    )

    return (
        list(numeric_features)
        +
        list(categorical_features)
    )


def predict_fraud(claim_data):

    model = load_model()

    if isinstance(
        claim_data,
        dict
    ):

        claim_data = pd.DataFrame(
            [claim_data]
        )

    required_features = (
        get_required_features()
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in claim_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing fraud model features: "
            + str(missing_features)
        )

    claim_data = claim_data[
        required_features
    ]

    probability = float(
        model.predict_proba(
            claim_data
        )[0, 1]
    )

    return {

        "model":
            "HistGradientBoosting",

        "fraud_probability":
            probability,

        "feature_count":
            len(required_features)
    }


if __name__ == "__main__":

    model = load_model()

    features = (
        get_required_features()
    )

    print(
        "HistGradientBoosting model loaded successfully."
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Required features:",
        len(features)
    )

    print()
    print(
        "Feature validation ready."
    )
