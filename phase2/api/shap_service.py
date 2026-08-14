# ============================================================
# PHASE 2 - SHAP EXPLANATION SERVICE
# Insurance Claim Fraud Detection
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import shap

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


# ============================================================
# LOAD MODEL
# ============================================================

model = CatBoostClassifier()

model.load_model(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)


print("SHAP explanation service loaded successfully!")
print("Model:", type(model).__name__)
print("Features:", len(feature_names))


# ============================================================
# EXPLANATION FUNCTION
# ============================================================

def explain_claim(claim_data, top_n=10):

    if not claim_data:
        raise ValueError(
            "Claim data cannot be empty."
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    claim_df = pd.DataFrame(
        [claim_data]
    )

    # --------------------------------------------------------
    # Preprocess claim
    # --------------------------------------------------------

    encoded_data = (
        preprocessor.transform(
            claim_df
        )
    )

    # --------------------------------------------------------
    # Convert sparse matrix to dense
    # --------------------------------------------------------

    if hasattr(encoded_data, "toarray"):

        encoded_dense = (
            encoded_data
            .toarray()
        )

    else:

        encoded_dense = np.asarray(
            encoded_data
        )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = (
        explainer
        .shap_values(
            encoded_dense
        )
    )

    # Handle SHAP output formats

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[-1]
        )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 2:

        shap_values = (
            shap_values[0]
        )

    # --------------------------------------------------------
    # Create explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "feature_value": encoded_dense[0]
    })

    explanation[
        "absolute_impact"
    ] = (
        explanation[
            "shap_value"
        ].abs()
    )

    explanation[
        "direction"
    ] = np.where(
        explanation[
            "shap_value"
        ] > 0,
        "Toward FRAUD",
        "Toward LEGITIMATE"
    )

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    explanation = (
        explanation
        .sort_values(
            "absolute_impact",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Convert NumPy values
    # to normal Python values
    # --------------------------------------------------------

    return [
        {
            "feature": str(row["feature"]),
            "shap_value": float(
                row["shap_value"]
            ),
            "feature_value": float(
                row["feature_value"]
            ),
            "direction": str(
                row["direction"]
            )
        }
        for _, row
        in explanation.iterrows()
    ]
