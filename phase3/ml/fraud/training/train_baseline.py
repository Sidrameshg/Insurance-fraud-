from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "datasets"
)

MODEL_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "models"
)

EVAL_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "evaluation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "FraudFound_P"


def load_data():

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    return train, validation, test


def build_preprocessor(X):

    numeric_columns = (
        X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns
    )


def evaluate_model(
    model,
    X,
    y,
    name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print(
        f"===== {name.upper()} RESULTS ====="
    )

    print(
        "Accuracy  :",
        round(accuracy, 4)
    )

    print(
        "Precision :",
        round(precision, 4)
    )

    print(
        "Recall    :",
        round(recall, 4)
    )

    print(
        "F1        :",
        round(f1, 4)
    )

    print(
        "ROC-AUC   :",
        round(roc_auc, 4)
    )

    print(
        "PR-AUC    :",
        round(pr_auc, 4)
    )

    print()
    print("Confusion Matrix:")

    print(cm)

    print()
    print("Classification Report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


def main():

    print()
    print("==========================================")
    print("       FRAUD ML BASELINE MODEL")
    print("==========================================")

    # ========================================================
    # LOAD
    # ========================================================

    train, validation, test = load_data()

    print()
    print("Train rows      :", len(train))
    print("Validation rows :", len(validation))
    print("Test rows       :", len(test))

    # ========================================================
    # FEATURES / TARGET
    # ========================================================

    X_train = train.drop(
        columns=[TARGET]
    )

    y_train = train[TARGET]

    X_val = validation.drop(
        columns=[TARGET]
    )

    y_val = validation[TARGET]

    X_test = test.drop(
        columns=[TARGET]
    )

    y_test = test[TARGET]

    print()
    print("Input features:", X_train.shape[1])

    # ========================================================
    # PREPROCESSOR
    # ========================================================

    (
        preprocessor,
        numeric_columns,
        categorical_columns
    ) = build_preprocessor(
        X_train
    )

    print()
    print(
        "Numeric columns:",
        len(numeric_columns)
    )

    print(
        "Categorical columns:",
        len(categorical_columns)
    )

    # ========================================================
    # MODEL
    # ========================================================

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("Training Logistic Regression...")

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    elapsed = time.time() - start

    print(
        "Training time:",
        round(elapsed, 2),
        "seconds"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation"
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR
        / "logistic_regression_baseline.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "model":
            "LogisticRegression",

        "model_type":
            "baseline",

        "class_weight":
            "balanced",

        "target":
            TARGET,

        "train_size":
            len(train),

        "validation_size":
            len(validation),

        "test_size":
            len(test),

        "input_feature_count":
            X_train.shape[1],

        "numeric_feature_count":
            len(numeric_columns),

        "categorical_feature_count":
            len(categorical_columns),

        "validation":
            validation_results,

        "test":
            test_results
    }

    results_path = (
        EVAL_DIR
        / "logistic_regression_baseline_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print()
    print("===== MODEL SAVED =====")

    print(
        model_path
    )

    print()
    print("===== RESULTS SAVED =====")

    print(
        results_path
    )

    print()
    print("==========================================")


if __name__ == "__main__":
    main()
