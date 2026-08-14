from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "datasets"
)

MODEL_PATH = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "models"
    / "hist_gradient_boosting_fraud.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "evaluation"
    / "hist_gradient_boosting_threshold_analysis.json"
)

TARGET = "FraudFound_P"


def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    }


def main():

    print()
    print("==========================================")
    print(" GRADIENT BOOSTING THRESHOLD ANALYSIS")
    print("==========================================")

    print()
    print("Loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X_val = validation.drop(
        columns=[TARGET]
    )

    y_val = validation[TARGET]

    X_test = test.drop(
        columns=[TARGET]
    )

    y_test = test[TARGET]

    validation_probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    thresholds = [
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
        0.22,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    validation_results = []

    for threshold in thresholds:

        validation_results.append(
            evaluate_threshold(
                y_val,
                validation_probabilities,
                threshold
            )
        )

    best_validation = max(
        validation_results,
        key=lambda x: x["f1"]
    )

    best_threshold = (
        best_validation["threshold"]
    )

    test_result = evaluate_threshold(
        y_test,
        test_probabilities,
        best_threshold
    )

    print()
    print("===== VALIDATION THRESHOLD RESULTS =====")

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
    )

    for result in validation_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['false_positive']:<10}"
            f"{result['false_negative']:<10}"
        )

    print()
    print("===== BEST VALIDATION THRESHOLD =====")

    print(
        "Threshold :",
        best_threshold
    )

    print(
        "Precision :",
        round(
            best_validation["precision"],
            4
        )
    )

    print(
        "Recall    :",
        round(
            best_validation["recall"],
            4
        )
    )

    print(
        "F1        :",
        round(
            best_validation["f1"],
            4
        )
    )

    print(
        "False Positives:",
        best_validation["false_positive"]
    )

    print(
        "False Negatives:",
        best_validation["false_negative"]
    )

    print()
    print("===== TEST AT SELECTED THRESHOLD =====")

    print(
        "Threshold :",
        best_threshold
    )

    print(
        "Precision :",
        round(
            test_result["precision"],
            4
        )
    )

    print(
        "Recall    :",
        round(
            test_result["recall"],
            4
        )
    )

    print(
        "F1        :",
        round(
            test_result["f1"],
            4
        )
    )

    print(
        "True Negatives :",
        test_result["true_negative"]
    )

    print(
        "False Positives:",
        test_result["false_positive"]
    )

    print(
        "False Negatives:",
        test_result["false_negative"]
    )

    print(
        "True Positives :",
        test_result["true_positive"]
    )

    output = {

        "model":
            "HistGradientBoostingClassifier",

        "selection_metric":
            "validation_f1",

        "best_threshold":
            best_threshold,

        "validation_results":
            validation_results,

        "selected_validation_result":
            best_validation,

        "test_result":
            test_result
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print("Results saved:")
    print(OUTPUT_PATH)

    print()
    print("==========================================")


if __name__ == "__main__":
    main()
