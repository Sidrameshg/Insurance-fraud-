from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

EVAL_DIR = (
    ROOT
    / "phase3"
    / "ml"
    / "fraud"
    / "evaluation"
)

OUTPUT_PATH = (
    EVAL_DIR
    / "fraud_model_comparison.json"
)


models = []


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic_results = EVAL_DIR / "logistic_regression_baseline_results.json"
logistic_threshold = EVAL_DIR / "threshold_analysis.json"

if logistic_results.exists():

    with open(
        logistic_results,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    test = data.get("test", data)

    threshold = None

    if logistic_threshold.exists():

        with open(
            logistic_threshold,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        threshold = threshold_data.get(
            "best_threshold"
        )

    models.append({

        "model":
            "Logistic Regression",

        "test_roc_auc":
            test.get("roc_auc"),

        "test_pr_auc":
            test.get("pr_auc"),

        "test_precision":
            test.get("precision"),

        "test_recall":
            test.get("recall"),

        "test_f1_default":
            test.get("f1"),

        "selected_threshold":
            threshold
    })


# ============================================================
# RANDOM FOREST
# ============================================================

rf_results = (
    EVAL_DIR
    / "random_forest_results.json"
)

rf_threshold = (
    EVAL_DIR
    / "random_forest_threshold_analysis.json"
)

if rf_results.exists():

    with open(
        rf_results,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    test = data["test"]

    threshold = None
    threshold_result = {}

    if rf_threshold.exists():

        with open(
            rf_threshold,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        threshold = threshold_data.get(
            "best_threshold"
        )

        threshold_result = threshold_data.get(
            "test_result",
            {}
        )

    models.append({

        "model":
            "Random Forest",

        "test_roc_auc":
            test.get("roc_auc"),

        "test_pr_auc":
            test.get("pr_auc"),

        "test_precision":
            threshold_result.get(
                "precision",
                test.get("precision")
            ),

        "test_recall":
            threshold_result.get(
                "recall",
                test.get("recall")
            ),

        "test_f1_default":
            test.get("f1"),

        "test_f1_threshold":
            threshold_result.get(
                "f1"
            ),

        "selected_threshold":
            threshold
    })


# ============================================================
# GRADIENT BOOSTING
# ============================================================

gb_results = (
    EVAL_DIR
    / "hist_gradient_boosting_results.json"
)

gb_threshold = (
    EVAL_DIR
    / "hist_gradient_boosting_threshold_analysis.json"
)

if gb_results.exists():

    with open(
        gb_results,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    test = data["test"]

    threshold = None
    threshold_result = {}

    if gb_threshold.exists():

        with open(
            gb_threshold,
            "r",
            encoding="utf-8"
        ) as file:

            threshold_data = json.load(file)

        threshold = threshold_data.get(
            "best_threshold"
        )

        threshold_result = threshold_data.get(
            "test_result",
            {}
        )

    models.append({

        "model":
            "HistGradientBoosting",

        "test_roc_auc":
            test.get("roc_auc"),

        "test_pr_auc":
            test.get("pr_auc"),

        "test_precision":
            threshold_result.get(
                "precision",
                test.get("precision")
            ),

        "test_recall":
            threshold_result.get(
                "recall",
                test.get("recall")
            ),

        "test_f1_default":
            test.get("f1"),

        "test_f1_threshold":
            threshold_result.get(
                "f1"
            ),

        "selected_threshold":
            threshold
    })


# ============================================================
# SORT BY THRESHOLD F1
# ============================================================

models = sorted(
    models,
    key=lambda x:
        x.get(
            "test_f1_threshold",
            x.get(
                "test_f1_default",
                0
            )
        ),
    reverse=True
)


# ============================================================
# PRINT
# ============================================================

print()
print("==========================================")
print("       FRAUD MODEL COMPARISON")
print("==========================================")

print()

print(
    f"{'Model':<28}"
    f"{'ROC-AUC':<12}"
    f"{'PR-AUC':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Threshold':<12}"
)

for model in models:

    f1 = model.get(
        "test_f1_threshold",
        model.get(
            "test_f1_default"
        )
    )

    print(
        f"{model['model']:<28}"
        f"{model.get('test_roc_auc', 0):<12.4f}"
        f"{model.get('test_pr_auc', 0):<12.4f}"
        f"{model.get('test_precision', 0):<12.4f}"
        f"{model.get('test_recall', 0):<12.4f}"
        f"{f1:<12.4f}"
        f"{str(model.get('selected_threshold')):<12}"
    )


# ============================================================
# BEST MODEL
# ============================================================

if models:

    best_model = models[0]

    print()
    print("===== CURRENT BEST MODEL =====")

    print(
        "Model:",
        best_model["model"]
    )

    print(
        "ROC-AUC:",
        best_model.get(
            "test_roc_auc"
        )
    )

    print(
        "PR-AUC:",
        best_model.get(
            "test_pr_auc"
        )
    )

    print(
        "Threshold:",
        best_model.get(
            "selected_threshold"
        )
    )

    print(
        "F1:",
        best_model.get(
            "test_f1_threshold",
            best_model.get(
                "test_f1_default"
            )
        )
    )


# ============================================================
# SAVE
# ============================================================

output = {

    "phase":
        "phase3",

    "comparison_metric":
        "test_f1_at_validation_selected_threshold",

    "models":
        models,

    "current_best_model":
        models[0]["model"]
        if models
        else None
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
print("Comparison saved:")
print(OUTPUT_PATH)

print()
print("==========================================")
