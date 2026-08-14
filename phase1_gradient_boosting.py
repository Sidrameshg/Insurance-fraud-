from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

gradient_boosting = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gradient_boosting.fit(
    X_train_smote,
    y_train_smote
)

gb_prob = gradient_boosting.predict_proba(
    X_test_encoded
)[:, 1]

print("Gradient Boosting training completed!")
print("First 10 fraud probabilities:")
print(gb_prob[:10])

gb_threshold_results = []

for t in np.arange(0.05, 0.91, 0.01):

    temp_pred = (gb_prob >= t).astype(int)

    precision = precision_score(
        y_test,
        temp_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        temp_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        temp_pred,
        zero_division=0
    )

    gb_threshold_results.append({
        "threshold": round(t, 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

gb_threshold_results = pd.DataFrame(
    gb_threshold_results
)

best_gb_row = gb_threshold_results.loc[
    gb_threshold_results["f1"].idxmax()
]

print()
print("===== GRADIENT BOOSTING BEST THRESHOLD =====")
print("Best Threshold:", best_gb_row["threshold"])
print("Best Precision:", best_gb_row["precision"])
print("Best Recall:", best_gb_row["recall"])
print("Best F1:", best_gb_row["f1"])
