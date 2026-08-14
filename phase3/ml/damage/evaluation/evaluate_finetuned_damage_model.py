import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from pathlib import Path
import json

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("./phase3/ml/damage/dataset")
MODEL_PATH = Path(
    "./phase3/ml/damage/models/mobilenetv2_damage_finetuned_best.pth"
)
REPORT_DIR = Path("./phase3/ml/damage/evaluation")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cpu")

# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# TEST DATASET
# ============================================================

print("===== LOADING TEST DATASET =====")

test_dataset = datasets.ImageFolder(
    DATASET_ROOT / "test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("Test images:", len(test_dataset))
print("Classes:", class_names)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("===== LOADING BEST MODEL =====")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = models.mobilenet_v2(weights=None)

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model:", checkpoint["architecture"])
print("Best validation F1:", checkpoint["best_val_f1"])

# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_confidences.extend(
            confidences.cpu().tolist()
        )

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)

print()
print("======================================")
print("FINAL TEST RESULTS")
print("======================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

# ============================================================
# PER-CLASS REPORT
# ============================================================

print()
print("===== PER-CLASS RESULTS =====")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("===== CONFUSION MATRIX =====")

print(matrix)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "MobileNetV2",
    "dataset": "Comprehensive Car Damage",
    "test_samples": len(test_dataset),
    "accuracy": accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "class_names": class_names,
    "confusion_matrix": matrix.tolist(),
    "classification_report": classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    ),
    "mean_confidence": sum(all_confidences) / len(
        all_confidences
    )
}

output_path = (
    REPORT_DIR /
    "mobilenetv2_finetuned_test_results.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

print()
print("Test report saved:")
print(output_path)

print()
print("===== EVALUATION COMPLETE =====")
