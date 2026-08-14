import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("./phase3/ml/damage/dataset")
MODEL_DIR = Path("./phase3/ml/damage/models")
REPORT_DIR = Path("./phase3/ml/damage/evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal",
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# DATASETS
# ============================================================

print("===== LOADING DATASETS =====")

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_ROOT / "val",
    transform=eval_transform
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = torch.bincount(
    torch.tensor(train_dataset.targets),
    minlength=NUM_CLASSES
)

total_samples = len(train_dataset)

class_weights = total_samples / (
    NUM_CLASSES * class_counts.float()
)

class_weights = class_weights.to(DEVICE)

print()
print("===== CLASS WEIGHTS =====")

for name, count, weight in zip(
    CLASS_NAMES,
    class_counts.tolist(),
    class_weights.tolist()
):
    print(
        f"{name:<15} "
        f"count={count:<4} "
        f"weight={weight:.4f}"
    )

# ============================================================
# MODEL
# ============================================================

print()
print("===== LOADING MOBILENETV2 =====")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace classifier
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)

print()
print("===== TRAINING CONFIGURATION =====")
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Optimizer: Adam")
print("Loss: Weighted CrossEntropyLoss")

# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples_eval = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            batch_size_actual = labels.size(0)

            total_loss += (
                loss.item() * batch_size_actual
            )

            total_samples_eval += batch_size_actual

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    avg_loss = (
        total_loss / total_samples_eval
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }

# ============================================================
# TRAINING
# ============================================================

best_f1 = -1.0
history = []

print()
print("======================================")
print("STARTING TRAINING")
print("======================================")

for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    validation = evaluate()

    elapsed = time.time() - start_time

    epoch_result = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": validation["loss"],
        "val_accuracy": validation["accuracy"],
        "val_precision": validation["precision"],
        "val_recall": validation["recall"],
        "val_f1": validation["f1"],
        "time_seconds": elapsed,
    }

    history.append(epoch_result)

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss: "
        f"{validation['loss']:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{validation['accuracy']:.4f}"
    )

    print(
        f"Val Precision: "
        f"{validation['precision']:.4f}"
    )

    print(
        f"Val Recall: "
        f"{validation['recall']:.4f}"
    )

    print(
        f"Val F1: "
        f"{validation['f1']:.4f}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f}s"
    )

    # Save best model based on macro F1
    if validation["f1"] > best_f1:

        best_f1 = validation["f1"]

        best_model_path = (
            MODEL_DIR /
            "mobilenetv2_damage_best.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "architecture": "MobileNetV2",
                "image_size": IMAGE_SIZE,
                "best_val_f1": best_f1,
            },
            best_model_path
        )

        print(
            "✓ New best model saved"
        )

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    REPORT_DIR /
    "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    f"Best Validation F1: "
    f"{best_f1:.4f}"
)

print()
print("Best model:")
print(
    MODEL_DIR /
    "mobilenetv2_damage_best.pth"
)

print()
print("Training history:")
print(history_path)
