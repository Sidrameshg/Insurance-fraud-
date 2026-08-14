import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter

DATASET_ROOT = Path("./phase3/ml/damage/dataset")

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 16

# -----------------------------
# TRANSFORMS
# -----------------------------

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
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# DATASETS
# -----------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

print("===== DATASET CLASSES =====")
print(train_dataset.classes)

print()
print("===== CLASS INDICES =====")
print(train_dataset.class_to_idx)

print()
print("===== DATASET SIZES =====")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# -----------------------------
# CLASS COUNTS
# -----------------------------

counts = Counter(train_dataset.targets)

print()
print("===== TRAIN CLASS COUNTS =====")

for class_index, class_name in enumerate(train_dataset.classes):
    print(
        f"{class_name:<15} "
        f"{counts[class_index]}"
    )

# -----------------------------
# CLASS WEIGHTS
# -----------------------------

total = len(train_dataset)
num_classes = len(train_dataset.classes)

class_weights = []

for class_index in range(num_classes):
    count = counts[class_index]

    weight = total / (num_classes * count)

    class_weights.append(weight)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print()
print("===== CLASS WEIGHTS =====")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(
        f"{class_name:<15} "
        f"{weight.item():.4f}"
    )

# -----------------------------
# DATALOADERS
# -----------------------------

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# TEST ONE BATCH
# -----------------------------

images, labels = next(iter(train_loader))

print()
print("===== FIRST BATCH =====")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Labels:", labels.tolist())

print()
print("===== DATALOADER READY =====")
print("Batch size:", BATCH_SIZE)
print("Image size:", IMAGE_SIZE)
print("Device: CPU")
