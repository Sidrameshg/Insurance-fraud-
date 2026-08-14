from datasets import load_dataset
from pathlib import Path
from collections import Counter
import gc

DATASET_NAME = "DrBimmer/comprehensive-car-damage"

ROOT = Path("./phase3/ml/damage/dataset")

TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"
TEST_DIR = ROOT / "test"

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)
full = dataset["train"]

names = full.features["label"].names

print()
print("Original dataset:")
print(full)

# -----------------------------
# STRATIFIED SPLIT
# -----------------------------

print()
print("Creating stratified split...")

split_1 = full.train_test_split(
    test_size=0.30,
    seed=42,
    stratify_by_column="label"
)

train_ds = split_1["train"]
temp_ds = split_1["test"]

split_2 = temp_ds.train_test_split(
    test_size=0.50,
    seed=42,
    stratify_by_column="label"
)

val_ds = split_2["train"]
test_ds = split_2["test"]

print()
print("===== FINAL SPLIT =====")
print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("Test:", len(test_ds))
print("Total:", len(full))

# -----------------------------
# SAVE ONE IMAGE AT A TIME
# -----------------------------

def save_split(split_dataset, split_name, output_dir):

    print()
    print(f"Saving {split_name} images...")

    counters = Counter()

    for index in range(len(split_dataset)):

        item = split_dataset[index]

        label_id = item["label"]
        label_name = names[label_id]

        class_dir = output_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        output_path = class_dir / f"{split_name}_{index:05d}.jpg"

        # Skip already saved images.
        if output_path.exists():
            counters[label_name] += 1
            continue

        image = item["image"]

        try:

            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(
                output_path,
                "JPEG",
                quality=90,
                optimize=False
            )

        finally:

            image.close()
            del image
            del item

        counters[label_name] += 1

        if (index + 1) % 50 == 0:
            gc.collect()
            print(f"Saved {index + 1}/{len(split_dataset)}")

    gc.collect()

    print(f"{split_name} completed.")

    for label_name in names:
        print(f"  {label_name:<15} {counters[label_name]}")

# -----------------------------
# EXECUTE
# -----------------------------

save_split(train_ds, "train", TRAIN_DIR)

gc.collect()

save_split(val_ds, "val", VAL_DIR)

gc.collect()

save_split(test_ds, "test", TEST_DIR)

gc.collect()

print()
print("======================================")
print("DATASET PREPARATION COMPLETE")
print("======================================")

print()
print("Dataset location:")
print(ROOT.resolve())

print()
print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("Test:", len(test_ds))
print("Total:", len(full))

print()
print("Classes:")

for name in names:
    print(" -", name)

print()
print("The dataset is ready for PyTorch.")
