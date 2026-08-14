import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_FEATURES_PATH = Path(
    "./phase3/integration/phase3_features.json"
)

ML_DAMAGE_PATH = Path(
    "./phase3/ml/damage/dataset/test/"
    "F_Breakage/test_00000.damage_ml.json"
)

OUTPUT_PATH = Path(
    "./phase3/integration/phase3_features_ml.json"
)


# ============================================================
# LOAD BASE FEATURE DOCUMENT
# ============================================================

if not BASE_FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Base feature file not found: "
        f"{BASE_FEATURES_PATH}"
    )

with open(
    BASE_FEATURES_PATH,
    "r",
    encoding="utf-8"
) as f:

    base_document = json.load(f)


# ============================================================
# EXTRACT ACTUAL FEATURES
# ============================================================

if "features" not in base_document:
    raise ValueError(
        "Base feature document does not contain "
        "'features'."
    )

base_features = base_document["features"]


# ============================================================
# VERIFY ORIGINAL FEATURE COUNT
# ============================================================

declared_count = base_document.get(
    "feature_count"
)

actual_count = len(base_features)

if actual_count != declared_count:

    raise ValueError(
        f"Feature count mismatch: "
        f"declared={declared_count}, "
        f"actual={actual_count}"
    )


# ============================================================
# LOAD ML DAMAGE RESULT
# ============================================================

if not ML_DAMAGE_PATH.exists():
    raise FileNotFoundError(
        f"ML damage result not found: "
        f"{ML_DAMAGE_PATH}"
    )

with open(
    ML_DAMAGE_PATH,
    "r",
    encoding="utf-8"
) as f:

    ml_damage = json.load(f)


# ============================================================
# BUILD ML DAMAGE FEATURES
# ============================================================

predicted_class = (
    ml_damage["predicted_class"]
)

front_or_rear = (
    ml_damage["front_or_rear"]
)

ml_features = {

    "damage_ml_class":
        predicted_class,

    "damage_ml_confidence":
        float(
            ml_damage["confidence"]
        ),

    "damage_ml_front":
        1 if front_or_rear == "front"
        else 0,

    "damage_ml_rear":
        1 if front_or_rear == "rear"
        else 0,

    "damage_ml_is_damage":
        1 if ml_damage["is_damage"]
        else 0,

    "damage_ml_is_breakage":
        1 if ml_damage["is_breakage"]
        else 0,

    "damage_ml_is_crushed":
        1 if ml_damage["is_crushed"]
        else 0,

    "damage_ml_is_normal":
        1 if ml_damage["is_normal"]
        else 0
}


# ============================================================
# COMBINE FEATURES
# ============================================================

combined_features = dict(
    base_features
)

combined_features.update(
    ml_features
)


# ============================================================
# FINAL COUNT
# ============================================================

final_count = len(
    combined_features
)

expected_count = (
    actual_count +
    len(ml_features)
)

if final_count != expected_count:

    raise ValueError(
        f"Final feature count mismatch: "
        f"expected={expected_count}, "
        f"actual={final_count}"
    )


# ============================================================
# BUILD FINAL DOCUMENT
# ============================================================

output_document = {

    "phase": "phase3",

    "feature_count":
        final_count,

    "base_feature_count":
        actual_count,

    "ml_damage_feature_count":
        len(ml_features),

    "features":
        combined_features
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_document,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print()
print(
    "===== PHASE 3 ML FEATURE INTEGRATION ====="
)

print()

print(
    "Base feature count:",
    actual_count
)

print(
    "ML damage feature count:",
    len(ml_features)
)

print(
    "Final feature count:",
    final_count
)

print()

print(
    "===== ML DAMAGE FEATURES ====="
)

for key, value in ml_features.items():

    print(
        f"{key:<30} {value}"
    )

print()

print(
    "Output file:"
)

print(
    OUTPUT_PATH
)

print()

print(
    "==========================================="
)
