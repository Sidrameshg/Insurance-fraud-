import sys
import json
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from phase3.ml.damage.inference.damage_classifier import (
    classify_image
)

from phase3.ml.damage.integration.damage_feature_builder import (
    build_ml_damage_features
)


def run_pipeline(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Vehicle image not found: {image_path}"
        )

    # ========================================================
    # STEP 1: ML CLASSIFICATION
    # ========================================================

    ml_result = classify_image(
        image_path
    )

    # ========================================================
    # STEP 2: BUILD ML FEATURES
    # ========================================================

    ml_features = build_ml_damage_features(
        ml_result
    )

    # ========================================================
    # STEP 3: COMBINE OUTPUT
    # ========================================================

    output = {
        "image": str(image_path),

        "model": ml_result["model"],

        "model_type": ml_result["model_type"],

        "prediction": {
            "class":
                ml_result["predicted_class"],

            "confidence":
                ml_result["confidence"],

            "front_or_rear":
                ml_result["front_or_rear"],

            "damage_type":
                ml_result["damage_type"],

            "is_damage":
                ml_result["is_damage"]
        },

        "ml_features":
            ml_features
    }

    return output


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print(
            "python run_ml_damage_pipeline.py "
            "<vehicle_image>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = run_pipeline(
            image_path
        )

        print()
        print(
            "===== DYNAMIC ML DAMAGE PIPELINE ====="
        )

        print()

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        image_path = Path(
            image_path
        )

        output_path = (
            image_path.parent /
            f"{image_path.stem}.ml_pipeline.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        print()
        print("Output file:")
        print(output_path)

        print()
        print(
            "======================================="
        )

    except Exception as error:

        print()
        print(
            "ML DAMAGE PIPELINE ERROR:",
            error
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
