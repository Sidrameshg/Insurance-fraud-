import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.ml.damage.pipeline.run_ml_damage_pipeline import (
    run_pipeline
)


BASE_FEATURES_PATH = Path(
    "./phase3/integration/phase3_features.json"
)

OUTPUT_PATH = Path(
    "./phase3/integration/phase3_features_ml_dynamic.json"
)


def load_base_features():

    with open(
        BASE_FEATURES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        document = json.load(f)

    features = document["features"]

    if len(features) != document["feature_count"]:
        raise ValueError(
            "Base feature count mismatch"
        )

    return document, features


def build_dynamic_features(
    vehicle_image
):

    # --------------------------------------------------------
    # Run MobileNetV2 inference dynamically
    # --------------------------------------------------------

    ml_result = run_pipeline(
        vehicle_image
    )

    prediction = ml_result["prediction"]

    ml_features = ml_result["ml_features"]

    return ml_result, ml_features


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "integrate_dynamic_ml_features.py "
            "<vehicle_image>"
        )

        sys.exit(1)

    vehicle_image = Path(
        sys.argv[1]
    )

    if not vehicle_image.exists():

        print(
            f"Vehicle image not found: "
            f"{vehicle_image}"
        )

        sys.exit(1)

    print()
    print(
        "===== DYNAMIC PHASE 3 FEATURE INTEGRATION ====="
    )

    # --------------------------------------------------------
    # Load existing 16 features
    # --------------------------------------------------------

    base_document, base_features = (
        load_base_features()
    )

    print(
        "Base feature count:",
        len(base_features)
    )

    # --------------------------------------------------------
    # Generate ML features dynamically
    # --------------------------------------------------------

    ml_result, ml_features = (
        build_dynamic_features(
            vehicle_image
        )
    )

    print(
        "ML damage feature count:",
        len(ml_features)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined_features = dict(
        base_features
    )

    combined_features.update(
        ml_features
    )

    final_count = len(
        combined_features
    )

    expected_count = (
        len(base_features)
        +
        len(ml_features)
    )

    if final_count != expected_count:

        raise ValueError(
            f"Final feature count mismatch: "
            f"expected={expected_count}, "
            f"actual={final_count}"
        )

    # --------------------------------------------------------
    # Build final document
    # --------------------------------------------------------

    output = {

        "phase": "phase3",

        "feature_count":
            final_count,

        "base_feature_count":
            len(base_features),

        "ml_damage_feature_count":
            len(ml_features),

        "vehicle_image":
            str(vehicle_image),

        "features":
            combined_features
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(
        "===== ML DAMAGE PREDICTION ====="
    )

    print(
        "Class:",
        ml_result["prediction"]["class"]
    )

    print(
        "Confidence:",
        ml_result["prediction"]["confidence"]
    )

    print(
        "Front/Rear:",
        ml_result["prediction"]["front_or_rear"]
    )

    print(
        "Damage type:",
        ml_result["prediction"]["damage_type"]
    )

    print()
    print(
        "===== FINAL FEATURE VECTOR ====="
    )

    print(
        "Base features:",
        len(base_features)
    )

    print(
        "ML features:",
        len(ml_features)
    )

    print(
        "Total features:",
        final_count
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
        "=============================================="
    )


if __name__ == "__main__":
    main()
