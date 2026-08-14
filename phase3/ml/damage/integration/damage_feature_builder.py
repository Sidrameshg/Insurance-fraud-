import json
from pathlib import Path


ML_DAMAGE_FEATURE_NAMES = [
    "damage_ml_class",
    "damage_ml_confidence",
    "damage_ml_front",
    "damage_ml_rear",
    "damage_ml_is_damage",
    "damage_ml_is_breakage",
    "damage_ml_is_crushed",
    "damage_ml_is_normal",
]


def load_ml_damage_result(json_path):

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"ML damage result not found: {json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def build_ml_damage_features(result):

    predicted_class = result["predicted_class"]

    front_or_rear = result["front_or_rear"]

    features = {

        "damage_ml_class":
            predicted_class,

        "damage_ml_confidence":
            float(
                result["confidence"]
            ),

        "damage_ml_front":
            1 if front_or_rear == "front"
            else 0,

        "damage_ml_rear":
            1 if front_or_rear == "rear"
            else 0,

        "damage_ml_is_damage":
            1 if result["is_damage"]
            else 0,

        "damage_ml_is_breakage":
            1 if result["is_breakage"]
            else 0,

        "damage_ml_is_crushed":
            1 if result["is_crushed"]
            else 0,

        "damage_ml_is_normal":
            1 if result["is_normal"]
            else 0,
    }

    return features


def load_and_build_ml_damage_features(
    json_path
):

    result = load_ml_damage_result(
        json_path
    )

    return build_ml_damage_features(
        result
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "damage_feature_builder.py "
            "<damage_ml_json>"
        )

        raise SystemExit(1)

    result_path = sys.argv[1]

    features = (
        load_and_build_ml_damage_features(
            result_path
        )
    )

    print()
    print(
        "===== ML DAMAGE FEATURES ====="
    )

    print(
        json.dumps(
            features,
            indent=2
        )
    )

    print()
    print(
        "Feature count:",
        len(features)
    )
