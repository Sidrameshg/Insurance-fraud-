from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageFilter


def load_image(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = Image.open(path).convert("RGB")

    return image


def preprocess(image):
    """
    Normalize the image before comparison.
    """

    image = image.resize((500, 300))

    image = image.convert("L")

    image = image.filter(
        ImageFilter.GaussianBlur(radius=1)
    )

    return np.array(image, dtype=np.float32)


def detect_damage(
    reference_path: str,
    claim_path: str,
    threshold: float = 30.0
):

    reference = preprocess(
        load_image(reference_path)
    )

    claim = preprocess(
        load_image(claim_path)
    )

    # Pixel-level absolute difference
    difference = np.abs(
        reference - claim
    )

    # Pixels above the difference threshold
    damage_mask = difference > threshold

    total_pixels = damage_mask.size

    damaged_pixels = int(
        np.sum(damage_mask)
    )

    damage_ratio = (
        damaged_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    mean_difference = float(
        np.mean(difference)
    )

    if damage_ratio >= 0.01:
        result = "DAMAGE_DETECTED"
    else:
        result = "NO_SIGNIFICANT_DAMAGE"

    return {
        "reference_image": str(reference_path),
        "claim_image": str(claim_path),
        "difference_threshold": threshold,
        "damaged_pixels": damaged_pixels,
        "total_pixels": int(total_pixels),
        "damage_ratio": round(
            damage_ratio,
            4
        ),
        "mean_pixel_difference": round(
            mean_difference,
            4
        ),
        "result": result
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:

        print(
            "Usage: python "
            "phase3/damage/damage_detector.py "
            "<reference_image> <claim_image>"
        )

        raise SystemExit(1)

    reference_path = sys.argv[1]
    claim_path = sys.argv[2]

    try:

        result = detect_damage(
            reference_path,
            claim_path
        )

        output_path = Path(
            claim_path
        ).with_suffix(
            ".damage.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=2
            )

        print(
            "\n===== DAMAGE DETECTION =====\n"
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        print(
            "\nOutput file:"
        )

        print(output_path)

        print(
            "\n============================\n"
        )

    except Exception as e:

        print(
            f"DAMAGE DETECTION ERROR: {e}"
        )

        raise SystemExit(1)
