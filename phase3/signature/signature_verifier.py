from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


def load_signature(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Signature image not found: {path}"
        )

    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)

    return np.array(image)


def normalize_signature(image):
    threshold = 200

    binary = image < threshold

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any() or not cols.any():
        return np.zeros((200, 500), dtype=np.uint8)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    row_start = int(row_indices.min())
    row_end = int(row_indices.max()) + 1

    col_start = int(col_indices.min())
    col_end = int(col_indices.max()) + 1

    cropped = binary[
        row_start:row_end,
        col_start:col_end
    ]

    normalized = Image.fromarray(
        (cropped * 255).astype(np.uint8)
    )

    normalized = normalized.resize(
        (500, 200)
    )

    return np.array(normalized)


def calculate_similarity(
    reference_path: str,
    claim_path: str
):

    reference = load_signature(reference_path)
    claim = load_signature(claim_path)

    reference = normalize_signature(reference)
    claim = normalize_signature(claim)

    score = structural_similarity(
        reference,
        claim,
        data_range=255
    )

    return float(score)


def verify_signature(
    reference_path: str,
    claim_path: str,
    threshold: float = 0.80
):

    score = calculate_similarity(
        reference_path,
        claim_path
    )

    result = (
        "MATCH"
        if score >= threshold
        else "MISMATCH"
    )

    return {
        "reference": str(reference_path),
        "claim": str(claim_path),
        "similarity_score": round(score, 4),
        "threshold": threshold,
        "result": result
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:

        print(
            "Usage: python "
            "phase3/signature/signature_verifier.py "
            "<reference> <claim>"
        )

        raise SystemExit(1)

    reference_path = sys.argv[1]
    claim_path = sys.argv[2]

    try:

        result = verify_signature(
            reference_path,
            claim_path
        )

        output_path = Path(
            claim_path
        ).with_suffix(".signature.json")

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
            "\n===== SIGNATURE VERIFICATION =====\n"
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
            "\n==================================\n"
        )

    except Exception as e:

        print(
            f"SIGNATURE VERIFICATION ERROR: {e}"
        )

        raise SystemExit(1)
