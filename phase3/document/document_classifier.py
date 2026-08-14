from pathlib import Path
import json


DOCUMENT_RULES = {
    "invoice": {
        "keywords": {
            "invoice number": 3.0,
            "invoice date": 3.0,
            "total amount": 3.0,
            "repair shop": 3.0,
            "invoice": 1.0
        }
    },

    "insurance_claim": {
        "keywords": {
            "insurance claim": 3.0,
            "claim number": 3.0,
            "accident date": 3.0,
            "claimant": 3.0,
            "policy number": 1.0
        }
    },

    "repair_estimate": {
        "keywords": {
            "repair estimate": 3.0,
            "estimate number": 3.0,
            "estimated cost": 3.0,
            "estimated amount": 3.0
        }
    }
}


def load_ocr_json(ocr_path: str):

    path = Path(ocr_path)

    if not path.exists():
        raise FileNotFoundError(
            f"OCR JSON not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_document(ocr_data):

    text = ocr_data.get("text", "").lower()

    scores = {}

    for document_type, rule_data in DOCUMENT_RULES.items():

        score = 0.0

        for keyword, weight in rule_data["keywords"].items():

            if keyword.lower() in text:
                score += weight

        scores[document_type] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_type, best_score = ranked[0]

    second_score = ranked[1][1]

    total_score = sum(scores.values())

    if best_score == 0:
        document_type = "unknown"
        confidence = 0.0
    else:
        document_type = best_type

        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0

    margin = best_score - second_score

    return {
        "document_type": document_type,
        "confidence": round(confidence, 3),
        "winning_score": best_score,
        "second_best_score": second_score,
        "score_margin": margin,
        "scores": scores
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage: python "
            "phase3/document/document_classifier.py "
            "<ocr_json_path>"
        )

        raise SystemExit(1)

    ocr_path = sys.argv[1]

    try:

        ocr_data = load_ocr_json(ocr_path)

        result = classify_document(ocr_data)

        output_path = Path(ocr_path).with_name(
            Path(ocr_path).stem + ".document.json"
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
            "\n===== DOCUMENT UNDERSTANDING =====\n"
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
            "\n===================================\n"
        )

    except Exception as e:

        print(
            f"DOCUMENT UNDERSTANDING ERROR: {e}"
        )

        raise SystemExit(1)
