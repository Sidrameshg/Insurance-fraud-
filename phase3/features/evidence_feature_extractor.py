from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_evidence.json"
)

OUTPUT_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features.json"
)


def load_evidence():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def find_numeric_value(text, label):

    import re

    pattern = rf"{label}\s*:\s*([0-9]+(?:\.[0-9]+)?)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


def extract_claim_repair_amount(claim_document):

    ocr = claim_document.get(
        "ocr",
        {}
    )

    text = ocr.get(
        "text",
        ""
    )

    value = find_numeric_value(
        text,
        "Repair Amount"
    )

    return value


def build_features(evidence):

    claim_document = evidence[
        "claim_document"
    ]

    document_understanding = (
        claim_document[
            "document_understanding"
        ]
    )

    invoice = evidence[
        "invoice"
    ]

    invoice_processing = invoice[
        "document_processing"
    ]

    invoice_understanding = (
        invoice_processing[
            "document_understanding"
        ]
    )

    invoice_extraction = invoice[
        "extraction"
    ]

    invoice_validation = (
        invoice_extraction[
            "validation"
        ]
    )

    signature = evidence[
        "signature_verification"
    ]

    damage = evidence[
        "damage_detection"
    ]

    claim_repair_amount = (
        extract_claim_repair_amount(
            claim_document
        )
    )

    invoice_total = (
        invoice_extraction[
            "total_amount"
        ]
    )

    if (
        claim_repair_amount is not None
        and invoice_total is not None
    ):

        amount_difference = (
            claim_repair_amount
            - invoice_total
        )

        amount_ratio = (
            invoice_total
            / claim_repair_amount
            if claim_repair_amount != 0
            else 0
        )

    else:

        amount_difference = None
        amount_ratio = None

    features = {

        "document_type_insurance_claim": int(
            document_understanding[
                "document_type"
            ] == "insurance_claim"
        ),

        "document_confidence": float(
            document_understanding[
                "confidence"
            ]
        ),

        "document_score_margin": float(
            document_understanding.get(
                "score_margin",
                0
            )
        ),

        "invoice_document_type": (
            invoice_understanding[
                "document_type"
            ]
        ),

        "invoice_document_confidence": float(
            invoice_understanding[
                "confidence"
            ]
        ),

        "invoice_total": float(
            invoice_total
        ),

        "invoice_items_total": float(
            invoice_validation[
                "calculated_items_total"
            ]
        ),

        "invoice_total_valid": int(
            invoice_validation[
                "total_matches"
            ]
        ),

        "claim_repair_amount": (
            claim_repair_amount
        ),

        "claim_invoice_amount_difference": (
            amount_difference
        ),

        "invoice_to_claim_amount_ratio": (
            amount_ratio
        ),

        "signature_similarity": float(
            signature[
                "similarity_score"
            ]
        ),

        "signature_match": int(
            signature[
                "result"
            ] == "MATCH"
        ),

        "damage_ratio": float(
            damage[
                "damage_ratio"
            ]
        ),

        "damage_detected": int(
            damage[
                "result"
            ] == "DAMAGE_DETECTED"
        ),

        "damage_mean_pixel_difference": float(
            damage[
                "mean_pixel_difference"
            ]
        )
    }

    return features


def main():

    print()
    print("==========================================")
    print("       PHASE 3 EVIDENCE FEATURES")
    print("==========================================")
    print()

    evidence = load_evidence()

    features = build_features(
        evidence
    )

    output = {
        "phase": "phase3",
        "feature_count": len(features),
        "features": features
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print("Features generated:")
    print()

    for name, value in features.items():

        print(
            f"{name:<40} {value}"
        )

    print()
    print("------------------------------------------")
    print(
        f"Feature count: {len(features)}"
    )
    print("------------------------------------------")
    print()
    print(
        "Output file:"
    )
    print(OUTPUT_FILE)
    print()
    print("==========================================")
    print()


if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print(
            "\nFEATURE EXTRACTION ERROR:",
            e
        )

        raise SystemExit(1)
