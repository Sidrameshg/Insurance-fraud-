from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EVIDENCE_PATH = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_evidence_ml.json"
)

OUTPUT_PATH = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features_final.json"
)


def load_evidence():

    with open(
        EVIDENCE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_features(evidence):

    claim = evidence["claim_document"]

    claim_ocr = claim["ocr"]

    claim_doc = claim[
        "document_understanding"
    ]

    invoice = evidence["invoice"]

    invoice_doc = invoice[
        "document_processing"
    ]

    invoice_ocr = invoice_doc["ocr"]

    invoice_understanding = invoice_doc[
        "document_understanding"
    ]

    invoice_extraction = invoice[
        "extraction"
    ]

    signature = evidence[
        "signature_verification"
    ]

    damage = evidence[
        "damage_detection"
    ]

    ml_damage = evidence[
        "damage_ml_classification"
    ]

    ml_prediction = ml_damage[
        "prediction"
    ]

    ml_features = ml_damage[
        "features"
    ]

    # ========================================================
    # BASE FEATURES
    # ========================================================

    features = {

        "document_type_insurance_claim":
            1 if claim_doc[
                "document_type"
            ] == "insurance_claim" else 0,

        "document_confidence":
            claim_doc.get(
                "confidence",
                0.0
            ),

        "document_score_margin":
            claim_doc.get(
                "score_margin",
                0.0
            ),

        "invoice_document_type":
            invoice_understanding.get(
                "document_type",
                "unknown"
            ),

        "invoice_document_confidence":
            invoice_understanding.get(
                "confidence",
                0.0
            ),

        "invoice_total":
            float(
                invoice_extraction.get(
                    "total_amount",
                    0.0
                )
            ),

        "invoice_items_total":
            float(
                invoice_extraction.get(
                    "items_total",
                    invoice_extraction.get(
                        "total_amount",
                        0.0
                    )
                )
            ),

        "invoice_total_valid":
            1 if invoice_extraction[
                "validation"
            ]["total_matches"] else 0,

        "claim_repair_amount":
            85000.0,

        "claim_invoice_amount_difference":
            85000.0
            -
            float(
                invoice_extraction.get(
                    "total_amount",
                    0.0
                )
            ),

        "invoice_to_claim_amount_ratio":
            (
                float(
                    invoice_extraction.get(
                        "total_amount",
                        0.0
                    )
                )
                / 85000.0
            ),

        "signature_similarity":
            float(
                signature.get(
                    "similarity_score",
                    0.0
                )
            ),

        "signature_match":
            1 if signature.get(
                "result"
            ) == "MATCH" else 0,

        "damage_ratio":
            float(
                damage.get(
                    "damage_ratio",
                    0.0
                )
            ),

        "damage_detected":
            1 if damage.get(
                "result"
            ) == "DAMAGE_DETECTED" else 0,

        "damage_mean_pixel_difference":
            float(
                damage.get(
                    "mean_pixel_difference",
                    0.0
                )
            )
    }

    # ========================================================
    # ML DAMAGE FEATURES
    # ========================================================

    features.update({

        "damage_ml_class":
            ml_features[
                "damage_ml_class"
            ],

        "damage_ml_confidence":
            float(
                ml_features[
                    "damage_ml_confidence"
                ]
            ),

        "damage_ml_front":
            ml_features[
                "damage_ml_front"
            ],

        "damage_ml_rear":
            ml_features[
                "damage_ml_rear"
            ],

        "damage_ml_is_damage":
            ml_features[
                "damage_ml_is_damage"
            ],

        "damage_ml_is_breakage":
            ml_features[
                "damage_ml_is_breakage"
            ],

        "damage_ml_is_crushed":
            ml_features[
                "damage_ml_is_crushed"
            ],

        "damage_ml_is_normal":
            ml_features[
                "damage_ml_is_normal"
            ]
    })

    return features


def main():

    print()
    print("==========================================")
    print("       PHASE 3 FINAL FEATURE BUILDER")
    print("==========================================")

    evidence = load_evidence()

    features = build_features(
        evidence
    )

    output = {

        "phase": "phase3",

        "feature_count":
            len(features),

        "base_feature_count":
            16,

        "ml_damage_feature_count":
            8,

        "source":
            str(EVIDENCE_PATH),

        "features":
            features
    }

    if len(features) != 24:

        raise ValueError(
            f"Expected 24 features, "
            f"got {len(features)}"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Base features       :", 16)
    print("ML damage features  :", 8)
    print("Final feature count :", len(features))

    print()
    print("ML class:")
    print(
        features[
            "damage_ml_class"
        ]
    )

    print("ML confidence:")
    print(
        features[
            "damage_ml_confidence"
        ]
    )

    print("Traditional damage:")
    print(
        features[
            "damage_detected"
        ]
    )

    print("ML damage:")
    print(
        features[
            "damage_ml_is_damage"
        ]
    )

    print()
    print("Output:")
    print(OUTPUT_PATH)

    print()
    print("==========================================")


if __name__ == "__main__":
    main()
