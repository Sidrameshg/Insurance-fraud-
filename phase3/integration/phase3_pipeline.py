from pathlib import Path
import sys
import json

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# PHASE 3 MODULES
# ============================================================

from phase3.ocr.ocr_v3 import extract_ocr
from phase3.invoice.invoice_extractor import extract_invoice
from phase3.document.document_classifier import classify_document
from phase3.signature.signature_verifier import verify_signature
from phase3.damage.damage_detector import detect_damage

from phase3.ml.damage.pipeline.run_ml_damage_pipeline import (
    run_pipeline as run_ml_damage_pipeline
)

from phase3.ml.damage.integration.ml_damage_confidence import (
    evaluate_ml_confidence,
    evaluate_damage_consistency
)

# ============================================================
# TEST INPUTS
# ============================================================

CLAIM_IMAGE = (
    ROOT / "phase3" / "ocr" / "test_document.png"
)

INVOICE_IMAGE = (
    ROOT / "phase3" / "invoice" / "test_invoice.png"
)

REFERENCE_SIGNATURE = (
    ROOT / "phase3" / "signature" / "signature_reference.png"
)

CLAIM_SIGNATURE = (
    ROOT / "phase3" / "signature" / "signature_claim.png"
)

REFERENCE_VEHICLE = (
    ROOT / "phase3" / "damage" / "vehicle_reference.png"
)

DAMAGED_VEHICLE = (
    ROOT / "phase3" / "damage" / "vehicle_damaged.png"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT / "phase3" / "integration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(data, output_path):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def process_document(image_path):

    ocr_result = extract_ocr(
        str(image_path)
    )

    document_result = classify_document(
        ocr_result
    )

    return {
        "ocr": ocr_result,
        "document_understanding": document_result
    }


def main():

    print()
    print("==========================================")
    print("       PHASE 3 INTEGRATED PIPELINE")
    print("==========================================")
    print()

    # ========================================================
    # 1. CLAIM DOCUMENT
    # ========================================================

    print("[1/5] Processing claim document...")

    claim_result = process_document(
        CLAIM_IMAGE
    )

    claim_type = (
        claim_result[
            "document_understanding"
        ]["document_type"]
    )

    print(
        "      Document type:",
        claim_type
    )

    # ========================================================
    # 2. INVOICE
    # ========================================================

    print()
    print("[2/5] Processing invoice...")

    invoice_document = process_document(
        INVOICE_IMAGE
    )

    invoice_type = (
        invoice_document[
            "document_understanding"
        ]["document_type"]
    )

    invoice_extraction = extract_invoice(
        invoice_document["ocr"]
    )

    print(
        "      Document type:",
        invoice_type
    )

    print(
        "      Vendor:",
        invoice_extraction["vendor"]
    )

    print(
        "      Invoice total:",
        invoice_extraction["total_amount"]
    )

    print(
        "      Total validation:",
        invoice_extraction[
            "validation"
        ]["total_matches"]
    )

    # ========================================================
    # 3. SIGNATURE
    # ========================================================

    print()
    print("[3/5] Verifying signature...")

    signature_result = verify_signature(
        str(REFERENCE_SIGNATURE),
        str(CLAIM_SIGNATURE)
    )

    print(
        "      Similarity:",
        signature_result["similarity_score"]
    )

    print(
        "      Result:",
        signature_result["result"]
    )

    # ========================================================
    # 4. TRADITIONAL DAMAGE
    # ========================================================

    print()
    print(
        "[4/5] Detecting vehicle damage "
        "using traditional detector..."
    )

    damage_result = detect_damage(
        str(REFERENCE_VEHICLE),
        str(DAMAGED_VEHICLE)
    )

    traditional_damage = (
        damage_result["result"] == "DAMAGE_DETECTED"
    )

    print(
        "      Damage ratio:",
        damage_result["damage_ratio"]
    )

    print(
        "      Result:",
        damage_result["result"]
    )

    # ========================================================
    # 5. ML DAMAGE
    # ========================================================

    print()
    print(
        "[5/5] Classifying vehicle damage "
        "using MobileNetV2..."
    )

    ml_damage_result = run_ml_damage_pipeline(
        DAMAGED_VEHICLE
    )

    ml_prediction = (
        ml_damage_result["prediction"]
    )

    ml_features = (
        ml_damage_result["ml_features"]
    )

    ml_confidence = float(
        ml_prediction["confidence"]
    )

    ml_is_damage = bool(
        ml_prediction["is_damage"]
    )

    confidence_status = evaluate_ml_confidence(
        ml_confidence
    )

    consistency = evaluate_damage_consistency(
        traditional_damage,
        ml_is_damage,
        ml_confidence
    )

    print(
        "      Model:",
        ml_damage_result["model"]
    )

    print(
        "      Model type:",
        ml_damage_result["model_type"]
    )

    print(
        "      Predicted class:",
        ml_prediction["class"]
    )

    print(
        "      Confidence:",
        ml_confidence
    )

    print(
        "      Confidence status:",
        confidence_status
    )

    print(
        "      Front/Rear:",
        ml_prediction["front_or_rear"]
    )

    print(
        "      Damage type:",
        ml_prediction["damage_type"]
    )

    print(
        "      Is damage:",
        ml_is_damage
    )

    print(
        "      Traditional/ML consistency:",
        consistency
    )

    # ========================================================
    # UNIFIED EVIDENCE
    # ========================================================

    evidence = {

        "phase": "phase3",

        "claim_document":
            claim_result,

        "invoice": {

            "document_processing":
                invoice_document,

            "extraction":
                invoice_extraction
        },

        "signature_verification":
            signature_result,

        "damage_detection":
            damage_result,

        "damage_ml_classification": {

            "model":
                ml_damage_result["model"],

            "model_type":
                ml_damage_result["model_type"],

            "image":
                ml_damage_result["image"],

            "prediction":
                ml_prediction,

            "confidence_status":
                confidence_status,

            "consistency":
                consistency,

            "features":
                ml_features
        },

        "summary": {

            "claim_document_type":
                claim_type,

            "invoice_document_type":
                invoice_type,

            "invoice_total_valid":
                invoice_extraction[
                    "validation"
                ]["total_matches"],

            "signature_result":
                signature_result["result"],

            "damage_result":
                damage_result["result"],

            "damage_ml_class":
                ml_prediction["class"],

            "damage_ml_confidence":
                ml_confidence,

            "damage_ml_confidence_status":
                confidence_status,

            "damage_ml_type":
                ml_prediction["damage_type"],

            "damage_ml_is_damage":
                ml_is_damage,

            "damage_ml_consistency":
                consistency
        }
    }

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / "phase3_evidence_ml.json"
    )

    save_json(
        evidence,
        output_path
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("==========================================")
    print("       PHASE 3 PIPELINE COMPLETE")
    print("==========================================")
    print()

    print("Unified output:")
    print(output_path)

    print()
    print("Summary:")

    print(
        "Claim document :",
        claim_type
    )

    print(
        "Invoice        :",
        invoice_type
    )

    print(
        "Invoice valid  :",
        invoice_extraction[
            "validation"
        ]["total_matches"]
    )

    print(
        "Signature      :",
        signature_result["result"]
    )

    print(
        "Traditional damage:",
        damage_result["result"]
    )

    print(
        "ML damage class:",
        ml_prediction["class"]
    )

    print(
        "ML confidence  :",
        ml_confidence
    )

    print(
        "ML confidence status:",
        confidence_status
    )

    print(
        "ML consistency:",
        consistency
    )

    print()
    print("==========================================")
    print()


if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print()
        print(
            "PHASE 3 PIPELINE ERROR:",
            e
        )
        print()

        raise SystemExit(1)
