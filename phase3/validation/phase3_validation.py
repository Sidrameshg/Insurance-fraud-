from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.ocr.ocr_v3 import extract_ocr
from phase3.invoice.invoice_extractor import extract_invoice
from phase3.document.document_classifier import classify_document
from phase3.signature.signature_verifier import verify_signature
from phase3.damage.damage_detector import detect_damage


def test_ocr():
    image = ROOT / "phase3" / "ocr" / "test_document.png"
    result = extract_ocr(str(image))
    text = result.get("text", "")

    return (
        "POL123456" in text
        and "CLM001" in text
        and "Honda City" in text
    )


def test_invoice():
    image = ROOT / "phase3" / "invoice" / "test_invoice.png"

    ocr = extract_ocr(str(image))
    result = extract_invoice(ocr)

    return (
        result["vendor"] == "ABC Motors"
        and result["invoice_number"] == "INV-2026-001"
        and result["total_amount"] == 34000.0
        and result["validation"]["total_matches"] is True
    )


def test_document():
    image = ROOT / "phase3" / "ocr" / "test_document.png"

    ocr = extract_ocr(str(image))
    result = classify_document(ocr)

    return result["document_type"] == "insurance_claim"


def test_signature_match():
    reference = ROOT / "phase3" / "signature" / "signature_reference.png"
    claim = ROOT / "phase3" / "signature" / "signature_claim.png"

    result = verify_signature(
        str(reference),
        str(claim)
    )

    return (
        result["result"] == "MATCH"
        and result["similarity_score"] >= result["threshold"]
    )


def test_signature_mismatch():
    reference = ROOT / "phase3" / "signature" / "signature_reference.png"
    different = ROOT / "phase3" / "signature" / "signature_different.png"

    result = verify_signature(
        str(reference),
        str(different)
    )

    return (
        result["result"] == "MISMATCH"
        and result["similarity_score"] < result["threshold"]
    )


def test_damage():
    reference = ROOT / "phase3" / "damage" / "vehicle_reference.png"
    damaged = ROOT / "phase3" / "damage" / "vehicle_damaged.png"

    result = detect_damage(
        str(reference),
        str(damaged)
    )

    return (
        result["result"] == "DAMAGE_DETECTED"
        and result["damage_ratio"] > 0
    )


def test_no_damage():
    reference = ROOT / "phase3" / "damage" / "vehicle_reference.png"

    result = detect_damage(
        str(reference),
        str(reference)
    )

    return (
        result["result"] == "NO_SIGNIFICANT_DAMAGE"
        and result["damage_ratio"] == 0.0
    )


def test_integration():
    path = (
        ROOT
        / "phase3"
        / "integration"
        / "phase3_evidence.json"
    )

    if not path.exists():
        return False

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    summary = data.get("summary", {})

    return (
        summary.get("claim_document_type") == "insurance_claim"
        and summary.get("invoice_document_type") == "invoice"
        and summary.get("invoice_total_valid") is True
        and summary.get("signature_result") == "MATCH"
        and summary.get("damage_result") == "DAMAGE_DETECTED"
    )


def main():

    tests = [
        ("OCR extraction", test_ocr),
        ("Invoice extraction", test_invoice),
        ("Document understanding", test_document),
        ("Signature MATCH", test_signature_match),
        ("Signature MISMATCH", test_signature_mismatch),
        ("Damage DETECTED", test_damage),
        ("Damage NOT DETECTED", test_no_damage),
        ("Integrated evidence", test_integration),
    ]

    results = []

    print()
    print("==========================================")
    print("        PHASE 3 VALIDATION SUITE")
    print("==========================================")
    print()

    for name, function in tests:

        try:
            passed = function()
            status = "PASS" if passed else "FAIL"

        except Exception as e:
            passed = False
            status = "FAIL"
            print(f"  Error: {e}")

        print(f"{name:<32} {status}")

        results.append({
            "test": name,
            "status": status
        })

    total = len(results)

    passed_count = sum(
        r["status"] == "PASS"
        for r in results
    )

    failed_count = total - passed_count

    pass_rate = (
        passed_count / total * 100
        if total
        else 0
    )

    report = {
        "phase": "phase3",
        "total_tests": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": round(pass_rate, 2),
        "tests": results
    }

    output = (
        ROOT
        / "phase3"
        / "validation"
        / "phase3_validation_report.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("------------------------------------------")
    print(f"Total tests : {total}")
    print(f"Passed      : {passed_count}")
    print(f"Failed      : {failed_count}")
    print(f"Pass rate   : {pass_rate:.2f}%")
    print("------------------------------------------")
    print()
    print(f"Report: {output}")
    print()

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
