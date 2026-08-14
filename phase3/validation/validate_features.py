from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features.json"
)


def check(name, actual, expected):

    if isinstance(expected, float):
        passed = math.isclose(
            float(actual),
            expected,
            rel_tol=1e-6,
            abs_tol=1e-6
        )
    else:
        passed = actual == expected

    print(
        f"{name:<42}"
        f"{'PASS' if passed else 'FAIL'}"
        f"    actual={actual}"
        f" expected={expected}"
    )

    return passed


def main():

    print()
    print("==========================================")
    print("     PHASE 3 FEATURE VALIDATION")
    print("==========================================")
    print()

    if not FEATURE_FILE.exists():
        print("Feature file not found:")
        print(FEATURE_FILE)
        sys.exit(1)

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    features = data["features"]

    tests = [
        (
            "document_type_insurance_claim",
            features["document_type_insurance_claim"],
            1
        ),
        (
            "document_confidence",
            features["document_confidence"],
            0.812
        ),
        (
            "document_score_margin",
            features["document_score_margin"],
            10.0
        ),
        (
            "invoice_document_type",
            features["invoice_document_type"],
            "invoice"
        ),
        (
            "invoice_document_confidence",
            features["invoice_document_confidence"],
            0.929
        ),
        (
            "invoice_total",
            features["invoice_total"],
            34000.0
        ),
        (
            "invoice_items_total",
            features["invoice_items_total"],
            34000.0
        ),
        (
            "invoice_total_valid",
            features["invoice_total_valid"],
            1
        ),
        (
            "claim_repair_amount",
            features["claim_repair_amount"],
            85000.0
        ),
        (
            "claim_invoice_amount_difference",
            features["claim_invoice_amount_difference"],
            51000.0
        ),
        (
            "invoice_to_claim_amount_ratio",
            features["invoice_to_claim_amount_ratio"],
            0.4
        ),
        (
            "signature_similarity",
            features["signature_similarity"],
            1.0
        ),
        (
            "signature_match",
            features["signature_match"],
            1
        ),
        (
            "damage_ratio",
            features["damage_ratio"],
            0.0355
        ),
        (
            "damage_detected",
            features["damage_detected"],
            1
        ),
        (
            "damage_mean_pixel_difference",
            features["damage_mean_pixel_difference"],
            3.3271
        )
    ]

    results = []

    for name, actual, expected in tests:

        results.append(
            check(
                name,
                actual,
                expected
            )
        )

    passed = sum(results)
    total = len(results)
    failed = total - passed

    print()
    print("------------------------------------------")
    print(f"Total features checked : {total}")
    print(f"Passed                 : {passed}")
    print(f"Failed                 : {failed}")
    print("------------------------------------------")
    print()

    report = {
        "phase": "phase3",
        "test": "evidence_feature_validation",
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(
            passed / total * 100,
            2
        )
    }

    output = (
        ROOT
        / "phase3"
        / "validation"
        / "phase3_feature_validation_report.json"
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

    print("Report:")
    print(output)
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
