from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_features_final.json"
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
    print("     PHASE 3 FINAL FEATURE VALIDATION")
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

    expected_features = {

        # ==============================================
        # BASE FEATURES
        # ==============================================

        "document_type_insurance_claim": 1,
        "document_confidence": 0.812,
        "document_score_margin": 10.0,

        "invoice_document_type": "invoice",
        "invoice_document_confidence": 0.929,

        "invoice_total": 34000.0,
        "invoice_items_total": 34000.0,

        "invoice_total_valid": 1,

        "claim_repair_amount": 85000.0,

        "claim_invoice_amount_difference": 51000.0,

        "invoice_to_claim_amount_ratio": 0.4,

        "signature_similarity": 1.0,
        "signature_match": 1,

        "damage_ratio": 0.0355,
        "damage_detected": 1,

        "damage_mean_pixel_difference": 3.3271,

        # ==============================================
        # ML DAMAGE FEATURES
        # ==============================================

        "damage_ml_class": "F_Normal",
        "damage_ml_confidence": 0.4393,

        "damage_ml_front": 1,
        "damage_ml_rear": 0,

        "damage_ml_is_damage": 0,
        "damage_ml_is_breakage": 0,
        "damage_ml_is_crushed": 0,
        "damage_ml_is_normal": 1
    }

    print(
        "Expected feature count:",
        len(expected_features)
    )

    print(
        "Actual feature count  :",
        len(features)
    )

    print()

    results = []

    # ==============================================
    # CHECK FEATURE COUNT
    # ==============================================

    count_passed = (
        len(features) == len(expected_features)
    )

    print(
        f"{'feature_count':<42}"
        f"{'PASS' if count_passed else 'FAIL'}"
        f"    actual={len(features)}"
        f" expected={len(expected_features)}"
    )

    results.append(count_passed)

    # ==============================================
    # CHECK EACH FEATURE
    # ==============================================

    for name, expected in expected_features.items():

        if name not in features:

            print(
                f"{name:<42}"
                f"FAIL    actual=MISSING"
                f" expected={expected}"
            )

            results.append(False)

            continue

        results.append(
            check(
                name,
                features[name],
                expected
            )
        )

    passed = sum(results)
    total = len(results)
    failed = total - passed

    print()
    print("------------------------------------------")
    print(f"Total checks           : {total}")
    print(f"Passed                 : {passed}")
    print(f"Failed                 : {failed}")
    print("------------------------------------------")
    print()

    report = {

        "phase": "phase3",

        "test":
            "final_feature_validation",

        "feature_file":
            str(FEATURE_FILE),

        "expected_feature_count":
            len(expected_features),

        "actual_feature_count":
            len(features),

        "total_checks":
            total,

        "passed":
            passed,

        "failed":
            failed,

        "pass_rate":
            round(
                passed / total * 100,
                2
            ),

        "status":
            "PASS"
            if failed == 0
            else
            "FAIL"
    }

    output = (
        ROOT
        / "phase3"
        / "validation"
        / "phase3_final_feature_validation_report.json"
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
