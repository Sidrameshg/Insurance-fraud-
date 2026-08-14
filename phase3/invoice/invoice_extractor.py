from pathlib import Path
import json
import re


def load_ocr_json(ocr_path: str):
    path = Path(ocr_path)

    if not path.exists():
        raise FileNotFoundError(f"OCR file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_value_after_label(text, label):
    pattern = rf"{re.escape(label)}\s*:\s*(.+?)(?=\s+[A-Z][A-Za-z ]*:\s|$)"

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def extract_invoice(ocr_data):

    text = ocr_data.get("text", "")

    result = {
        "document_type": "invoice",
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "vehicle": None,
        "policy_number": None,
        "items": [],
        "total_amount": None,
        "validation": {}
    }

    # -----------------------------
    # Vendor
    # -----------------------------

    match = re.search(
        r"Repair Shop:\s*(.+?)(?=\s+Invoice Number:)",
        text,
        re.IGNORECASE
    )

    if match:
        result["vendor"] = match.group(1).strip()

    # -----------------------------
    # Invoice Number
    # -----------------------------

    match = re.search(
        r"Invoice Number:\s*([A-Za-z0-9-]+)",
        text,
        re.IGNORECASE
    )

    if match:
        result["invoice_number"] = match.group(1)

    # -----------------------------
    # Invoice Date
    # -----------------------------

    match = re.search(
        r"Invoice Date:\s*(\d{2}/\d{2}/\d{4})",
        text,
        re.IGNORECASE
    )

    if match:
        result["invoice_date"] = match.group(1)

    # -----------------------------
    # Vehicle
    # -----------------------------

    match = re.search(
        r"Vehicle:\s*(.+?)(?=\s+Policy Number:)",
        text,
        re.IGNORECASE
    )

    if match:
        result["vehicle"] = match.group(1).strip()

    # -----------------------------
    # Policy Number
    # -----------------------------

    match = re.search(
        r"Policy Number:\s*([A-Za-z0-9-]+)",
        text,
        re.IGNORECASE
    )

    if match:
        result["policy_number"] = match.group(1)

    # -----------------------------
    # Invoice Items
    # -----------------------------

    item_patterns = [
        (r"Front Bumper\s+(\d+(?:\.\d+)?)", "Front Bumper"),
        (r"Headlight\s+(\d+(?:\.\d+)?)", "Headlight"),
        (r"Labour\s+(\d+(?:\.\d+)?)", "Labour")
    ]

    for pattern, description in item_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            amount = float(match.group(1))

            result["items"].append({
                "description": description,
                "amount": amount
            })

    # -----------------------------
    # Total Amount
    # -----------------------------

    match = re.search(
        r"Total Amount\s+(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        result["total_amount"] = float(match.group(1))

    # -----------------------------
    # Validation
    # -----------------------------

    calculated_total = sum(
        item["amount"]
        for item in result["items"]
    )

    invoice_total = result["total_amount"]

    if invoice_total is not None:

        difference = round(
            calculated_total - invoice_total,
            2
        )

        result["validation"] = {
            "calculated_items_total": calculated_total,
            "invoice_total": invoice_total,
            "difference": difference,
            "total_matches": difference == 0
        }

    return result


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage: python phase3/invoice/invoice_extractor.py "
            "<ocr_json_path>"
        )

        raise SystemExit(1)

    ocr_path = sys.argv[1]

    try:

        ocr_data = load_ocr_json(ocr_path)

        invoice = extract_invoice(ocr_data)

        output_path = Path(ocr_path).with_name(
            Path(ocr_path).stem + ".invoice.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                invoice,
                file,
                indent=2,
                ensure_ascii=False
            )

        print("\n===== INVOICE EXTRACTION RESULT =====\n")

        print(
            json.dumps(
                invoice,
                indent=2,
                ensure_ascii=False
            )
        )

        print("\nOutput file:")
        print(output_path)

        print("\n=====================================\n")

    except Exception as e:

        print(f"INVOICE EXTRACTION ERROR: {e}")
        raise SystemExit(1)
