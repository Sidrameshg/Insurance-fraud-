from pathlib import Path
import json

import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter


def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_ocr(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    processed_image = preprocess_image(image)

    data = pytesseract.image_to_data(
        processed_image,
        output_type=Output.DICT
    )

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not text or confidence < 0:
            continue

        words.append({
            "text": text,
            "confidence": confidence,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i])
        })

    full_text = " ".join(word["text"] for word in words)

    result = {
        "document": {
            "type": "insurance_claim",
            "source": str(path)
        },
        "text": full_text,
        "word_count": len(words),
        "words": words
    }

    return result


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python phase3/ocr/ocr_v3.py <image_path>"
        )
        raise SystemExit(1)

    image_path = sys.argv[1]

    try:

        result = extract_ocr(image_path)

        output_path = Path(image_path).with_suffix(".ocr.json")

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False
            )

        print("\n===== OCR V3 RESULT =====\n")

        print("Document type :", result["document"]["type"])
        print("Word count    :", result["word_count"])
        print("Output file   :", output_path)

        print("\nExtracted text:\n")
        print(result["text"])

        print("\n=========================\n")

    except Exception as e:

        print(f"OCR ERROR: {e}")
        raise SystemExit(1)
