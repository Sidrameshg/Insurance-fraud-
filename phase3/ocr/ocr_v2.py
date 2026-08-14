from pathlib import Path
import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter
import pandas as pd


def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_ocr_data(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    processed_image = preprocess_image(image)

    data = pytesseract.image_to_data(
        processed_image,
        output_type=Output.DICT
    )

    records = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if text and confidence >= 0:
            records.append({
                "text": text,
                "confidence": confidence,
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            })

    return records


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python phase3/ocr/ocr_v2.py <image_path>")
        raise SystemExit(1)

    image_path = sys.argv[1]

    try:
        records = extract_ocr_data(image_path)

        print("\n===== OCR V2 RESULT =====\n")

        if not records:
            print("No text detected.")
        else:
            df = pd.DataFrame(records)
            print(df.to_string(index=False))

        print("\n=========================\n")

    except Exception as e:
        print(f"OCR ERROR: {e}")
        raise SystemExit(1)
