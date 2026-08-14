from pathlib import Path
import pytesseract
from PIL import Image


def extract_text(image_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        image = Image.open(path)
    except Exception as e:
        raise ValueError(f"Unable to open image: {e}")

    text = pytesseract.image_to_string(image)

    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python phase3/ocr/ocr_engine.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        text = extract_text(image_path)

        print("\n===== OCR RESULT =====\n")
        print(text)
        print("\n======================\n")

    except Exception as e:
        print(f"OCR ERROR: {e}")
        sys.exit(1)
