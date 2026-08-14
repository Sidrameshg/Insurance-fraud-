from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

output = Path("phase3/ocr/test_document.png")

width, height = 1200, 700
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

font_paths = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf"
]

font = None

for font_path in font_paths:
    if Path(font_path).exists():
        font = ImageFont.truetype(font_path, 32)
        break

if font is None:
    font = ImageFont.load_default()

title_font = ImageFont.truetype(
    "C:/Windows/Fonts/arialbd.ttf", 42
) if Path("C:/Windows/Fonts/arialbd.ttf").exists() else font

draw.text((70, 50), "INSURANCE CLAIM DOCUMENT", fill="black", font=title_font)

lines = [
    "Policy Number: POL123456",
    "Claim Number: CLM001",
    "Vehicle: Honda City",
    "Accident Date: 12/08/2026",
    "Repair Amount: 85000",
    "Repair Shop: ABC Motors",
    "Claimant: Raj Kumar"
]

y = 140

for line in lines:
    draw.text((80, y), line, fill="black", font=font)
    y += 70

image.save(output)

print(f"Test document created: {output}")
