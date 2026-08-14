from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

output = Path("phase3/invoice/test_invoice.png")

width, height = 1200, 850

image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

regular_path = "C:/Windows/Fonts/arial.ttf"
bold_path = "C:/Windows/Fonts/arialbd.ttf"

font = ImageFont.truetype(regular_path, 30)
bold = ImageFont.truetype(bold_path, 42)

draw.text(
    (70, 50),
    "VEHICLE REPAIR INVOICE",
    fill="black",
    font=bold
)

lines = [
    "Repair Shop: ABC Motors",
    "Invoice Number: INV-2026-001",
    "Invoice Date: 12/08/2026",
    "Vehicle: Honda City",
    "Policy Number: POL123456",
    "",
    "Front Bumper        15000",
    "Headlight           12000",
    "Labour               7000",
    "",
    "Total Amount        34000"
]

y = 140

for line in lines:
    draw.text(
        (80, y),
        line,
        fill="black",
        font=font
    )
    y += 60

image.save(output)

print(f"Invoice test document created: {output}")
