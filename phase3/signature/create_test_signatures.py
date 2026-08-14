from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

output_dir = Path("phase3/signature")
output_dir.mkdir(parents=True, exist_ok=True)

width = 1000
height = 400

try:
    font = ImageFont.truetype(
        "C:/Windows/Fonts/segoesc.ttf",
        90
    )
except:
    font = ImageFont.truetype(
        "C:/Windows/Fonts/ariali.ttf",
        90
    )

# Reference signature
reference = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(reference)

draw.text(
    (100, 120),
    "Raj Kumar",
    fill="black",
    font=font
)

reference.save(
    output_dir / "signature_reference.png"
)

# Claim signature - intentionally similar
claim = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(claim)

draw.text(
    (108, 124),
    "Raj Kumar",
    fill="black",
    font=font
)

claim.save(
    output_dir / "signature_claim.png"
)

print("Signature test images created:")
print(output_dir / "signature_reference.png")
print(output_dir / "signature_claim.png")
