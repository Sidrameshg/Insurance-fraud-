from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

output = Path("phase3/signature/signature_different.png")

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

image = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(image)

draw.text(
    (250, 150),
    "A Sharma",
    fill="black",
    font=font
)

image.save(output)

print(f"Different signature created: {output}")
