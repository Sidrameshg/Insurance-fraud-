from pathlib import Path
from PIL import Image, ImageDraw

output_dir = Path("phase3/damage")
output_dir.mkdir(parents=True, exist_ok=True)

width = 1000
height = 600

# -----------------------------
# Reference / undamaged image
# -----------------------------

reference = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(reference)

# Vehicle body/panel
draw.rectangle(
    (150, 150, 850, 450),
    fill=(180, 180, 180),
    outline=(50, 50, 50),
    width=5
)

# Headlight
draw.rectangle(
    (220, 220, 350, 290),
    fill=(230, 230, 230),
    outline=(50, 50, 50),
    width=3
)

reference_path = output_dir / "vehicle_reference.png"

reference.save(reference_path)

# -----------------------------
# Damaged image
# -----------------------------

damaged = reference.copy()

draw = ImageDraw.Draw(damaged)

# Simulated dent
draw.ellipse(
    (480, 250, 650, 380),
    fill=(90, 90, 90)
)

# Simulated scratch
draw.line(
    (680, 200, 790, 390),
    fill=(20, 20, 20),
    width=12
)

damaged_path = output_dir / "vehicle_damaged.png"

damaged.save(damaged_path)

print("Damage test images created:")
print(reference_path)
print(damaged_path)
