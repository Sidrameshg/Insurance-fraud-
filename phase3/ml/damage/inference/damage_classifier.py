import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = Path(
    "./phase3/ml/damage/models/"
    "mobilenetv2_damage_finetuned_best.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal",
]


def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = models.mobilenet_v2(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def get_transform():

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def classify_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_model()

    transform = get_transform()

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    predicted_class = (
        CLASS_NAMES[predicted_index]
    )

    confidence_value = confidence.item()

    probability_values = (
        probabilities[0].cpu().tolist()
    )

    result = {

        "model": "MobileNetV2",

        "model_type":
            "fine_tuned",

        "image":
            str(image_path),

        "predicted_class":
            predicted_class,

        "confidence":
            round(
                confidence_value,
                6
            ),

        "front_or_rear":
            (
                "front"
                if predicted_class.startswith("F_")
                else "rear"
            ),

        "damage_type":
            (
                "breakage"
                if "Breakage" in predicted_class
                else
                "crushed"
                if "Crushed" in predicted_class
                else
                "normal"
            ),

        "is_damage":
            predicted_class not in [
                "F_Normal",
                "R_Normal"
            ],

        "is_breakage":
            "Breakage" in predicted_class,

        "is_crushed":
            "Crushed" in predicted_class,

        "is_normal":
            "Normal" in predicted_class,

        "probabilities": {
            CLASS_NAMES[i]:
                round(
                    probability_values[i],
                    6
                )
            for i in range(
                len(CLASS_NAMES)
            )
        }
    }

    return result


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "damage_classifier.py "
            "<image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = classify_image(
            image_path
        )

        print()
        print(
            "===== ML DAMAGE CLASSIFICATION ====="
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        output_path = (
            Path(image_path).with_suffix(
                ".damage_ml.json"
            )
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        print()
        print(
            "Output file:"
        )

        print(output_path)

        print()
        print(
            "====================================="
        )

    except Exception as error:

        print(
            "ML DAMAGE CLASSIFICATION ERROR:",
            error
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
