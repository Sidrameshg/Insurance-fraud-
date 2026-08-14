import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 6

CLASS_NAMES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal",
]

DEVICE = torch.device("cpu")

print("===== BUILDING MOBILENETV2 =====")
print("Device:", DEVICE)
print("Classes:", NUM_CLASSES)

# Load pretrained MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace final classifier
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print()
print("===== MODEL CONFIGURATION =====")
print("Architecture: MobileNetV2")
print("Input features:", input_features)
print("Output classes:", NUM_CLASSES)

print()
print("===== TRAINABLE PARAMETERS =====")

trainable = 0
total = 0

for parameter in model.parameters():

    total += parameter.numel()

    if parameter.requires_grad:
        trainable += parameter.numel()

print("Total parameters:", total)
print("Trainable parameters:", trainable)
print("Frozen parameters:", total - trainable)

print()
print("===== CLASSIFIER =====")
print(model.classifier)

print()
print("===== MODEL READY =====")

# Save initial model structure
output_path = "./phase3/ml/damage/models/mobilenetv2_initial.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "architecture": "MobileNetV2",
    },
    output_path
)

print()
print("Initial model saved:")
print(output_path)
