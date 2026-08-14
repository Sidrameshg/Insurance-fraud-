# Phase 3 Damage ML Dataset

## Task

Binary vehicle damage classification.

Classes:

- damage
- no_damage

## Dataset split

train/
val/
test/

## Important

The existing Phase 3 baseline images are NOT part of the training dataset.

They remain independent validation examples for comparison against the original pixel-difference damage detector.

## Goal

Compare:

1. Existing pixel-difference baseline
2. Transfer-learning CNN

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

No model should be considered better based on accuracy alone.
