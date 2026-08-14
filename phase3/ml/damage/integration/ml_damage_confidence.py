def evaluate_ml_confidence(confidence):

    confidence = float(confidence)

    if confidence >= 0.70:
        return "HIGH_CONFIDENCE"

    if confidence >= 0.50:
        return "MEDIUM_CONFIDENCE"

    return "LOW_CONFIDENCE"


def evaluate_damage_consistency(
    traditional_damage_detected,
    ml_is_damage,
    ml_confidence
):

    confidence_status = evaluate_ml_confidence(
        ml_confidence
    )

    traditional = bool(
        traditional_damage_detected
    )

    ml_damage = bool(
        ml_is_damage
    )

    if confidence_status == "LOW_CONFIDENCE":

        return "ML_UNCERTAIN"

    if traditional == ml_damage:

        return "AGREEMENT"

    return "DISAGREEMENT"
