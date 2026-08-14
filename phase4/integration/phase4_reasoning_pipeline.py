from pathlib import Path
import sys
import json


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =================================================
# PHASE 4
# =================================================

from phase4.llm.reasoning.reasoning_engine import (
    InvestigationReasoningEngine
)


# =================================================
# INPUT / OUTPUT
# =================================================

INPUT_FILE = (
    ROOT
    / "phase3"
    / "integration"
    / "phase3_risk_fusion_dynamic.json"
)

OUTPUT_DIR = (
    ROOT
    / "phase4"
    / "integration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "phase4_reasoning_result.json"
)


# =================================================
# LOAD PHASE 3 RESULT
# =================================================

def load_phase3_result():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Phase 3 result not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =================================================
# BUILD REASONING INPUT
# =================================================

def build_reasoning_input(
    phase3_result
):

    return {

        "phase":
            "phase3",

        "fraud_model":
            phase3_result.get(
                "fraud_model",
                {}
            ),

        "evidence":
            phase3_result.get(
                "evidence",
                {}
            ),

        "final_risk":
            phase3_result.get(
                "final_risk",
                {}
            )
    }


# =================================================
# MAIN
# =================================================

def main():

    print()
    print("==========================================")
    print("       PHASE 4 REASONING PIPELINE")
    print("==========================================")

    # ---------------------------------------------
    # Load Phase 3
    # ---------------------------------------------

    print()
    print(
        "[1/3] Loading Phase 3 result..."
    )

    phase3_result = (
        load_phase3_result()
    )

    print(
        "      Phase 3 loaded successfully."
    )

    # ---------------------------------------------
    # Build reasoning input
    # ---------------------------------------------

    print()
    print(
        "[2/3] Building reasoning input..."
    )

    reasoning_input = (
        build_reasoning_input(
            phase3_result
        )
    )

    print(
        "      Reasoning input prepared."
    )

    # ---------------------------------------------
    # Run reasoning
    # ---------------------------------------------

    print()
    print(
        "[3/3] Running Phase 4 reasoning..."
    )

    engine = (
        InvestigationReasoningEngine()
    )

    reasoning_result = (
        engine.analyze(
            reasoning_input
        )
    )

    # ---------------------------------------------
    # Final result
    # ---------------------------------------------

    final_result = {

        "phase":
            "phase4",

        "stage":
            "llm_reasoning",

        "source":
            str(INPUT_FILE),

        "reasoning":
            reasoning_result
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("==========================================")
    print("       PHASE 4 REASONING COMPLETE")
    print("==========================================")

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
