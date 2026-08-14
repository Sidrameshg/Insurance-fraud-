from pathlib import Path
import sys
import json


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =================================================
# PHASE 4 MODULES
# =================================================

from phase4.llm.client.llm_client import (
    LLMClient
)

from phase4.llm.prompts.investigation_prompt import (
    SYSTEM_PROMPT,
    build_investigation_prompt
)


# =================================================
# REASONING ENGINE
# =================================================

class InvestigationReasoningEngine:

    def __init__(self):

        self.llm = LLMClient()

    def analyze(
        self,
        evidence
    ):

        user_prompt = (
            build_investigation_prompt(
                evidence
            )
        )

        response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        return {

            "engine":
                "Phase4 Investigation Reasoning",

            "llm":
                response,

            "input_evidence":
                evidence
        }


# =================================================
# TEST
# =================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("       PHASE 4 REASONING ENGINE")
    print("==========================================")
    print()

    sample_evidence = {

        "fraud_probability":
            0.72,

        "evidence_score":
            65,

        "risk_level":
            "HIGH_RISK"
    }

    engine = (
        InvestigationReasoningEngine()
    )

    result = engine.analyze(
        sample_evidence
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print()
    print(
        "Reasoning engine test complete."
    )

    print()
    print("==========================================")
