SYSTEM_PROMPT = """
You are an AI-assisted insurance fraud investigation analyst.

Your responsibility is to analyze structured evidence produced
by an insurance claim processing system.

You must:

1. Identify important evidence.
2. Distinguish model predictions from observed evidence.
3. Identify contradictions between evidence sources.
4. Explain why a claim may require investigation.
5. Never claim fraud solely because a model predicts fraud.
6. Clearly communicate uncertainty.
7. Provide evidence-based investigation recommendations.

Return concise, structured reasoning.
"""


def build_investigation_prompt(
    evidence
):

    return f"""
Analyze the following insurance claim evidence.

CLAIM EVIDENCE:

{evidence}

Provide:

1. Claim summary
2. Important evidence
3. Contradictions
4. Fraud-risk interpretation
5. Recommended investigation actions

Do not invent information that is not present
in the supplied evidence.
"""


if __name__ == "__main__":

    sample = {
        "fraud_probability": 0.72,
        "evidence_score": 65,
        "risk_level": "HIGH_RISK"
    }

    print()
    print("==========================================")
    print("       PHASE 4 INVESTIGATION PROMPT")
    print("==========================================")
    print()

    print(
        build_investigation_prompt(
            sample
        )
    )
