import json
from pathlib import Path

from investigation_agent import build_investigation


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "agent" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_grounded_prompt(investigation):

    claim_evidence = investigation["claim_evidence"]
    investigation_data = investigation["investigation"]
    knowledge = investigation["retrieved_knowledge"]

    knowledge_text = "\n\n".join(
        [
            (
                f"SOURCE: {item['source']}\n"
                f"CHUNK: {item['chunk_id']}\n"
                f"SIMILARITY: {item['similarity']:.4f}\n"
                f"TEXT:\n{item['text']}"
            )
            for item in knowledge
        ]
    )

    prompt = f"""
You are an insurance claim investigation assistant.

Your task is to analyze the supplied claim evidence and
provide a grounded investigation report.

IMPORTANT RULES:

1. Do not invent information.
2. Treat the fraud model prediction as a risk signal,
   not definitive proof of fraud.
3. Clearly identify contradictions.
4. Use the retrieved knowledge only as investigation guidance.
5. Separate observed evidence from interpretation.
6. Recommend investigation actions based only on the
   supplied evidence and retrieved guidance.

CLAIM EVIDENCE:

Fraud probability:
{claim_evidence["fraud_probability"]}

Fraud prediction:
{claim_evidence["fraud_prediction"]}

Risk level:
{claim_evidence["risk_level"]}

Evidence score:
{claim_evidence["evidence_score"]}

Contradictions:
{json.dumps(claim_evidence["contradictions"], indent=2)}

IMPORTANT EVIDENCE:

{json.dumps(investigation_data["important_evidence"], indent=2)}

RETRIEVED INVESTIGATION KNOWLEDGE:

{knowledge_text}

Provide the investigation response using these sections:

1. Claim Summary
2. Important Evidence
3. Contradictions
4. Fraud-Risk Interpretation
5. Recommended Investigation Actions
6. Uncertainty / Limitations
7. Knowledge Sources

Do not introduce facts that are absent from the supplied evidence.
"""

    return prompt


def run_mock_llm(prompt):

    return (
        "Phase 4 grounded LLM mock response.\n\n"
        "The investigation agent successfully prepared a "
        "grounded prompt using Phase 3 evidence and retrieved "
        "investigation knowledge.\n\n"
        "The real LLM provider can be connected to this interface "
        "without changing the evidence or RAG components."
    )


def main():

    print("=" * 60)
    print("       PHASE 4 AGENT + LLM")
    print("=" * 60)

    print()
    print("[1/4] Loading investigation agent...")

    investigation, _ = build_investigation()

    print("      Investigation data loaded.")

    print()
    print("[2/4] Building grounded LLM prompt...")

    prompt = build_grounded_prompt(investigation)

    print("      Grounded prompt prepared.")

    print()
    print("[3/4] Running LLM...")

    llm_response = run_mock_llm(prompt)

    print("      LLM response generated.")

    result = {
        "phase": "phase4",
        "component": "agent_llm",
        "provider": "mock",
        "model": "mock-model",

        "claim_evidence": investigation[
            "claim_evidence"
        ],

        "retrieved_knowledge": investigation[
            "retrieved_knowledge"
        ],

        "grounded_prompt": prompt,

        "llm_response": llm_response
    }

    output_file = (
        OUTPUT_DIR
        / "agent_llm_result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("[4/4] Saving result...")

    print()
    print("=" * 60)
    print("       PHASE 4 AGENT + LLM COMPLETE")
    print("=" * 60)

    print()
    print("Provider:", result["provider"])
    print("Model:", result["model"])

    print()
    print("Fraud probability:",
          result["claim_evidence"]["fraud_probability"])

    print("Fraud prediction:",
          result["claim_evidence"]["fraud_prediction"])

    print("Risk level:",
          result["claim_evidence"]["risk_level"])

    print(
        "Retrieved knowledge:",
        len(result["retrieved_knowledge"])
    )

    print()
    print("LLM Response:")
    print(result["llm_response"])

    print()
    print("Output:")
    print(output_file)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
