from pathlib import Path
import os


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]


# =================================================
# LLM CONFIGURATION
# =================================================

LLM_PROVIDER = os.getenv(
    "PHASE4_LLM_PROVIDER",
    "mock"
)

LLM_MODEL = os.getenv(
    "PHASE4_LLM_MODEL",
    "mock-model"
)

LLM_TEMPERATURE = float(
    os.getenv(
        "PHASE4_LLM_TEMPERATURE",
        "0.1"
    )
)


def get_llm_config():

    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE
    }


if __name__ == "__main__":

    print()
    print("==========================================")
    print("       PHASE 4 LLM CONFIGURATION")
    print("==========================================")
    print()

    config = get_llm_config()

    print(
        "Provider:",
        config["provider"]
    )

    print(
        "Model:",
        config["model"]
    )

    print(
        "Temperature:",
        config["temperature"]
    )

    print()
    print("LLM configuration ready.")
    print()
