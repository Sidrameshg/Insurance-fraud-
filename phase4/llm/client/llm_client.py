from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from phase4.llm.client.llm_config import (
    get_llm_config
)


class LLMClient:

    def __init__(self):

        self.config = (
            get_llm_config()
        )

    def generate(
        self,
        system_prompt,
        user_prompt
    ):

        provider = (
            self.config[
                "provider"
            ]
        )

        if provider == "mock":

            return self._mock_response(
                system_prompt,
                user_prompt
            )

        raise NotImplementedError(
            f"LLM provider '{provider}' "
            "is not implemented yet."
        )

    def _mock_response(
        self,
        system_prompt,
        user_prompt
    ):

        return {
            "provider": "mock",
            "model": self.config["model"],
            "response": (
                "Phase 4 LLM mock response. "
                "The LLM interface is working "
                "and ready for a real reasoning model."
            )
        }


if __name__ == "__main__":

    print()
    print("==========================================")
    print("          PHASE 4 LLM CLIENT")
    print("==========================================")
    print()

    client = LLMClient()

    result = client.generate(
        system_prompt=(
            "You are an insurance fraud "
            "investigation assistant."
        ),
        user_prompt=(
            "Analyze this claim."
        )
    )

    print(
        "Provider:",
        result["provider"]
    )

    print(
        "Model:",
        result["model"]
    )

    print(
        "Response:",
        result["response"]
    )

    print()
    print("LLM client test complete.")
    print()
