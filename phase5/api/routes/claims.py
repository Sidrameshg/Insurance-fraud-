from fastapi import APIRouter, HTTPException

from phase5.api.schemas import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse
)

from phase5.services.claim_analysis import analyze_claim

from pathlib import Path
import json


router = APIRouter(
    prefix="/claims",
    tags=["Claims"]
)


ROOT = Path(__file__).resolve().parents[3]

PHASE4_REPORT = (
    ROOT
    / "phase4"
    / "final_integration"
    / "output"
    / "phase4_final_investigation_report.json"
)


@router.post(
    "/analyze",
    response_model=ClaimAnalysisResponse
)
def analyze_claim_route(
    request: ClaimAnalysisRequest
):

    try:

        result = analyze_claim(
            request.claim_id
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Claim '{request.claim_id}' "
                    "not found"
                )
            )

        return ClaimAnalysisResponse(

            claim_id=result["claim_id"],

            claim_exists=True,

            fraud_probability=result[
                "fraud_probability"
            ],

            fraud_prediction=result[
                "fraud_prediction"
            ],

            fraud_threshold=result[
                "fraud_threshold"
            ],

            evidence_score=result[
                "evidence_score"
            ],

            evidence_reasons=result[
                "evidence_reasons"
            ],

            risk_level=result[
                "risk_level"
            ],

            uncertainty_level=result[
                "uncertainty_level"
            ],

            human_review_required=result[
                "human_review_required"
            ],

            human_review_reason=result[
                "human_review_reason"
            ],

            phase3_fusion_available=result[
                "phase3_fusion_available"
            ],

            status=result[
                "status"
            ]
        )

    except HTTPException:

        raise

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Claim analysis failed: {exc}"
            )
        )


@router.get(
    "/{claim_id}/investigation"
)
def get_claim_investigation(
    claim_id: str
):

    if not PHASE4_REPORT.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Phase 4 investigation "
                "report not found"
            )
        )

    try:

        with open(
            PHASE4_REPORT,
            "r",
            encoding="utf-8-sig"
        ) as file:

            report = json.load(file)

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid Phase 4 "
                "investigation report JSON"
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load investigation "
                f"report: {exc}"
            )
        )

    return {

        "claim_id": claim_id,

        "phase": report.get(
            "phase"
        ),

        "component": report.get(
            "component"
        ),

        "status": report.get(
            "status"
        ),

        "claim_summary": report.get(
            "claim_summary",
            {}
        ),

        "important_evidence": report.get(
            "important_evidence",
            {}
        ),

        "contradictions": report.get(
            "contradictions",
            []
        ),

        "fraud_risk_interpretation": report.get(
            "fraud_risk_interpretation"
        ),

        "recommended_investigation_actions": report.get(
            "recommended_investigation_actions",
            []
        ),

        "retrieved_knowledge": report.get(
            "retrieved_knowledge",
            []
        ),

        "uncertainty": report.get(
            "uncertainty",
            {}
        ),

        "source_components": report.get(
            "source_components",
            {}
        )
    }