from pathlib import Path
import json

from fastapi import APIRouter, HTTPException

from phase5.api.schemas import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse
)

router = APIRouter(
    prefix="/claims",
    tags=["Claims"]
)


PHASE4_REPORT = (
    Path(__file__).resolve().parents[3]
    / "phase4"
    / "final_integration"
    / "output"
    / "phase4_final_investigation_report.json"
)


@router.post(
    "/analyze",
    response_model=ClaimAnalysisResponse
)
def analyze_claim(request: ClaimAnalysisRequest):

    if not request.use_phase4:
        return ClaimAnalysisResponse(
            claim_id=request.claim_id,
            fraud_probability=None,
            fraud_prediction=None,
            risk_level=None,
            evidence_score=None,
            phase4_enabled=False,
            status="PHASE4_DISABLED"
        )

    if not PHASE4_REPORT.exists():
        raise HTTPException(
            status_code=500,
            detail="Phase 4 investigation report not found"
        )

    try:
        with open(PHASE4_REPORT, "r", encoding="utf-8-sig") as file:
            report = json.load(file)

        claim_summary = report.get("claim_summary", {})

        return ClaimAnalysisResponse(
            claim_id=request.claim_id,
            fraud_probability=claim_summary.get("fraud_probability"),
            fraud_prediction=claim_summary.get("fraud_prediction"),
            risk_level=claim_summary.get("risk_level"),
            evidence_score=claim_summary.get("evidence_score"),
            phase4_enabled=True,
            status="ANALYSIS_COMPLETE"
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid Phase 4 investigation report JSON"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Phase 4 investigation report: {exc}"
        )
