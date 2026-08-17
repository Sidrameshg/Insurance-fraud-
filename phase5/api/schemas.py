from pydantic import BaseModel, Field
from typing import Optional


class ClaimAnalysisRequest(BaseModel):

    claim_id: str = Field(
        ...,
        description="Unique claim identifier mapped to PolicyNumber"
    )


class ClaimAnalysisResponse(BaseModel):

    claim_id: str

    claim_exists: bool

    fraud_probability: Optional[float] = None

    fraud_prediction: Optional[str] = None

    fraud_threshold: Optional[float] = None

    evidence_score: Optional[float] = None

    evidence_reasons: list[str] = []

    risk_level: Optional[str] = None

    uncertainty_level: Optional[str] = None

    human_review_required: Optional[bool] = None

    human_review_reason: Optional[str] = None

    phase3_fusion_available: bool = False

    status: str