from pydantic import BaseModel, Field
from typing import Optional


class ClaimAnalysisRequest(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier")
    use_phase4: bool = Field(
        default=True,
        description="Whether to include Phase 4 investigation"
    )


class ClaimAnalysisResponse(BaseModel):
    claim_id: str
    fraud_probability: Optional[float] = None
    fraud_prediction: Optional[str] = None
    risk_level: Optional[str] = None
    evidence_score: Optional[float] = None
    phase4_enabled: bool
    status: str
