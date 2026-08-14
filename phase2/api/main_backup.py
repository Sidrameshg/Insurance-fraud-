# ============================================================
# PHASE 2 — FASTAPI BACKEND
# Insurance Claim Fraud Detection
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .model_service import predict_claim


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Insurance Claim Fraud Detection API",
    description="AI-powered insurance claim fraud prediction service",
    version="1.0.0"
)


# ============================================================
# CLAIM INPUT MODEL
# ============================================================

class ClaimRequest(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )

    claim_data: dict


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Insurance Claim Fraud Detection API",
        "version": "1.0.0"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: ClaimRequest):

    try:

        result = predict_claim(
            request.claim_data
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )