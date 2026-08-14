from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Insurance-Fraud-AI",
        "phase": "phase5"
    }
