from fastapi import FastAPI

from phase5.api.routes.health import router as health_router
from phase5.api.routes.claims import router as claims_router


app = FastAPI(
    title="Insurance Fraud AI API",
    description="Production API for the Insurance-Fraud-AI system",
    version="1.0.0"
)


app.include_router(health_router)
app.include_router(claims_router)


@app.get("/")
def root():
    return {
        "service": "Insurance-Fraud-AI",
        "phase": "phase5",
        "status": "running"
    }
