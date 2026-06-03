import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.schemas import GuardrailsRequest, GuardrailsResponse, HealthResponse
from app.rails import check_input, check_output

app = FastAPI(
    title="Guardrails Service",
    description="Input and output safety validation for the property triage pipeline.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="guardrails_service")


@app.post("/check/input", response_model=GuardrailsResponse)
async def validate_input(body: GuardrailsRequest):
    """
    Validates that the submitted text is a genuine property listing.
    Rejects spam, off-topic content, and offensive submissions.
    """
    try:
        result = await check_input(body.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content={
        "pass": result["pass"],
        "reason": result["reason"],
        "safe_text": result["safe_text"],
    })


@app.post("/check/output", response_model=GuardrailsResponse)
async def validate_output(body: GuardrailsRequest):
    """
    Validates that the AI-generated listing report contains no fabricated facts,
    price guarantees, or false legal claims.
    """
    try:
        result = await check_output(body.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content={
        "pass": result["pass"],
        "reason": result["reason"],
        "safe_text": result["safe_text"],
    })
