import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException, UploadFile, File

from shared.schemas import ImageRequest, ImageResponse, ImageBatchRequest, ImageBatchResponse, ImageBatchResult, HealthResponse
from app.inference import predict, predict_from_bytes

_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title="Image Analyser Service",
    description="Classifies property images by room type and assigns a condition score using a fine-tuned ResNet-50.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="image_analyser")


@app.post("/analyse", response_model=ImageResponse)
async def analyse(body: ImageRequest):
    """
    Downloads the image at the given URL, runs it through the ResNet-50 classifier,
    and returns room_type, condition_score (1–5), and confidence.

    If confidence is below the threshold (default 0.5), room_type is 'uncertain'
    and condition_score is null.
    """
    try:
        result = predict(str(body.image_url))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return ImageResponse(**result)


@app.post("/analyse/batch", response_model=ImageBatchResponse)
async def analyse_batch(body: ImageBatchRequest):
    """
    Accepts up to 10 image URLs and returns classification results for all of them.
    Images are processed in parallel. Failed URLs return an error field instead of crashing the whole batch.
    """
    loop = asyncio.get_event_loop()

    async def _predict_one(url: str) -> ImageBatchResult:
        try:
            result = await loop.run_in_executor(_executor, predict, url)
            return ImageBatchResult(image_url=url, **result)
        except Exception as exc:
            return ImageBatchResult(image_url=url, error=str(exc))

    results = await asyncio.gather(*[_predict_one(url) for url in body.image_urls])

    succeeded = sum(1 for r in results if r.error is None)
    return ImageBatchResponse(
        results=list(results),
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )


@app.post("/analyse/upload", response_model=ImageResponse)
async def analyse_upload(file: UploadFile = File(...)):
    """
    Accepts a direct image file upload (JPEG, PNG, WEBP).
    Runs the same ResNet-50 classifier and returns room_type,
    condition_score (1–5), and confidence.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WEBP.")

    try:
        image_bytes = await file.read()
        result = predict_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return ImageResponse(**result)
