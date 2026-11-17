import time

from fastapi import FastAPI, Request

from news_classification.classifier.router import classifier_router
from news_classification.core.logger_utils import get_logger

logger = get_logger("api")

app = FastAPI()

app.include_router(classifier_router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info(f"--> {request.method} {request.url.path}")
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"<-- {request.method} {request.url.path} {response.status_code} ({duration_ms:.1f}ms)")
    return response
