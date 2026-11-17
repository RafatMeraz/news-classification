from fastapi import status
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = status.HTTP_400_BAD_REQUEST
    detail: str = "something went wrong"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail


class BadRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Bad request"


async def app_error_handler(app_error: AppError) -> JSONResponse:
    return JSONResponse(status_code=app_error.status_code, content={"detail": app_error.detail})
