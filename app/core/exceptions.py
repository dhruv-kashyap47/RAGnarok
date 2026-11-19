from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.core.logger import logger


async def http_exception_handler(request: Request, exc: HTTPException):
    # Preserve headers (e.g. WWW-Authenticate on 401) so auth challenges stay
    # spec-compliant and OAuth2 clients behave correctly.
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.detail},
        headers=getattr(exc, "headers", None),
    )


async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error"},
    )
