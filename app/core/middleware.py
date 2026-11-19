"""
ASGI middleware for defensive request-body size limiting.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class BodySizeLimitMiddleware:
    """
    Reject request bodies larger than *max_bytes*.

    - If the client declares an oversized ``Content-Length`` the body is
      refused on the first read, before any payload is buffered.
    - Chunked uploads without (or with a lying) ``Content-Length`` are counted
      as they arrive and aborted once the limit is crossed.

    The 413 is raised as an ``HTTPException`` from inside ``receive`` (while
    the router is reading the body): FastAPI explicitly re-raises
    HTTPException raised during body parsing, so it reaches the registered
    exception handler as a clean 413 instead of being converted into a
    generic 400 "error parsing the body".
    """

    def __init__(self, app: Callable, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared_length: int | None = None
        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared_length = int(value)
                except ValueError:
                    declared_length = None
                break

        total = 0
        prechecked = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal total, prechecked

            if not prechecked:
                prechecked = True
                if declared_length is not None and declared_length > self.max_bytes:
                    raise HTTPException(
                        status_code=413, detail="Request body too large."
                    )

            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body") or b"")
                if total > self.max_bytes:
                    raise HTTPException(
                        status_code=413, detail="Request body too large."
                    )
            return message

        await self.app(scope, limited_receive, send)
