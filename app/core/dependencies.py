from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_access_token
from app.db.database import get_db
from app.services.user_service import UserService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_AUTH_HEADER = {"WWW-Authenticate": "Bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    identity = verify_access_token(token)
    if identity is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired token", headers=_AUTH_HEADER
        )

    # Tokens are minted by us and always carry a UUID user id. Reject anything
    # else here so malformed (or legacy) tokens cannot cause type errors and
    # 500s deep inside the data layer.
    try:
        UUID(str(identity["user_id"]))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=401, detail="Invalid or expired token", headers=_AUTH_HEADER
        )

    return identity


def get_user_service(db: AsyncSession = Depends(get_db)):
    return UserService(db)
