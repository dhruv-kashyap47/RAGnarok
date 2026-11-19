import json

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.logger import logger
from app.core.redis import safe_cache_get, safe_cache_set
from app.core.security import (
    create_access_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.models.user import User

# Computed lazily on first use: hashing a throwaway password keeps the
# "unknown email" login path as slow as the "wrong password" path so response
# timing does not reveal whether an account exists.
_DUMMY_PASSWORD_HASH: str | None = None


def _dummy_password_hash() -> str:
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = hash_password("timing-equalizer-not-a-real-password")
    return _DUMMY_PASSWORD_HASH


class UserService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, email: str, password: str):
        normalized_email = email.strip().lower()
        logger.info("Create user request: email=%s", normalized_email)

        try:
            result = await self.db.execute(
                select(User).where(User.email == normalized_email)
            )
            existing_user = result.scalars().first()

            if existing_user:
                raise HTTPException(status_code=400, detail="Email already exists")

            # bcrypt is CPU-bound (~100ms); keep it off the event loop.
            hashed = await run_in_threadpool(hash_password, password)

            new_user = User(
                email=normalized_email,
                password=hashed,
                role="user",
            )
            self.db.add(new_user)
            await self.db.commit()
            await self.db.refresh(new_user)

            logger.info("User created: id=%s", new_user.id)
            return {
                "id": str(new_user.id),
                "email": new_user.email,
                "role": new_user.role,
            }

        except HTTPException:
            raise
        except IntegrityError:
            # Concurrent registration with the same email: the pre-check above
            # lost the race, but the unique index caught it. Map to the same
            # client-facing error instead of leaking a 500.
            await self.db.rollback()
            raise HTTPException(status_code=400, detail="Email already exists")
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating user: %s", e)
            raise

    async def login_user(self, email: str, password: str):
        normalized_email = email.strip().lower()

        try:
            result = await self.db.execute(
                select(User).where(User.email == normalized_email)
            )
            db_user = result.scalars().first()

            if not db_user:
                # Equalize work with the existing-user path (timing defense).
                await run_in_threadpool(verify_password, password, _dummy_password_hash())
                raise HTTPException(status_code=400, detail="Invalid email or password")

            valid = await run_in_threadpool(verify_password, password, db_user.password)
            if not valid:
                raise HTTPException(status_code=400, detail="Invalid email or password")

            if password_needs_rehash(db_user.password):
                db_user.password = await run_in_threadpool(hash_password, password)
                await self.db.commit()
                await self.db.refresh(db_user)

            access_token = create_access_token(
                data={"user_id": str(db_user.id), "role": db_user.role}
            )

            # Cache user profile in Redis
            user_data = {
                "id": str(db_user.id),
                "email": db_user.email,
                "role": db_user.role,
            }
            await safe_cache_set(f"user:{db_user.id}", json.dumps(user_data), ex=1800)

            return {"access_token": access_token, "token_type": "bearer"}

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error("Login error: %s", e)
            raise

    async def get_user_by_id(self, user_id: str):
        try:
            # Check Redis first
            cached = await safe_cache_get(f"user:{user_id}")
            if cached:
                return json.loads(cached)

            # Fallback to DB
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            db_user = result.scalars().first()

            if not db_user:
                raise HTTPException(status_code=404, detail="User not found")

            user_data = {
                "id": str(db_user.id),
                "email": db_user.email,
                "role": db_user.role,
            }
            await safe_cache_set(f"user:{user_id}", json.dumps(user_data), ex=1800)
            return user_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error fetching user %s: %s", user_id, e)
            raise

    async def get_users(self, skip: int = 0, limit: int = 10, email: str | None = None):
        try:
            query = select(User)
            if email:
                query = query.where(User.email == email)

            result = await self.db.execute(query.offset(skip).limit(limit))
            users = result.scalars().all()

            return [
                {"id": str(u.id), "email": u.email, "role": u.role}
                for u in users
            ]
        except Exception as e:
            logger.error("Error fetching users: %s", e)
            raise
