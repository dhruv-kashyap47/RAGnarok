from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
import json

from app.models.user import User
from app.core.security import (
    hash_password,
    verify_password,
    password_needs_rehash,
    create_access_token,
)
from app.core.logger import logger
from app.core.redis import safe_cache_get, safe_cache_set


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

            new_user = User(
                email=normalized_email,
                password=hash_password(password),
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
                raise HTTPException(status_code=400, detail="Invalid email or password")

            if not verify_password(password, db_user.password):
                raise HTTPException(status_code=400, detail="Invalid email or password")

            if password_needs_rehash(db_user.password):
                db_user.password = hash_password(password)
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
            safe_cache_set(f"user:{db_user.id}", json.dumps(user_data), ex=1800)

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
            cached = safe_cache_get(f"user:{user_id}")
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
            safe_cache_set(f"user:{user_id}", json.dumps(user_data), ex=1800)
            return user_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error fetching user %s: %s", user_id, e)
            raise

    async def get_users(self, skip: int = 0, limit: int = 10, email: str = None):
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
