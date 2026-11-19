import asyncio

from app.db.database import Base, engine

# 🔥 IMPORT ALL MODELS (THIS IS THE KEY FIX)
from app.models import documents, user  # noqa: F401  (registers ORM metadata)


async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(init())
