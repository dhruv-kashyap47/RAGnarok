from app.db.database import Base
from app.db.database import engine

# 🔥 IMPORT ALL MODELS (THIS IS THE KEY FIX)
from app.models import user, documents

import asyncio

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(init())
