import uuid

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")
    # Used to reap abandoned guest accounts (see GUEST_CLEANUP_DAYS).
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
