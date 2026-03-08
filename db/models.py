import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend.db.database import Base

# 유저 테이블
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=True)
    provider = Column(String, nullable=False)  # google / kakao / guest
    provider_user_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# 무료 사용량 테이블
class UserUsage(Base):
    __tablename__ = "user_usage"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    free_used_count = Column(Integer, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


# 구독 상태 테이블
class Subscription(Base):
    __tablename__ = "subscriptions"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    plan = Column(String, default="free")
    status = Column(String, default="inactive")
    expires_at = Column(DateTime(timezone=True))