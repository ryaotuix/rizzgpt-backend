from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import User, UserUsage, Subscription

router = APIRouter(prefix="/auth", tags=["auth"])


# ==============================
# DB 세션 의존성
# ==============================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==============================
# 요청 / 응답 스키마
# ==============================
class GuestAuthRequest(BaseModel):
    device_id: str


class GuestAuthResponse(BaseModel):
    user_id: str
    provider: str
    provider_user_id: str
    email: str | None
    free_used_count: int
    plan: str
    status: str


class MeResponse(BaseModel):
    user_id: str
    provider: str
    provider_user_id: str
    email: str | None
    free_used_count: int
    free_remaining_count: int
    plan: str
    status: str


# ==============================
# guest 로그인 / 회원 생성
# ==============================
@router.post("/guest", response_model=GuestAuthResponse)
def auth_guest(payload: GuestAuthRequest, db: Session = Depends(get_db)):
    # 기존 guest 유저 조회
    user = (
        db.query(User)
        .filter(
            User.provider == "guest",
            User.provider_user_id == payload.device_id,
        )
        .first()
    )

    # 없으면 생성
    if user is None:
        user = User(
            provider="guest",
            provider_user_id=payload.device_id,
            email=None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # 사용량 row 없으면 생성
    usage = db.query(UserUsage).filter(UserUsage.user_id == user.id).first()
    if usage is None:
        usage = UserUsage(user_id=user.id, free_used_count=0)
        db.add(usage)
        db.commit()
        db.refresh(usage)

    # 구독 row 없으면 생성
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if sub is None:
        sub = Subscription(user_id=user.id, plan="free", status="inactive")
        db.add(sub)
        db.commit()
        db.refresh(sub)

    return GuestAuthResponse(
        user_id=str(user.id),
        provider=user.provider,
        provider_user_id=user.provider_user_id,
        email=user.email,
        free_used_count=usage.free_used_count,
        plan=sub.plan,
        status=sub.status,
    )


# ==============================
# 내 정보 조회
# 지금은 user_id를 직접 받는 단순 버전
# 나중에 JWT 붙이면 header 기반으로 바꾸면 됨
# ==============================
@router.get("/me/{user_id}", response_model=MeResponse)
def get_me(user_id: UUID, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    usage = db.query(UserUsage).filter(UserUsage.user_id == user.id).first()
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()

    free_used_count = usage.free_used_count if usage else 0
    free_remaining_count = max(0, 5 - free_used_count)

    plan = sub.plan if sub else "free"
    status = sub.status if sub else "inactive"

    return MeResponse(
        user_id=str(user.id),
        provider=user.provider,
        provider_user_id=user.provider_user_id,
        email=user.email,
        free_used_count=free_used_count,
        free_remaining_count=free_remaining_count,
        plan=plan,
        status=status,
    )
