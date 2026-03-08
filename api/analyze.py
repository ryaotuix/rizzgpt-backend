from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import User, UserUsage, Subscription

router = APIRouter(prefix="/analyze", tags=["analyze"])


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
# 무료 5회 제한 체크 전용 API
# 실제 analyze-image 전에 이 로직 그대로 재사용하면 됨
# ==============================
@router.get("/can-use/{device_id}")
def can_use_analyze(device_id: str, db: Session = Depends(get_db)):
    user = (
        db.query(User)
        .filter(
            User.provider == "guest",
            User.provider_user_id == device_id,
        )
        .first()
    )
    if user is None:
        raise HTTPException(status_code=404, detail="Guest user not found")

    usage = db.query(UserUsage).filter(UserUsage.user_id == user.id).first()
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()

    free_used_count = usage.free_used_count if usage else 0
    plan = sub.plan if sub else "free"
    status = sub.status if sub else "inactive"

    is_premium = plan != "free" and status == "active"
    can_use = is_premium or free_used_count < 5

    return {
        "user_id": str(user.id),
        "device_id": device_id,
        "free_used_count": free_used_count,
        "free_remaining_count": max(0, 5 - free_used_count),
        "plan": plan,
        "status": status,
        "is_premium": is_premium,
        "can_use": can_use,
    }
