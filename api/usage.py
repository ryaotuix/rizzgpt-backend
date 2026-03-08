from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import User, UserUsage, Subscription

router = APIRouter(prefix="/usage", tags=["usage"])


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
# device_id 기준 사용량 조회
# ==============================
@router.get("/guest/{device_id}")
def get_guest_usage(device_id: str, db: Session = Depends(get_db)):
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
    free_remaining_count = max(0, 5 - free_used_count)

    plan = sub.plan if sub else "free"
    status = sub.status if sub else "inactive"
    is_premium = plan != "free" and status == "active"

    return {
        "user_id": str(user.id),
        "provider": user.provider,
        "device_id": device_id,
        "free_used_count": free_used_count,
        "free_remaining_count": free_remaining_count,
        "plan": plan,
        "status": status,
        "is_premium": is_premium,
    }
