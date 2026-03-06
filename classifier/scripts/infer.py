"""
infer.py
- 역할: 스크린샷에서 말풍선(bubble) 박스만 YOLO로 검출
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from PIL import Image

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _YOLO_IMPORT_ERROR = e
else:
    _YOLO_IMPORT_ERROR = None


# =========================================================
# 기본 모델 경로 (문자열로 고정)
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "best.pt")


# =========================================================
# YOLO 싱글턴 캐시
# =========================================================
_YOLO_MODEL = None
_YOLO_MODEL_PATH: Optional[str] = None


def _get_yolo_model(model_path: str):
    """
    YOLO 모델을 최초 1회만 로드
    - model_path가 바뀌면 새로 로드
    """
    global _YOLO_MODEL, _YOLO_MODEL_PATH  # ✅ 오타 제거

    if YOLO is None:
        raise RuntimeError(
            "ultralytics YOLO import 실패. pip install ultralytics 필요"
        ) from _YOLO_IMPORT_ERROR

    model_path = str(Path(model_path).resolve())

    # ✅ 이미 같은 경로면 재사용
    if _YOLO_MODEL is not None and _YOLO_MODEL_PATH == model_path:
        return _YOLO_MODEL

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"YOLO model not found: {p.resolve()}")

    _YOLO_MODEL = YOLO(str(p))
    _YOLO_MODEL_PATH = model_path
    return _YOLO_MODEL


def detect_bubbles(
    image: Union[str, Image.Image],
    *,
    model_path: Optional[str] = None,
    conf: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    말풍선 검출 (YOLO)
    """

    model_path = str(model_path or DEFAULT_MODEL_PATH)
    model = _get_yolo_model(model_path)

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False,
    )

    if not results:
        return []

    r0 = results[0]
    boxes = r0.boxes
    names = r0.names or {}

    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []

    detections: List[Dict[str, Any]] = []

    for (x1, y1, x2, y2), cf, cls_id in zip(xyxy, confs, clss):
        cls_id = int(cls_id)
        raw_label = str(names.get(cls_id, cls_id)).lower()

        if raw_label in ("me", "mine", "self", "right"):
            label = "me"
        elif raw_label in ("other", "opponent", "them", "left"):
            label = "other"
        else:
            label = raw_label

        detections.append({
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "label": label,
            "conf": float(cf),
        })

    return detections
