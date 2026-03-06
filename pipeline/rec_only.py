# backend/pipeline/rec_only.py
from __future__ import annotations

from typing import Any, List, Optional
from dataclasses import dataclass
from PIL import Image

try:
    import paddlex as pdx
except Exception:
    pdx = None


@dataclass
class RecLine:
    text: str
    conf: float


_REC_MODEL = None


def load_rec_model(model_name: str = "korean_PP-OCRv5_mobile_rec"):
    """
    ✅ PaddleX의 rec(인식) 모델만 로드
    - det/uvdoc/ori 같은 파이프라인을 안 씀 (속도 이득)
    """
    global _REC_MODEL
    if _REC_MODEL is not None:
        return _REC_MODEL

    if pdx is None:
        raise RuntimeError("paddlex import 실패. pip install paddlex 필요")

    # PaddleX 버전에 따라 create_model / create_predictor 이름이 다를 수 있어서 유연하게 처리
    if hasattr(pdx, "create_model"):
        _REC_MODEL = pdx.create_model(model_name)
    else:
        # 구버전 대비
        _REC_MODEL = pdx.create_predictor(model_name)  # type: ignore

    return _REC_MODEL


def _resize_for_rec(img: Image.Image, target_h: int = 48) -> Image.Image:
    """
    ✅ rec 모델은 높이를 고정(예:48)하면 빨라지고 안정적임
    - 가로는 비율 유지
    """
    w, h = img.size
    if h <= 0:
        return img
    if h == target_h:
        return img
    new_w = max(1, int(w * (target_h / h)))
    return img.resize((new_w, target_h))


def rec_batch(
    crops: List[Image.Image],
    rec_model: Any,
    *,
    resize_h: int = 48,
    debug: bool = False,
) -> List[List[RecLine]]:
    if not crops:
        return []

    crops2 = [_resize_for_rec(c, target_h=resize_h) for c in crops]

    # ✅ PaddleX는 numpy.ndarray 또는 str만 지원한다고 로그가 말해줌
    import numpy as np
    arrs = [np.array(c) for c in crops2]   # <-- PIL -> ndarray 변환 (필수)

    res = rec_model.predict(arrs)          # <-- 여기엔 arrs만 넣기

    # ✅ generator면 list로 펼치기
    if not isinstance(res, list):
        res = list(res)

    if debug:
        print("[REC DEBUG] materialized type(res) =", type(res), flush=True)
        print("[REC DEBUG] len(res) =", len(res), flush=True)
        print("[REC DEBUG] first =", (res[0] if res else None), flush=True)
        if res:
            try:
                print("[REC DEBUG] first keys =", list(dict(res[0]).keys()), flush=True)
            except Exception as e:
                print("[REC DEBUG] dict() failed:", repr(e), flush=True)
                print("[REC DEBUG] dir(first) =", [x for x in dir(res[0]) if not x.startswith("_")][:60], flush=True)

    # ---------- 파싱 ----------
    def _pick(item, key, default=None):
        try:
            if hasattr(item, "get"):
                return item.get(key, default)
            d = dict(item)
            return d.get(key, default)
        except Exception:
            return default

    out: List[List[RecLine]] = []
    for item in res:
        lines: List[RecLine] = []

        # 단수형
        t = _pick(item, "rec_text", None) or _pick(item, "text", None)
        sc = _pick(item, "rec_score", None) or _pick(item, "score", None)

        # 복수형
        if not t:
            rec_texts = _pick(item, "rec_texts", None)
            rec_scores = _pick(item, "rec_scores", None)
            if isinstance(rec_texts, list) and rec_texts:
                t = " ".join([str(x).strip() for x in rec_texts if str(x).strip()])
                if isinstance(rec_scores, list) and rec_scores:
                    try:
                        sc = float(sum(float(x) for x in rec_scores) / max(1, len(rec_scores)))
                    except Exception:
                        sc = 0.0

        t = str(t).strip() if t is not None else ""
        try:
            conf = float(sc) if sc is not None else 0.0
        except Exception:
            conf = 0.0

        if t:
            lines.append(RecLine(text=t, conf=conf))

        out.append(lines)

    # 길이 보정
    if len(out) < len(crops):
        out.extend([[] for _ in range(len(crops) - len(out))])
    elif len(out) > len(crops):
        out = out[:len(crops)]

    return out
