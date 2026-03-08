# backend/pipeline/extract_text_and_order.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image

# ✅ OCRLine은 ensemble 쪽 정의를 단일 소스로 사용
from backend.pipeline.ocr_ensemble import OCRLine, OCREnsemble
from backend.pipeline.rec_only import rec_batch


# ------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------
@dataclass
class Turn:
    who: str   # "me" | "other"
    text: str


# ------------------------------------------------------------
# 라벨 정규화 / 정렬
# ------------------------------------------------------------
def _norm_label(label: str) -> str:
    label = (label or "").strip().lower()
    if label in ("me", "mine", "self", "right"):
        return "me"
    if label in ("other", "opponent", "them", "left"):
        return "other"
    return label


def order_detections_by_y(
    detections: List[Dict[str, Any]],
    y_tolerance: int = 6,
) -> List[Dict[str, Any]]:
    dets = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        dets.append({
            "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "label": _norm_label(d.get("label", "")),
            "conf": float(d.get("conf", 0.0)),
        })

    dets.sort(key=lambda a: (a["xyxy"][1], a["xyxy"][0]))  # (y1, x1)

    grouped: List[List[Dict[str, Any]]] = []
    for det in dets:
        if not grouped:
            grouped.append([det])
            continue
        last = grouped[-1][-1]
        if abs(det["xyxy"][1] - last["xyxy"][1]) <= y_tolerance:
            grouped[-1].append(det)
        else:
            grouped.append([det])

    ordered: List[Dict[str, Any]] = []
    for g in grouped:
        g.sort(key=lambda a: a["xyxy"][0])  # x1
        ordered.extend(g)

    return ordered


def _safe_crop(img: Image.Image, xyxy: List[int], pad: int = 2) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    w, h = img.size
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))


# ------------------------------------------------------------
# 텍스트/라인 유틸
# ------------------------------------------------------------
def _clean_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return " ".join(s.split())


def _join_ocr_lines(lines: List[OCRLine]) -> str:
    return _clean_text(" ".join([l.text for l in lines if (l.text or "").strip()]))


def _avg_conf(lines: List[OCRLine]) -> float:
    if not lines:
        return 0.0
    return float(sum(l.conf for l in lines) / max(1, len(lines)))


# ------------------------------------------------------------
# is_suspicious
# ------------------------------------------------------------
def is_suspicious_text(text: str, lines: List[OCRLine]) -> Tuple[bool, List[str], float]:
    t = _clean_text(text)
    reasons: List[str] = []
    score = 0.0

    avg = _avg_conf(lines)

    # 1) 빈 텍스트는 무조건 의심
    if not t:
        return True, ["empty_text"], 10.0

    # 2) ✅ rec-only 기준: conf < 0.60 이면 무조건 fallback
    if avg > 0.0 and avg < 0.60:
        reasons.append(f"low_avg_conf:{avg:.2f}")
        score += 10.0

    # 3) 특수기호
    bad_symbols = ["|", "{", "}", "\\", "/"]
    if any(s in t for s in bad_symbols):
        reasons.append("symbol_noise")
        score += 4.0

    return (score >= 4.0), reasons, score


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def extract_text_and_order(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    ocr: Optional[Any] = None,                  # (지금 구조에선 안 씀)
    y_tolerance: int = 6,
    ocr_ensemble: Optional[OCREnsemble] = None, # ✅ rec vs easyocr 선택용
    rec_model: Optional[Any] = None,            # ✅ rec-only 모델
    use_rec_first: bool = True,
) -> Tuple[List[Turn], List[str]]:
    ordered = order_detections_by_y(detections, y_tolerance=y_tolerance)

    turns: List[Turn] = []
    lines_out: List[str] = []

    # 0) label 필터 + crop 모으기
    valid_items: List[Tuple[int, Dict[str, Any], Image.Image]] = []
    idx = 0
    for det in ordered:
        if det["label"] not in ("me", "other"):
            continue
        idx += 1
        crop = _safe_crop(image, det["xyxy"])
        valid_items.append((idx, det, crop))

    if not valid_items:
        return [], []

    crops = [c for (_, _, c) in valid_items]

    # 1) ✅ 1차: rec-only 배치
    if rec_model is not None and use_rec_first:
        rec_lines_batch = rec_batch(crops, rec_model, resize_h=48, debug=False)
        primary_lines_batch: List[List[OCRLine]] = [
            [OCRLine(text=l.text, conf=float(l.conf)) for l in rec_lines]
            for rec_lines in rec_lines_batch
        ]
    else:
        primary_lines_batch = [[] for _ in crops]

    # 2) ✅ suspicious 판정은 "딱 1번만"
    sus_flags: List[bool] = []
    sus_reasons: List[List[str]] = []
    sus_scores: List[float] = []

    primary_texts: List[str] = []
    for primary_lines in primary_lines_batch:
        primary_text = _join_ocr_lines(primary_lines)
        primary_texts.append(primary_text)

        sus, reasons, score = is_suspicious_text(primary_text, primary_lines)
        sus_flags.append(sus)
        sus_reasons.append(reasons)
        sus_scores.append(score)

    # 3) ✅ suspicious인 것만 EasyOCR 실행
    easy_text_map: Dict[int, str] = {}
    if ocr_ensemble is not None and getattr(ocr_ensemble, "easy", None) is not None:
        sus_idx = [i for i, s in enumerate(sus_flags) if s]
        if sus_idx:
            sus_crops = [crops[i] for i in sus_idx]

            # ✅ 네 코드에 batch 함수가 있으면 그걸 쓰고, 없으면 단건으로라도 동작하게 처리
            if hasattr(ocr_ensemble.easy, "extract_lines_batch"):
                batch_lines = ocr_ensemble.easy.extract_lines_batch(sus_crops)
                for i, lines in zip(sus_idx, batch_lines):
                    easy_text_map[i] = _join_ocr_lines(lines)
            else:
                for i in sus_idx:
                    lines = ocr_ensemble.easy.extract_lines(crops[i])
                    easy_text_map[i] = _join_ocr_lines(lines)

    # 4) ✅ 최종 조립 (여기서 turns/lines_out "한 번만" append)
    for i, (idx, det, crop) in enumerate(valid_items):
        primary_text = primary_texts[i]
        final_text = primary_text
        final_engine = "rec"

        if sus_flags[i]:
            easy_text = _clean_text(easy_text_map.get(i, ""))

            # ✅ easyocr 결과가 비어있지 않으면 교체
            if easy_text:
                final_text = easy_text
                final_engine = "easyocr"

            lines_out.append(
                f"{det['label']}({idx}): {final_text}  "
                f"[fallback:{final_engine} from=rec sus={sus_scores[i]:.1f} {sus_reasons[i]}]"
            )
        else:
            lines_out.append(f"{det['label']}({idx}): {final_text}  [rec]")

        turns.append(Turn(who=det["label"], text=final_text))

    return turns, lines_out
