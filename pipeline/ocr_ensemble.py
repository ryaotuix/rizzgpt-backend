# backend/pipeline/ocr_ensemble.py
# ✅ 목적
# - 1차(rec_model) 결과가 "의심스러운 경우"에만
#   2차(EasyOCR)를 돌려서 더 나은 결과를 자동으로 선택한다.
#
# ✅ 최적화 포인트
# - suspicious crop들을 EasyOCR로 "배치" 처리(readtext_batched가 있으면 사용)
# - easyocr 버전에 따라 batched가 없을 수 있으므로 안전 fallback

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OCRLine:
    text: str
    conf: float  # 0~1


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return " ".join(s.split())


def _join_lines(lines: List[OCRLine]) -> str:
    return _clean_text(" ".join([l.text for l in lines if (l.text or "").strip()]))


def _avg_conf(lines: List[OCRLine]) -> float:
    if not lines:
        return 0.0
    return float(sum(l.conf for l in lines) / max(1, len(lines)))


class EasyOCREngine:
    """
    ✅ EasyOCR
    - YOLO로 crop을 이미 했으니, 여기서는 readtext만 사용(정확도 우선)
    - suspicious가 여러 개면 batch로 한 번에 처리 시도
    """

    name = "easyocr"

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False, debug: bool = False):
        self.languages = languages or ["ko", "en"]
        self.gpu = gpu
        self.debug = debug
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def _parse_readtext(self, res) -> List[OCRLine]:
        out: List[OCRLine] = []
        if isinstance(res, list):
            for item in res:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                text = str(item[1]).strip()
                try:
                    conf = float(item[2])
                except Exception:
                    conf = 0.0
                if text:
                    out.append(OCRLine(text=text, conf=conf))
        return out

    def extract_lines(self, crop_img) -> List[OCRLine]:
        try:
            import numpy as np
            arr = np.array(crop_img)
        except Exception:
            return []

        try:
            reader = self._get_reader()
            res = reader.readtext(arr, detail=1, paragraph=False, decoder="greedy")
            return self._parse_readtext(res)
        except Exception:
            return []

    def extract_lines_batch(self, crop_imgs: List[Any]) -> List[List[OCRLine]]:
        """
        ✅ suspicious crop들을 한 번에 처리 (가능하면 readtext_batched 사용)
        - readtext_batched가 없으면 기존처럼 루프
        """
        if not crop_imgs:
            return []

        try:
            import numpy as np
            arrs = [np.array(img) for img in crop_imgs]
        except Exception:
            return [[] for _ in crop_imgs]

        reader = self._get_reader()

        # 1) batched API가 있으면 사용
        if hasattr(reader, "readtext_batched"):
            try:
                batched = reader.readtext_batched(
                    arrs,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                )
                out_all: List[List[OCRLine]] = []
                for one in (batched or []):
                    out_all.append(self._parse_readtext(one))

                if self.debug:
                    print(f"[EASYOCR] batched_used=True n={len(crop_imgs)}")

                # 길이 이상하면 안전 fallback
                if len(out_all) != len(crop_imgs):
                    if self.debug:
                        print("[EASYOCR] batched_len_mismatch -> fallback loop")
                    return [self.extract_lines(img) for img in crop_imgs]

                return out_all
            except Exception as e:
                if self.debug:
                    print(f"[EASYOCR] batched_failed -> fallback loop ({type(e).__name__})")

        # 2) 없거나 실패하면 루프
        if self.debug:
            print(f"[EASYOCR] batched_used=False n={len(crop_imgs)}")
        return [self.extract_lines(img) for img in crop_imgs]


class OCREnsemble:
    """
    ✅ 가벼운 앙상블
    - primary(rec_model 결과) vs easyocr 결과 중 선택
    - 여기서는 batch 실행 지원만 제공(선택은 호출부에서 해도 됨)
    """

    def __init__(
        self,
        enable_easyocr: bool = True,
        easyocr_gpu: bool = False,
        easyocr_langs: Optional[List[str]] = None,
        easyocr_debug: bool = False,
    ):
        self.easy: Optional[EasyOCREngine] = None
        if enable_easyocr:
            try:
                self.easy = EasyOCREngine(
                    languages=easyocr_langs or ["ko", "en"],
                    gpu=easyocr_gpu,
                    debug=easyocr_debug,
                )
            except Exception:
                self.easy = None

    def extract_best(
        self,
        crop_img,
        primary_text: Optional[str] = None,
        primary_lines: Optional[List[OCRLine]] = None,
        suspicious: bool = True,
        **kwargs,
    ) -> Tuple[str, List[OCRLine], Dict[str, Any]]:
        """
        ✅ (기존 단건 API 유지용)
        - 배치 최적화는 extract_text_and_order 쪽에서 사용
        """
        primary_lines = primary_lines or []
        if primary_text is None:
            primary_text = _join_lines(primary_lines)

        meta: Dict[str, Any] = {"candidates": [], "chosen": None}

        if not suspicious or self.easy is None:
            meta["chosen"] = {"engine": "rec", "reason": "not_suspicious_or_no_easy"}
            return _clean_text(primary_text), primary_lines, meta

        easy_lines = self.easy.extract_lines(crop_img)
        easy_text = _join_lines(easy_lines)

        if easy_text:
            meta["chosen"] = {"engine": "easyocr", "reason": "easy_text_non_empty"}
            return _clean_text(easy_text), easy_lines, meta

        meta["chosen"] = {"engine": "rec", "reason": "easy_text_empty"}
        return _clean_text(primary_text), primary_lines, meta
