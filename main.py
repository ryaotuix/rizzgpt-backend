# backend/main.py
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from classifier.scripts.infer import detect_bubbles

from pipeline.extract_text_and_order import extract_text_and_order
from pipeline.ocr_ensemble import OCREnsemble
from pipeline.rec_only import load_rec_model
from pipeline.convo_preprocess import turns_to_llm_convo, Turn as LlmTurn

from llm.openai_engine import OpenAIEngine

from dotenv import load_dotenv

# =========================================================
# 0) 환경변수 로드
# =========================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI()

# =========================================================
# 1) CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 편의. 배포 시 도메인 제한 추천
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 전역 로드 (요청마다 만들면 느림)
#    ✅ "생성"만 여기서 하고
#    ✅ "실제 첫 추론(warmup)"은 /warmup에서 하자
# =========================================================

# --- EasyOCR 앙상블(옵션) ---
ocr_ensemble: Optional[OCREnsemble] = None
try:
    ocr_ensemble = OCREnsemble(
        enable_easyocr=True,
        easyocr_gpu=False,        # 맥이면 보통 False
        easyocr_langs=["ko", "en"],
        easyocr_debug=False,      # 필요할 때만 True
    )
    print("✅ OCR Ensemble ready (easyocr optional)")
except Exception as e:
    print("❌ OCR Ensemble init failed:", repr(e))
    ocr_ensemble = None

# --- Rec-only 모델(초고속 1차 인식) ---
rec_model = None
try:
    rec_model = load_rec_model()
    print("✅ Rec-only model loaded")
except Exception as e:
    print("❌ Rec-only model load failed:", repr(e))
    rec_model = None

# --- OpenAI 엔진 ---
openai_engine: Optional[OpenAIEngine] = None
try:
    openai_engine = OpenAIEngine()  # OPENAI_API_KEY 필요
    print("✅ OpenAI engine loaded")
except Exception as e:
    print("❌ OpenAI engine init failed:", repr(e))
    openai_engine = None


# =========================================================
# 3) warmup 함수들
# =========================================================
def warmup_easyocr(ensemble: Optional[OCREnsemble]) -> None:
    """
    ✅ EasyOCR는 첫 호출이 느릴 수 있어서 더미 1회로 워밍업
    """
    if ensemble is None or getattr(ensemble, "easy", None) is None:
        return
    import numpy as np

    dummy = Image.fromarray(np.zeros((64, 256, 3), dtype=np.uint8))
    _ = ensemble.easy.extract_lines(dummy)


def warmup_yolo_once() -> None:
    """
    ✅ detect_bubbles 내부에서 YOLO 모델/세션 로드가 일어난다면
    더미 이미지로 1회 호출해서 선불 처리
    """
    dummy = Image.new("RGB", (640, 640), (255, 255, 255))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
            dummy.save(tmp_path)

        _ = detect_bubbles(image=tmp_path, conf=0.25)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================================================
# ✅ 4) HEALTH (가벼워야 함)
# =========================================================
@app.get("/health")
def health():
    """
    ✅ 서버 생존 + 전역 객체 생성 여부만 확인
    (여기서 절대 무거운 warmup/추론 하지 말기)
    """
    return {
        "ok": True,
        "openai_engine": openai_engine is not None,
        "rec_model": rec_model is not None,
        "ocr_ensemble": ocr_ensemble is not None,
        "time": time.time(),
    }


# =========================================================
# ✅ 5) WARMUP (앱 시작 시 1회 호출 추천)
# =========================================================
@app.post("/warmup")
def warmup():
    """
    ✅ 앱 시작 시 호출해서 '첫 요청 느려짐'을 선불로 처리
    - YOLO 1회
    - EasyOCR 1회
    - OpenAI는 비용 때문에 호출하지 않음(엔진 존재 체크만)
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # (A) YOLO warmup
    try:
        t1 = time.perf_counter()
        warmup_yolo_once()
        timings["warmup_yolo_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    except Exception as e:
        timings["warmup_yolo_error"] = repr(e)

    # (B) OCR warmup
    try:
        t2 = time.perf_counter()
        warmup_easyocr(ocr_ensemble)
        timings["warmup_ocr_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    except Exception as e:
        timings["warmup_ocr_error"] = repr(e)

    # (C) OpenAI 체크(비용 방지)
    timings["openai_engine_ready"] = 1.0 if openai_engine is not None else 0.0
    timings["total_warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "ok": True,
        "timings": timings,
        "note": "Warmup done. YOLO+OCR initialized. OpenAI not called to avoid cost.",
    }


# =========================================================
# ✅ 6) ANALYZE
# =========================================================
@app.post("/analyze-image")
async def analyze_image(
    device_id: str = Form(...),
    image: UploadFile = File(...),
):
    # ✅ OpenAI가 없으면 바로 에러
    if openai_engine is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI engine not available. Check OPENAI_API_KEY and llm/openai_engine.py.",
        )

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # (A) 업로드 파일 저장
    suffix = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await image.read())

    t1 = time.perf_counter()
    timings["save_image_ms"] = round((t1 - t0) * 1000, 1)

    try:
        # (B) YOLO 말풍선 검출
        dets = detect_bubbles(image=tmp_path, conf=0.25)
        t2 = time.perf_counter()
        timings["yolo_detect_ms"] = round((t2 - t1) * 1000, 1)

        # (C) 이미지 로드
        pil_img = Image.open(tmp_path).convert("RGB")
        t3 = time.perf_counter()
        timings["load_image_ms"] = round((t3 - t2) * 1000, 1)

        # (D) OCR + 순서 정렬
        t4_0 = time.perf_counter()
        turns, debug_lines = extract_text_and_order(
            image=pil_img,
            detections=dets,
            ocr=None,                  # ✅ PaddleOCR 끔
            ocr_ensemble=ocr_ensemble, # ✅ EasyOCR fallback
            rec_model=rec_model,       # ✅ rec-only 모델
            use_rec_first=True,
        )
        t4_1 = time.perf_counter()
        timings["ocr_ms"] = round((t4_1 - t4_0) * 1000, 1)

        # (E) LLM
        t5_0 = time.perf_counter()

        llm_turns = [LlmTurn(who=t.who, text=t.text) for t in turns]
        convo = turns_to_llm_convo(llm_turns)  # ✅ m/o 축약 + 연속 동일 화자 압축

        print(convo)

        out = openai_engine.generate_reply(convo)
        reply_text = (out.get("reply", "") if isinstance(out, dict) else "") or ""

        replies = {"reply": reply_text, "provider": "openai:gpt-4o-mini"}

        t5_1 = time.perf_counter()
        timings["llm_ms"] = round((t5_1 - t5_0) * 1000, 1)

        return {
            "success": True,
            "device_id": device_id,
            "num_detections": len(dets),
            "timings": timings,
            "detections": dets,
            "turns": [{"who": t.who, "text": t.text} for t in turns],
            "debug_lines": debug_lines,
            "replies": replies,
        }

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
