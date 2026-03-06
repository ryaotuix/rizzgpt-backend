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
    allow_origins=["*"],  # 배포 후엔 도메인 제한 추천
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 전역 객체 (처음엔 비워둠)
#    ✅ import 시점엔 무거운 로딩 금지
# =========================================================
ocr_ensemble: Optional[OCREnsemble] = None
rec_model = None
openai_engine: Optional[OpenAIEngine] = None


# =========================================================
# 3) 지연 로딩 함수
# =========================================================
def ensure_openai_engine():
    global openai_engine
    if openai_engine is None:
        try:
            openai_engine = OpenAIEngine()
            print("✅ OpenAI engine loaded")
        except Exception as e:
            print("❌ OpenAI engine init failed:", repr(e))
            openai_engine = None
    return openai_engine


def ensure_ocr_ensemble():
    global ocr_ensemble
    if ocr_ensemble is None:
        try:
            ocr_ensemble = OCREnsemble(
                enable_easyocr=True,
                easyocr_gpu=False,
                easyocr_langs=["ko", "en"],
                easyocr_debug=False,
            )
            print("✅ OCR Ensemble ready (easyocr optional)")
        except Exception as e:
            print("❌ OCR Ensemble init failed:", repr(e))
            ocr_ensemble = None
    return ocr_ensemble


def ensure_rec_model():
    global rec_model
    if rec_model is None:
        try:
            rec_model = load_rec_model()
            print("✅ Rec-only model loaded")
        except Exception as e:
            print("❌ Rec-only model load failed:", repr(e))
            rec_model = None
    return rec_model


# =========================================================
# 4) warmup 함수들
# =========================================================
def warmup_easyocr(ensemble: Optional[OCREnsemble]) -> None:
    if ensemble is None or getattr(ensemble, "easy", None) is None:
        return

    import numpy as np
    dummy = Image.fromarray(np.zeros((64, 256, 3), dtype=np.uint8))
    _ = ensemble.easy.extract_lines(dummy)


def warmup_yolo_once() -> None:
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
# 5) HEALTH
# =========================================================
@app.get("/health")
def health():
    """
    ✅ 절대 무거운 로딩 안 함
    """
    return {
        "ok": True,
        "openai_engine_ready": openai_engine is not None,
        "rec_model_ready": rec_model is not None,
        "ocr_ensemble_ready": ocr_ensemble is not None,
        "time": time.time(),
    }


# =========================================================
# 6) WARMUP
# =========================================================
@app.post("/warmup")
def warmup():
    """
    ✅ 앱 시작 시 1회 호출 추천
    - 여기서 무거운 모델들을 실제 로드
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # (A) OpenAI 엔진 준비
    t1 = time.perf_counter()
    ensure_openai_engine()
    timings["openai_engine_init_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    # (B) OCR ensemble 준비
    t2 = time.perf_counter()
    ensemble = ensure_ocr_ensemble()
    timings["ocr_ensemble_init_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    # (C) Rec model 준비
    t3 = time.perf_counter()
    rm = ensure_rec_model()
    timings["rec_model_init_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    # (D) YOLO warmup
    try:
        t4 = time.perf_counter()
        warmup_yolo_once()
        timings["warmup_yolo_ms"] = round((time.perf_counter() - t4) * 1000, 1)
    except Exception as e:
        timings["warmup_yolo_error"] = repr(e)

    # (E) OCR warmup
    try:
        t5 = time.perf_counter()
        warmup_easyocr(ensemble)
        timings["warmup_ocr_ms"] = round((time.perf_counter() - t5) * 1000, 1)
    except Exception as e:
        timings["warmup_ocr_error"] = repr(e)

    timings["openai_engine_ready"] = 1.0 if openai_engine is not None else 0.0
    timings["rec_model_ready"] = 1.0 if rm is not None else 0.0
    timings["ocr_ensemble_ready"] = 1.0 if ensemble is not None else 0.0
    timings["total_warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "ok": True,
        "timings": timings,
        "note": "Warmup done. Models initialized lazily after server boot.",
    }


# =========================================================
# 7) ANALYZE
# =========================================================
@app.post("/analyze-image")
async def analyze_image(
    device_id: str = Form(...),
    image: UploadFile = File(...),
):
    # ✅ 첫 요청에서도 자동 준비
    engine = ensure_openai_engine()
    ensemble = ensure_ocr_ensemble()
    rm = ensure_rec_model()

    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI engine not available. Check OPENAI_API_KEY and llm/openai_engine.py.",
        )

    if rm is None:
        raise HTTPException(
            status_code=500,
            detail="Rec model not available.",
        )

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    suffix = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await image.read())

    t1 = time.perf_counter()
    timings["save_image_ms"] = round((t1 - t0) * 1000, 1)

    try:
        # (B) YOLO 검출
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
            ocr=None,
            ocr_ensemble=ensemble,
            rec_model=rm,
            use_rec_first=True,
        )
        t4_1 = time.perf_counter()
        timings["ocr_ms"] = round((t4_1 - t4_0) * 1000, 1)

        # (E) LLM
        t5_0 = time.perf_counter()
        llm_turns = [LlmTurn(who=t.who, text=t.text) for t in turns]
        convo = turns_to_llm_convo(llm_turns)

        print(convo)

        out = engine.generate_reply(convo)
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