# backend/main.py
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
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
    allow_origins=["*"],  # 배포 후에는 앱 도메인만 허용 추천
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 전역 캐시 (처음엔 None)
#    ✅ 무거운 import / 모델 로드는 절대 top-level에서 하지 않음
# =========================================================
detect_bubbles_fn = None
extract_text_and_order_fn = None
turns_to_llm_convo_fn = None
LlmTurnCls = None

ocr_ensemble = None
rec_model = None
openai_engine = None


# =========================================================
# 3) 지연 import / 지연 로드 함수
# =========================================================
def get_detect_bubbles():
    global detect_bubbles_fn
    if detect_bubbles_fn is None:
        from classifier.scripts.infer import detect_bubbles
        detect_bubbles_fn = detect_bubbles
    return detect_bubbles_fn


def get_extract_text_and_order():
    global extract_text_and_order_fn
    if extract_text_and_order_fn is None:
        from pipeline.extract_text_and_order import extract_text_and_order
        extract_text_and_order_fn = extract_text_and_order
    return extract_text_and_order_fn


def get_convo_helpers():
    global turns_to_llm_convo_fn, LlmTurnCls
    if turns_to_llm_convo_fn is None or LlmTurnCls is None:
        from pipeline.convo_preprocess import turns_to_llm_convo, Turn
        turns_to_llm_convo_fn = turns_to_llm_convo
        LlmTurnCls = Turn
    return turns_to_llm_convo_fn, LlmTurnCls


def ensure_openai_engine():
    global openai_engine
    if openai_engine is None:
        try:
            from llm.openai_engine import OpenAIEngine
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
            from pipeline.ocr_ensemble import OCREnsemble
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
            from pipeline.rec_only import load_rec_model
            rec_model = load_rec_model()
            print("✅ Rec-only model loaded")
        except Exception as e:
            print("❌ Rec-only model load failed:", repr(e))
            rec_model = None
    return rec_model


# =========================================================
# 4) warmup 함수
# =========================================================
def warmup_easyocr(ensemble) -> None:
    if ensemble is None or getattr(ensemble, "easy", None) is None:
        return

    import numpy as np
    dummy = Image.fromarray(np.zeros((64, 256, 3), dtype=np.uint8))
    _ = ensemble.easy.extract_lines(dummy)


def warmup_yolo_once() -> None:
    detect_bubbles = get_detect_bubbles()
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
# 5) 아주 가벼운 루트 엔드포인트
#    ✅ 서버가 떴는지 바로 확인 가능
# =========================================================
@app.get("/")
def root():
    return {"ok": True, "service": "rizzgpt-backend"}


# =========================================================
# 6) HEALTH
#    ✅ 무거운 로드 절대 금지
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_engine_ready": openai_engine is not None,
        "rec_model_ready": rec_model is not None,
        "ocr_ensemble_ready": ocr_ensemble is not None,
        "time": time.time(),
    }


# =========================================================
# 7) WARMUP
#    ✅ 앱 시작 후 1회 호출 권장
# =========================================================
@app.post("/warmup")
def warmup():
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # OpenAI 엔진 준비
    t1 = time.perf_counter()
    ensure_openai_engine()
    timings["openai_engine_init_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    # OCR ensemble 준비
    t2 = time.perf_counter()
    ensemble = ensure_ocr_ensemble()
    timings["ocr_ensemble_init_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    # rec model 준비
    t3 = time.perf_counter()
    rm = ensure_rec_model()
    timings["rec_model_init_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    # YOLO warmup
    try:
        t4 = time.perf_counter()
        warmup_yolo_once()
        timings["warmup_yolo_ms"] = round((time.perf_counter() - t4) * 1000, 1)
    except Exception as e:
        timings["warmup_yolo_error"] = repr(e)

    # EasyOCR warmup
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
# 8) ANALYZE
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
            detail="OpenAI engine not available. Check OPENAI_API_KEY.",
        )

    if rm is None:
        raise HTTPException(
            status_code=500,
            detail="Rec model not available.",
        )

    detect_bubbles = get_detect_bubbles()
    extract_text_and_order = get_extract_text_and_order()
    turns_to_llm_convo, LlmTurn = get_convo_helpers()

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 업로드 파일 저장
    suffix = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await image.read())

    t1 = time.perf_counter()
    timings["save_image_ms"] = round((t1 - t0) * 1000, 1)

    try:
        # YOLO 검출
        dets = detect_bubbles(image=tmp_path, conf=0.25)
        t2 = time.perf_counter()
        timings["yolo_detect_ms"] = round((t2 - t1) * 1000, 1)

        # 이미지 로드
        pil_img = Image.open(tmp_path).convert("RGB")
        t3 = time.perf_counter()
        timings["load_image_ms"] = round((t3 - t2) * 1000, 1)

        # OCR + 순서 정렬
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

        # LLM
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