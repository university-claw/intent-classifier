"""mjuclaw-intent-serving — KcELECTRA 기반 한국어 의도 분류 추론 서버.

router 가 메시지를 forward하기 전에 호출하여:
- abuse 차단(LLM 0회로 차단 결정)
- (향후) service.* 라우팅 분기

15-class taxonomy v1 모델 (model/ 디렉토리). abuse 라벨은 임계값
보정으로 recall 보강 (top1이 abuse 아니어도 p(abuse) >= ABUSE_THRESHOLD 면
abuse 로 덮어씀 — repl.py 와 동일 로직).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

# Mac libomp 중복 링크 회피 (Linux 컨테이너에선 무해).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger("mjuclaw-intent-serving")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ── 모델 로드 ────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/intent-classifier/model"))
ABUSE_THRESHOLD = float(os.environ.get("ABUSE_THRESHOLD", "0.25"))
AUTH_TOKEN = os.environ.get("CLASSIFIER_AUTH_TOKEN", "")

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info("loading model from %s on device=%s", MODEL_DIR, DEVICE)
_t0 = time.perf_counter()
try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
except (ValueError, KeyError) as exc:
    # tokenizer_config.json의 `tokenizer_class`가 현재 transformers 버전에 없는
    # 이름(예: "TokenizersBackend")이면 AutoTokenizer가 실패한다. tokenizer.json
    # (HuggingFace tokenizers raw)을 PreTrainedTokenizerFast로 직접 로드해 우회.
    logger.warning(
        "AutoTokenizer 실패 → PreTrainedTokenizerFast(tokenizer.json) fallback (이유: %s)",
        exc,
    )
    with open(MODEL_DIR / "tokenizer_config.json", encoding="utf-8") as fh:
        _tok_cfg = json.load(fh)
    TOKENIZER = PreTrainedTokenizerFast(
        tokenizer_file=str(MODEL_DIR / "tokenizer.json"),
        unk_token=_tok_cfg.get("unk_token", "[UNK]"),
        pad_token=_tok_cfg.get("pad_token", "[PAD]"),
        cls_token=_tok_cfg.get("cls_token", "[CLS]"),
        sep_token=_tok_cfg.get("sep_token", "[SEP]"),
        mask_token=_tok_cfg.get("mask_token", "[MASK]"),
        model_max_length=_tok_cfg.get("model_max_length", 512),
    )
MODEL = (
    AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
)
with open(MODEL_DIR / "label_map.json", encoding="utf-8") as fh:
    _label_map = json.load(fh)
ID2LABEL: dict[int, str] = {int(k): v for k, v in _label_map["id2label"].items()}
ABUSE_ID: int = int(_label_map["label2id"]["abuse"])
MAX_LEN: int = int(_label_map.get("max_length", 64))
logger.info(
    "model loaded in %.1fs (classes=%d, max_len=%d, abuse_threshold=%.2f)",
    time.perf_counter() - _t0,
    len(ID2LABEL),
    MAX_LEN,
    ABUSE_THRESHOLD,
)


# ── FastAPI ─────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(3, ge=1, le=15)


class TopItem(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    final: str
    overridden_to_abuse: bool
    p_abuse: float
    top: list[TopItem]
    latency_ms: float


app = FastAPI(title="mjuclaw-intent-serving", version="0.1.0")


def _verify_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return  # auth disabled (dev only)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    # 상수 시간 비교
    if len(token) != len(AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="invalid token")
    mismatch = 0
    for a, b in zip(token, AUTH_TOKEN):
        mismatch |= ord(a) ^ ord(b)
    if mismatch != 0:
        raise HTTPException(status_code=401, detail="invalid token")


@torch.inference_mode()
def _predict(text: str, top_k: int) -> dict[str, Any]:
    enc = TOKENIZER(
        text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(DEVICE)
    t0 = time.perf_counter()
    logits = MODEL(**enc).logits[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    order = probs.argsort()[::-1]
    top = [
        {"label": ID2LABEL[int(i)], "score": float(probs[int(i)])}
        for i in order[:top_k]
    ]
    top1_label = top[0]["label"]
    p_abuse = float(probs[ABUSE_ID])
    overridden = top1_label != "abuse" and p_abuse >= ABUSE_THRESHOLD
    final = "abuse" if overridden else top1_label
    return {
        "final": final,
        "overridden_to_abuse": overridden,
        "p_abuse": p_abuse,
        "top": top,
        "latency_ms": latency_ms,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "device": DEVICE,
        "classes": len(ID2LABEL),
        "abuse_threshold": ABUSE_THRESHOLD,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    req: ClassifyRequest,
    authorization: str | None = Header(default=None),
) -> ClassifyResponse:
    _verify_auth(authorization)
    result = _predict(req.text, req.top_k)
    if result["overridden_to_abuse"]:
        # 너무 많이 찍힐 수 있으니 메시지 본문은 요약만 로그.
        logger.info(
            "abuse override: p_abuse=%.3f top1=%s text_len=%d",
            result["p_abuse"],
            result["top"][0]["label"],
            len(req.text),
        )
    return ClassifyResponse(**result)


# 워밍업 — 첫 요청 latency 줄이기
try:
    _warmup = _predict("안녕", 1)
    logger.info("warmup ok (%.1fms)", _warmup["latency_ms"])
except Exception as exc:  # noqa: BLE001
    logger.warning("warmup failed: %s", exc)
