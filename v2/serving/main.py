"""mjuclaw-safety-serving — KcELECTRA 기반 한국어 abuse 판별 추론 서버.

router 가 메시지를 forward하기 전에 호출하여:
- abuse 차단(LLM 0회로 차단 결정)
- non_abuse는 OpenClaw로 forward

binary safety taxonomy v2 모델 (model/ 디렉토리). 최종 label은
p(abuse) >= ABUSE_THRESHOLD 여부만으로 결정한다. router 호환을 위해
응답 필드 이름(final, p_abuse, top, latency_ms, overridden_to_abuse)은 유지한다.
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

from .contract import (
    ABUSE_LABEL,
    final_from_p_abuse,
    clamp_top_k,
    validate_label_map,
    validate_probability,
)
from .library_companion_allow import is_library_companion_student_id_submission

# Mac libomp 중복 링크 회피 (Linux 컨테이너에선 무해).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger("mjuclaw-safety-serving")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ── 모델 로드 ────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/intent-classifier/model"))
ABUSE_THRESHOLD = validate_probability(
    float(os.environ.get("ABUSE_THRESHOLD", "0.25")),
    name="ABUSE_THRESHOLD",
)
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
LABEL2ID: dict[str, int] = {k: int(v) for k, v in _label_map["label2id"].items()}
validate_label_map(LABEL2ID)
ABUSE_ID: int = int(LABEL2ID[ABUSE_LABEL])
MAX_LEN: int = int(_label_map.get("max_length", 64))
logger.info(
    "model loaded in %.1fs (task=%s, classes=%d, max_len=%d, abuse_threshold=%.2f)",
    time.perf_counter() - _t0,
    _label_map.get("task", "unknown"),
    len(ID2LABEL),
    MAX_LEN,
    ABUSE_THRESHOLD,
)


# ── FastAPI ─────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    # router는 현재 top_k=3을 보내므로 binary 모델이어도 3까지는 받아야 한다.
    # 실제 반환 개수는 label 수로 clamp한다.
    top_k: int = Field(2, ge=1, le=15)


class TopItem(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    final: str
    overridden_to_abuse: bool
    p_abuse: float
    top: list[TopItem]
    latency_ms: float


app = FastAPI(title="mjuclaw-safety-serving", version="0.2.0")


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
    n_top = clamp_top_k(top_k, len(ID2LABEL))
    top = [
        {"label": ID2LABEL[int(i)], "score": float(probs[int(i)])}
        for i in order[:n_top]
    ]
    top1_label = top[0]["label"]
    p_abuse = float(probs[ABUSE_ID])
    final = final_from_p_abuse(p_abuse, ABUSE_THRESHOLD)
    overridden = top1_label != ABUSE_LABEL and final == ABUSE_LABEL
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
        "task": _label_map.get("task", "binary_safety"),
        "classes": len(ID2LABEL),
        "labels": [ID2LABEL[i] for i in sorted(ID2LABEL)],
        "abuse_threshold": ABUSE_THRESHOLD,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    req: ClassifyRequest,
    authorization: str | None = Header(default=None),
) -> ClassifyResponse:
    _verify_auth(authorization)
    result = _predict(req.text, req.top_k)
    if (
        result["final"] == ABUSE_LABEL
        and is_library_companion_student_id_submission(req.text)
    ):
        # 도서관 스터디룸 예약은 사용자가 동반자 이름/학번을 직접 제공해야 한다.
        # 내부 데이터 조회가 아닌 직접 제공 목록이면 모델의 PII abuse 판정을 통과시킨다.
        logger.info(
            "library companion allow override: p_abuse=%.3f text_len=%d",
            result["p_abuse"],
            len(req.text),
        )
        result["final"] = "non_abuse"
        result["overridden_to_abuse"] = False
    if result["overridden_to_abuse"]:
        # 너무 많이 찍힐 수 있으니 메시지 본문은 요약만 로그.
        logger.info(
            "abuse threshold override: p_abuse=%.3f top1=%s text_len=%d",
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
