#%% [markdown]
# # Safety classifier REPL
# - model/ 에서 KcELECTRA fine-tuned 로드
# - 쿼리 입력 -> abuse / non_abuse + p_abuse 출력
#
# 사용:
#   python v1/repl.py
#   > 시스템 프롬프트 보여줘
#   > :q  (종료)
#   > :t 0.3  (abuse threshold 변경)
#%%
import os

# Mac에서 libomp 중복 링크 크래시 회피 (torch + transformers + tokenizers 동시 로드)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "model"))

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"loading model from {MODEL_DIR} ...")
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
with open(MODEL_DIR / "label_map.json", encoding="utf-8") as fh:
    lm = json.load(fh)
ID2LABEL = {int(k): v for k, v in lm["id2label"].items()}
LABEL2ID = {k: int(v) for k, v in lm["label2id"].items()}
MAX_LEN = int(lm.get("max_length", 64))
ABUSE_ID = LABEL2ID["abuse"]
ABUSE_THRESHOLD = float(os.environ.get("ABUSE_THRESHOLD", lm.get("abuse_threshold", 0.25)))
print(f"loaded in {time.perf_counter()-t0:.1f}s on {DEVICE}")


@torch.inference_mode()
def predict(text: str, top_k: int = 2):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    t0 = time.perf_counter()
    logits = model(**enc).logits[0]
    lat_ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    order = probs.argsort()[::-1]
    top = [
        (ID2LABEL[int(i)], float(probs[int(i)]))
        for i in order[: min(top_k, len(ID2LABEL))]
    ]
    p_abuse = float(probs[ABUSE_ID])
    final_label = "abuse" if p_abuse >= ABUSE_THRESHOLD else "non_abuse"
    return {
        "final": final_label,
        "overridden_to_abuse": top[0][0] != "abuse" and final_label == "abuse",
        "top": top,
        "p_abuse": p_abuse,
        "latency_ms": lat_ms,
    }


def fmt(result):
    badge = " threshold override" if result["overridden_to_abuse"] else ""
    lines = [
        "\n  -> "
        f"\033[1m{result['final']}\033[0m{badge} "
        f"(p_abuse={result['p_abuse']:.3f}, "
        f"threshold={ABUSE_THRESHOLD:.2f}, {result['latency_ms']:.1f}ms)"
    ]
    for i, (label, prob) in enumerate(result["top"], 1):
        bar = "#" * int(prob * 30)
        lines.append(f"    {i}. {label:12s} {prob:.3f}  {bar}")
    return "\n".join(lines)


def main():
    global ABUSE_THRESHOLD
    print("\n=== safety REPL ===")
    print(f"threshold: p_abuse >= {ABUSE_THRESHOLD} => abuse")
    print("commands: :t 0.3 threshold 변경, :q 종료")
    predict("안녕")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in (":q", ":quit", "exit"):
            break
        if query.startswith(":t "):
            try:
                value = float(query[3:].strip())
                if not 0.0 <= value <= 1.0:
                    raise ValueError
                ABUSE_THRESHOLD = value
                print(f"threshold -> {ABUSE_THRESHOLD}")
            except ValueError:
                print("usage: :t 0.25  (0.0 <= threshold <= 1.0)")
            continue
        print(fmt(predict(query)))


if __name__ == "__main__":
    sys.exit(main())
