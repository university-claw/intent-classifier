#%% [markdown]
# # 실시간 추론 REPL
# - model/ 에서 KcELECTRA fine-tuned 로드
# - 쿼리 입력 → top-3 intent + confidence 출력
# - 장비: MPS(Mac) > CUDA > CPU 자동 선택
#
# 사용:
#   python3 v1/repl.py
#   > 과제 뭐남았어
#   > :q  (종료)
#   > :t 0.3  (abuse threshold 변경)
#%%
import os
# Mac에서 libomp 중복 링크 크래시 회피 (torch + transformers + tokenizers 동시 로드)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"

# 장비 선택
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
with open(MODEL_DIR / "label_map.json") as f:
    lm = json.load(f)
ID2LABEL = {int(k): v for k, v in lm["id2label"].items()}
MAX_LEN = lm.get("max_length", 64)
ABUSE_ID = lm["label2id"]["abuse"]
print(f"loaded in {time.perf_counter()-t0:.1f}s on {DEVICE}")

# abuse recall이 baseline에서 0.795였으므로 threshold 보정으로 recall 끌어올림
# p(abuse) > ABUSE_THRESHOLD 면 top1 아니어도 abuse로 덮어씀
ABUSE_THRESHOLD = 0.25

#%% predict 1회
@torch.inference_mode()
def predict(text: str, top_k: int = 3):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    t0 = time.perf_counter()
    logits = model(**enc).logits[0]
    lat_ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    order = probs.argsort()[::-1]
    top = [(ID2LABEL[int(i)], float(probs[int(i)])) for i in order[:top_k]]
    # abuse override
    top1_label = top[0][0]
    final_label = top1_label
    overridden = False
    if top1_label != "abuse" and probs[ABUSE_ID] >= ABUSE_THRESHOLD:
        final_label = "abuse"
        overridden = True
    return {
        "final": final_label,
        "overridden_to_abuse": overridden,
        "top": top,
        "p_abuse": float(probs[ABUSE_ID]),
        "latency_ms": lat_ms,
    }

#%% REPL
def fmt(r):
    badge = " ⚠️abuse override" if r["overridden_to_abuse"] else ""
    lines = [f"\n  → \033[1m{r['final']}\033[0m{badge}   (p_abuse={r['p_abuse']:.3f}, {r['latency_ms']:.1f}ms)"]
    for i, (lab, p) in enumerate(r["top"], 1):
        bar = "█" * int(p * 30)
        lines.append(f"    {i}. {lab:35s} {p:.3f}  {bar}")
    return "\n".join(lines)

def main():
    global ABUSE_THRESHOLD
    print("\n=== intent REPL ===")
    print(f"threshold: abuse > {ABUSE_THRESHOLD}  (명령: :t 0.3 로 변경, :q 종료)")
    # 워밍업 (첫 호출이 느림)
    predict("안녕")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q: continue
        if q in (":q", ":quit", "exit"): break
        if q.startswith(":t "):
            try:
                ABUSE_THRESHOLD = float(q[3:].strip())
                print(f"threshold → {ABUSE_THRESHOLD}")
            except ValueError:
                print("usage: :t 0.25")
            continue
        r = predict(q)
        print(fmt(r))

if __name__ == "__main__":
    main()
