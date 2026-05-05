#%% [markdown]
# # Stratified train/val split
# - synth-data/*.jsonl → data/v1/{train,val}.jsonl
# - 85/15 비율, 클래스별 독립 샘플링 (stratified)
# - 같은 norm(text)는 한쪽에만 가도록 leakage 차단
#%%
import json, re, random, unicodedata
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]              # intent-classifier/
RAW = ROOT.parent / "synth-data"                         # mjuclaw/synth-data/
OUT = ROOT / "data" / "v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_RATIO = 0.15

FILE_TO_INTENT = {
    "abuse.jsonl": "abuse",
    "chat.jsonl": "chat",
    "service-cafeteria-today.jsonl": "service.cafeteria.today",
    "service-library-my-loans.jsonl": "service.library.my_loans",
    "service-library-search.jsonl": "service.library.search",
    "service-lms-digest.jsonl": "service.lms.digest",
    "service-lms-due-assignment.jsonl": "service.lms.due_assignments",
    "service-lms-incomplete-online.jsonl": "service.lms.incomplete_online",
    "service-lms-unread-notices.jsonl": "service.lms.unread_notices",
    "service-lms-unsubmitted.jsonl": "service.lms.unsubmitted",
    "service-msi-grades.jsonl": "service.msi.grades",
    "service-msi-schedule.jsonl": "service.msi.schedule",
    "service-news-recent.jsonl": "service.news.recent",
    "service-news-search.jsonl": "service.news.search",
    "service-ucheck-attendance.jsonl": "service.ucheck.attendance",
}

def norm(t):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", t)).lower()

#%% 로드 + leakage 방지
by_intent = defaultdict(list)
seen_keys = set()  # 전체에서 처음 본 텍스트만 채택 (혹시 파일간 중복 남아있어도 차단)
for fname, intent in FILE_TO_INTENT.items():
    p = RAW / fname
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            assert r["intent"] == intent, f"label mismatch in {fname}: {r}"
            k = norm(r["text"])
            if k in seen_keys: continue
            seen_keys.add(k)
            by_intent[intent].append(r)

print("클래스별 총량:")
for k in sorted(by_intent): print(f"  {k:40s} {len(by_intent[k])}")

#%% stratified split
random.seed(SEED)
train, val = [], []
for intent, rows in by_intent.items():
    rows = rows[:]  # copy
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val.extend(rows[:n_val])
    train.extend(rows[n_val:])

random.shuffle(train)
random.shuffle(val)

#%% 저장 (최종 스키마: {text, intent} 만 유지)
def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "intent": r["intent"]}, ensure_ascii=False) + "\n")

dump(OUT / "train.jsonl", train)
dump(OUT / "val.jsonl", val)

print(f"\ntrain: {len(train)}")
print(f"val:   {len(val)}")
print(f"\nval 클래스 분포:")
for k, v in sorted(Counter(r['intent'] for r in val).items()):
    print(f"  {k:40s} {v}")

#%% leak sanity check
train_keys = {norm(r["text"]) for r in train}
val_keys = {norm(r["text"]) for r in val}
leak = train_keys & val_keys
assert not leak, f"LEAK: {len(leak)} overlapping texts between train/val"
print(f"\n✓ leakage 없음 (train∩val = {len(leak)})")
