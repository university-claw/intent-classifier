#%% [markdown]
# # Safety train/val split
# - synth-data/*.jsonl -> data/v2/{train,val}.jsonl
# - v2 canonical label은 abuse / non_abuse 뿐이다.
#   - abuse*.jsonl 또는 {"intent":"abuse"} -> abuse
#   - non_abuse*.jsonl, non-abuse*.jsonl 또는 {"intent":"non_abuse"} -> non_abuse
# - 85/15 비율, 최종 label 기준 stratified split
# - 같은 norm(text)는 한쪽에만 가도록 leakage 차단
#%%
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # intent-classifier/
RAW = ROOT.parent / "synth-data"  # mjuclaw/synth-data/
OUT = ROOT / "data" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_RATIO = 0.15

ABUSE = "abuse"
NON_ABUSE = "non_abuse"
TERMINAL_PUNCT = ".。!?！？"
ALLOW_LEGACY_NON_ABUSE = os.environ.get("ALLOW_LEGACY_NON_ABUSE", "").lower() in (
    "1",
    "true",
    "yes",
)


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def strip_terminal_punct(text: str) -> str:
    return text.strip().rstrip(TERMINAL_PUNCT).strip()


def text_variants(text: str) -> list[str]:
    """Balance terminal punctuation so the model cannot learn punctuation labels."""
    base = strip_terminal_punct(text)
    if not base:
        return [text.strip()]

    variants = [text.strip(), base, f"{base}.", f"{base}?"]
    unique = []
    seen = set()
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return unique


def label_for_file(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("abuse"):
        return ABUSE
    if name.startswith("non_abuse") or name.startswith("non-abuse"):
        return NON_ABUSE
    if ALLOW_LEGACY_NON_ABUSE and (
        name.startswith("chat") or name.startswith("service-")
    ):
        return NON_ABUSE
    raise ValueError(
        f"unknown synth-data file naming convention: {path.name} "
        "(expected abuse*.jsonl or non_abuse*.jsonl; "
        "set ALLOW_LEGACY_NON_ABUSE=1 only when converting old chat/service files)"
    )


def normalize_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in (ABUSE, NON_ABUSE):
        return normalized
    return None


def label_for_row(path: Path, row: dict) -> str:
    label = normalize_label(row.get("intent"))
    if label:
        return label
    return label_for_file(path)


def iter_rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path.name}:{line_no} missing non-empty text")
            label = label_for_row(path, row)
            for variant in text_variants(text):
                yield {
                    "text": variant,
                    "intent": label,
                    "source_intent": row.get("intent", path.stem),
                    "source_file": path.name,
                }


#%% 로드 + leakage 방지
by_label = defaultdict(list)
source_counts = Counter()
seen_keys = set()

files = sorted(RAW.glob("*.jsonl"))
if not files:
    raise SystemExit(f"no jsonl files found in {RAW}")

for path in files:
    for row in iter_rows(path):
        key = norm(row["text"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        label = row["intent"]
        by_label[label].append(row)
        source_counts[(label, row["source_intent"], row["source_file"])] += 1

print("source distribution:")
for (label, source_intent, source_file), count in sorted(source_counts.items()):
    print(f"  {label:10s} {str(source_intent):35s} {source_file:35s} {count}")

print("\nlabel distribution:")
for label in (NON_ABUSE, ABUSE):
    print(f"  {label:10s} {len(by_label[label])}")

missing = [label for label in (NON_ABUSE, ABUSE) if not by_label[label]]
if missing:
    raise SystemExit(f"missing required labels after split input load: {missing}")

#%% stratified split
random.seed(SEED)
train, val = [], []
for label, rows in by_label.items():
    rows = rows[:]
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val.extend(rows[:n_val])
    train.extend(rows[n_val:])

random.shuffle(train)
random.shuffle(val)


#%% 저장 (최종 학습 스키마: {text, intent})
def dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {"text": row["text"], "intent": row["intent"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


dump(OUT / "train.jsonl", train)
dump(OUT / "val.jsonl", val)

print(f"\ntrain: {len(train)}")
print(f"val:   {len(val)}")
print("\nval label distribution:")
for label, count in sorted(Counter(row["intent"] for row in val).items()):
    print(f"  {label:10s} {count}")

#%% leak sanity check
train_keys = {norm(row["text"]) for row in train}
val_keys = {norm(row["text"]) for row in val}
leak = train_keys & val_keys
assert not leak, f"LEAK: {len(leak)} overlapping texts between train/val"
print(f"\n✓ leakage 없음 (train∩val = {len(leak)})")
