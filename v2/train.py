#%% [markdown]
# # Safety Classifier v2 — KcELECTRA-base fine-tune
# - 입력: data/v2/train.jsonl, val.jsonl
# - 라벨: non_abuse / abuse
# - 출력: v2/ckpt/best/ (model.safetensors + tokenizer + label_map.json)
#
# 이 모델은 service.* 라우팅을 하지 않는다. router는 final == "abuse"인지만
# 보고 차단하고, non_abuse는 OpenClaw로 전달한다.
#%%
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# --- 경로 자동 감지 (Kaggle / 로컬) ---
KAGGLE = Path("/kaggle/input").exists()
if KAGGLE:
    DATA = Path("/kaggle/input/mjuclaw-safety-v1")
    OUT = Path("/kaggle/working/ckpt")
else:
    HERE = Path(__file__).resolve().parent
    DATA = HERE.parent / "data" / "v2"
    OUT = HERE / "ckpt"
OUT.mkdir(parents=True, exist_ok=True)

# --- 장비 자동 감지 ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    PRECISION = "fp16"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    PRECISION = "bf16"
else:
    DEVICE = "cpu"
    PRECISION = "fp32"
print(f"device={DEVICE}, precision={PRECISION}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

MODEL_ID = os.environ.get("MODEL_ID", "beomi/KcELECTRA-base")
MAX_LEN = int(os.environ.get("MAX_LEN", "64"))
BATCH = int(os.environ.get("BATCH", "32"))
LR = float(os.environ.get("LR", "3e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
ABUSE_THRESHOLD = float(os.environ.get("ABUSE_THRESHOLD", "0.25"))
THRESHOLDS_TO_REPORT = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

LABELS = ["non_abuse", "abuse"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
N_LABELS = len(LABELS)
ABUSE_ID = LABEL2ID["abuse"]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_with_threshold(logits: np.ndarray, threshold: float) -> np.ndarray:
    probs = softmax_np(logits)
    return np.where(probs[:, ABUSE_ID] >= threshold, ABUSE_ID, LABEL2ID["non_abuse"])


def binary_report(logits: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = predict_with_threshold(logits, threshold)
    precision, recall, f1_binary, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=ABUSE_ID,
        zero_division=0,
    )
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "abuse_precision": float(precision),
        "abuse_recall": float(recall),
        "abuse_f1": float(f1_binary),
    }


#%% 데이터 로드 + label map
ds = load_dataset(
    "json",
    data_files={
        "train": str(DATA / "train.jsonl"),
        "val": str(DATA / "val.jsonl"),
    },
)
print(ds)

#%% 토크나이즈
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def preprocess(ex):
    enc = tok(ex["text"], truncation=True, max_length=MAX_LEN)
    enc["label"] = [LABEL2ID[label] for label in ex["intent"]]
    return enc


ds = ds.map(preprocess, batched=True, remove_columns=["text", "intent"])
print(ds)

#%% class weights (abuse 데이터가 적을 수 있으므로 보정)
y_train = np.array([ex["label"] for ex in ds["train"]])
cw = compute_class_weight("balanced", classes=np.arange(N_LABELS), y=y_train)
CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float32)
print("class weights:", {ID2LABEL[i]: round(float(cw[i]), 3) for i in range(N_LABELS)})

#%% 모델
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=N_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights,
        )
        return (loss, outputs) if return_outputs else loss


#%% 메트릭
def metrics(eval_pred):
    logits, labels = eval_pred
    report = binary_report(logits, labels, ABUSE_THRESHOLD)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["f1_macro"],
        "abuse_precision": report["abuse_precision"],
        "abuse_recall": report["abuse_recall"],
        "abuse_f1": report["abuse_f1"],
    }


#%% TrainingArguments
args = TrainingArguments(
    output_dir=str(OUT),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH * 2,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=(PRECISION == "fp16"),
    bf16=(PRECISION == "bf16"),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = WeightedTrainer(
    class_weights=CLASS_WEIGHTS,
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

#%% 학습
train_result = trainer.train()
print(train_result)

#%% 평가 + threshold sweep
preds_out = trainer.predict(ds["val"])
logits = preds_out.predictions
y_true = preds_out.label_ids
y_pred = predict_with_threshold(logits, ABUSE_THRESHOLD)

print("=== final eval ===")
for key, value in preds_out.metrics.items():
    print(f"  {key}: {value}")

print(f"\n=== classification report @ threshold={ABUSE_THRESHOLD:.2f} ===")
print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

print("\n=== threshold sweep ===")
threshold_reports = [
    binary_report(logits, y_true, threshold) for threshold in THRESHOLDS_TO_REPORT
]
for row in threshold_reports:
    print(
        "  t={threshold:.2f} acc={accuracy:.3f} macro_f1={f1_macro:.3f} "
        "abuse_p={abuse_precision:.3f} abuse_r={abuse_recall:.3f} "
        "abuse_f1={abuse_f1:.3f}".format(**row)
    )

cm = confusion_matrix(y_true, y_pred, labels=list(range(N_LABELS)))
np.save(OUT / "confusion.npy", cm)
with open(OUT / "threshold_sweep.json", "w", encoding="utf-8") as fh:
    json.dump(threshold_reports, fh, ensure_ascii=False, indent=2)

#%% 저장: 최종 모델 + label map
best_dir = OUT / "best"
trainer.save_model(str(best_dir))
tok.save_pretrained(str(best_dir))
with open(best_dir / "label_map.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "id2label": ID2LABEL,
            "label2id": LABEL2ID,
            "max_length": MAX_LEN,
            "task": "binary_safety",
            "abuse_threshold": ABUSE_THRESHOLD,
        },
        fh,
        ensure_ascii=False,
        indent=2,
    )

print(f"\n✓ saved to {best_dir}")

#%% 시각화 — confusion matrix
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_LABELS))
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_yticks(range(N_LABELS))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix @ threshold={ABUSE_THRESHOLD:.2f}")
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT / "confusion.png", dpi=120)
    print(f"✓ confusion plot: {OUT / 'confusion.png'}")
except Exception as exc:
    print(f"plot skipped: {exc}")
