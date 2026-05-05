#%% [markdown]
# # Intent Classifier v1 — KcELECTRA-base fine-tune
# - Kaggle(T4/P100) 또는 로컬 M4(MPS) 어디서든 실행
# - 입력: data/v1/train.jsonl, val.jsonl  (Kaggle에선 /kaggle/input/<dataset>/)
# - 출력: v1/ckpt/best/  (model.safetensors + tokenizer + label_map.json)
#
# ### [SOTA Alert]
# - KcELECTRA 계열은 댓글/구어체 한국어에 특화. Discriminator pretraining → 분류 task에서
#   같은 크기 BERT보다 파라미터 효율이 높다.
# - PyTorch 2.x의 SDPA(FlashAttention-2)는 encoder 짧은 seq에도 자동 활용.
#
# ### [Tensor Risk]
# - 클래스 불균형 (291~300)은 작지만 abuse recall에 영향. class_weight로 보정.
# - max_length 64로 잘라도 P99 커버 (데이터 기준). 128로 학습 후 추론시 64로 잘라도 정확도 유지.
# - MPS에서 fp16은 불안정 → bf16 사용. CUDA는 fp16 OK.
#%%
import os, json, random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --- 경로 자동 감지 (Kaggle / 로컬) ---
KAGGLE = Path("/kaggle/input").exists()
if KAGGLE:
    # Kaggle 데이터셋을 "mjuclaw-intent-v1" 이름으로 업로드했다고 가정
    DATA = Path("/kaggle/input/mjuclaw-intent-v1")
    OUT = Path("/kaggle/working/ckpt")
else:
    HERE = Path(__file__).resolve().parent
    DATA = HERE.parent / "data" / "v1"
    OUT = HERE / "ckpt"
OUT.mkdir(parents=True, exist_ok=True)

# --- 장비 자동 감지 ---
if torch.cuda.is_available():
    DEVICE = "cuda"; PRECISION = "fp16"
elif torch.backends.mps.is_available():
    DEVICE = "mps"; PRECISION = "bf16"
else:
    DEVICE = "cpu"; PRECISION = "fp32"
print(f"device={DEVICE}, precision={PRECISION}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

MODEL_ID = "beomi/KcELECTRA-base"
MAX_LEN = 64   # 실 분포상 p99 커버
BATCH = 32
LR = 3e-5      # KcELECTRA + 소규모 data에서 2e-5는 느림. 3e-5로 가속.
EPOCHS = 15    # 5 epoch은 under-trained. EarlyStopping이 알아서 끊음.

#%% 데이터 로드 + label map
ds = load_dataset("json", data_files={
    "train": str(DATA / "train.jsonl"),
    "val":   str(DATA / "val.jsonl"),
})
print(ds)

# taxonomy.yaml 순서 고정 (모델 배포 때 id↔name 안전)
LABELS = [
    "service.lms.unsubmitted", "service.lms.due_assignments",
    "service.lms.unread_notices", "service.lms.incomplete_online",
    "service.lms.digest", "service.ucheck.attendance",
    "service.msi.grades", "service.msi.schedule",
    "service.library.search", "service.library.my_loans",
    "service.news.recent", "service.news.search",
    "service.cafeteria.today", "chat", "abuse",
]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
N_LABELS = len(LABELS)

#%% 토크나이즈
tok = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess(ex):
    enc = tok(ex["text"], truncation=True, max_length=MAX_LEN)
    enc["label"] = [LABEL2ID[i] for i in ex["intent"]]
    return enc

ds = ds.map(preprocess, batched=True, remove_columns=["text", "intent"])
print(ds)

#%% class weights (불균형 보정)
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

# --- weighted loss trainer ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights.to(self.args.device)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights
        )
        return (loss, outputs) if return_outputs else loss

#%% 메트릭
def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # abuse recall 따로 추적 — 보안 관점에서 가장 중요
    abuse_id = LABEL2ID["abuse"]
    abuse_mask = labels == abuse_id
    abuse_recall = (preds[abuse_mask] == abuse_id).mean() if abuse_mask.any() else 0.0
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "abuse_recall": float(abuse_recall),
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
    processing_class=tok,  # transformers 4.46+ 은 tokenizer= → processing_class=
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

#%% 학습
train_result = trainer.train()
print(train_result)

#%% 평가 + 상세 리포트
# Note: trainer.evaluate()는 notebook callback 버그로 크래시. predict()로 바로 간다.
preds_out = trainer.predict(ds["val"])
print("=== final eval ===")
for k, v in preds_out.metrics.items(): print(f"  {k}: {v}")
y_pred = np.argmax(preds_out.predictions, axis=-1)
y_true = preds_out.label_ids
print("\n=== per-class report ===")
print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=list(range(N_LABELS)))
np.save(OUT / "confusion.npy", cm)

#%% 저장: 최종 모델 + label map
best_dir = OUT / "best"
trainer.save_model(str(best_dir))
tok.save_pretrained(str(best_dir))
with open(best_dir / "label_map.json", "w") as f:
    json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID, "max_length": MAX_LEN}, f, ensure_ascii=False, indent=2)

print(f"\n✓ saved to {best_dir}")

#%% 시각화 — confusion matrix
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_LABELS)); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_LABELS)); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title("Confusion Matrix (val)")
    # 셀에 숫자
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT / "confusion.png", dpi=120)
    print(f"✓ confusion plot: {OUT / 'confusion.png'}")
except Exception as e:
    print(f"plot skipped: {e}")
