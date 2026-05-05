#%% [markdown]
# # HuggingFace 업로드
# - 모델: kbsooo/mjuclaw-intent-classifier (public)
# - 데이터셋: kbsooo/mjuclaw-intent-dataset (public)
# 실행 전에 `hf auth login` 또는 HF_TOKEN 환경변수 설정 필요.
#%%
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ.get("HF_TOKEN"))
whoami = api.whoami()
USER = whoami["name"]
print(f"logged in as: {USER}")
assert USER == "kbsooo", f"unexpected user: {USER}"

MODEL_REPO   = f"{USER}/mjuclaw-intent-classifier"
DATASET_REPO = f"{USER}/mjuclaw-intent-dataset"

MODEL_DIR = Path("/Users/kbsoo/Codes/projects/mjuclaw/intent-classifier/model")
DATA_DIR  = Path("/tmp/hf-dataset-upload")

#%% 1) 데이터셋 업로드 (모델 카드가 dataset을 참조하므로 먼저)
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Initial v1 release: 4,474 Korean synthetic Discord-bot intent samples across 15 classes",
)
print(f"✓ dataset: https://huggingface.co/datasets/{DATASET_REPO}")

#%% 2) 모델 업로드
create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Initial v1 release: KcELECTRA-base fine-tuned, macro F1 0.935 on val (15-class intent)",
    ignore_patterns=["*.bin.tmp", "__pycache__/*"],
)
print(f"✓ model: https://huggingface.co/{MODEL_REPO}")
