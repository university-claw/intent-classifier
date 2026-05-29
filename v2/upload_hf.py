#%% [markdown]
# # HuggingFace 업로드
# - 모델 기본값: <현재 로그인 계정>/mjuclaw-safety-classifier
# - 데이터셋 기본값: <현재 로그인 계정>/mjuclaw-safety-dataset
# - repo를 명시하려면 MODEL_REPO_ID / DATASET_REPO_ID 환경변수로 지정한다.
#   HF_MODEL_REPO / HF_DATASET_REPO도 하위 호환으로 지원한다.
# - private repo가 기본이다. 공개 업로드가 필요하면 HF_PRIVATE=false.
# - 실행 전에 `hf auth login` 또는 HF_TOKEN 환경변수 설정 필요.
#%%
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ.get("HF_TOKEN"))
whoami = api.whoami()
USER = whoami["name"]
print(f"logged in as: {USER}")

MODEL_REPO = (
    os.environ.get("MODEL_REPO_ID")
    or os.environ.get("HF_MODEL_REPO")
    or f"{USER}/mjuclaw-safety-classifier"
)
DATASET_REPO = (
    os.environ.get("DATASET_REPO_ID")
    or os.environ.get("HF_DATASET_REPO")
    or f"{USER}/mjuclaw-safety-dataset"
)
PRIVATE = os.environ.get("HF_PRIVATE", "true").lower() not in ("0", "false", "no")

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "ckpt" / "best"))
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE.parent / "data" / "v2"))
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", HERE.parent.parent / "synth-data"))

print(f"model repo:   {MODEL_REPO}")
print(f"dataset repo: {DATASET_REPO}")
print(f"private:      {PRIVATE}")
print(f"model dir:    {MODEL_DIR}")
print(f"data dir:     {DATA_DIR}")
print(f"raw data dir: {RAW_DATA_DIR}")

if not MODEL_DIR.exists():
    raise SystemExit(f"MODEL_DIR does not exist: {MODEL_DIR}")
if not DATA_DIR.exists():
    raise SystemExit(f"DATA_DIR does not exist: {DATA_DIR}")
if not RAW_DATA_DIR.exists():
    raise SystemExit(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

#%% 1) 데이터셋 업로드 (모델 카드가 dataset을 참조할 수 있으므로 먼저)
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=PRIVATE)
api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Release safety dataset: binary abuse/non_abuse Korean samples",
)
api.upload_folder(
    folder_path=str(RAW_DATA_DIR),
    path_in_repo="raw",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Release raw safety dataset sources",
)
print(f"✓ dataset: https://huggingface.co/datasets/{DATASET_REPO}")

#%% 2) 모델 업로드
create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=PRIVATE)
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Release safety classifier: KcELECTRA-base binary abuse gate",
    ignore_patterns=["*.bin.tmp", "__pycache__/*"],
)
print(f"✓ model: https://huggingface.co/{MODEL_REPO}")
