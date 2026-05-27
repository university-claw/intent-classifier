# syntax=docker/dockerfile:1.6
# mjuclaw-safety-serving — KcELECTRA 기반 한국어 abuse 판별 FastAPI.
# build context = intent-classifier/v2 (serving/, taxonomy.yaml 만 필요).
# 모델 가중치는 빌드 시 HuggingFace Hub의 safety classifier repo에서
# huggingface_hub.snapshot_download 로 자동 download하여 image에 포함된다.
# 호스트에 model/ 디렉토리가 없어도 빌드 가능 (포터블).
# router 가 docker-compose 내부 네트워크로만 접근하도록 host port는 매핑하지 않는다.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/intent-classifier

# requirements 먼저 복사 → 레이어 캐시 활용
COPY serving/requirements.txt /tmp/requirements.txt
# CPU 전용 torch 휠 사용 (CUDA 없는 가벼운 컨테이너).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt

ARG MODEL_REPO_ID=nullhyeon/mjuclaw-safety-classifier
ARG MODEL_REVISION=main

# 모델 가중치는 HF Hub에서 다운로드 → image에 포함. private repo는 BuildKit secret
# `hf_token`으로 토큰을 전달한다. cache는 download 후 제거해서 이미지 비대화 방지.
RUN --mount=type=secret,id=hf_token \
    python -c "from pathlib import Path; \
from huggingface_hub import snapshot_download; \
token_path = Path('/run/secrets/hf_token'); \
token = token_path.read_text().strip() if token_path.exists() else None; \
snapshot_download('${MODEL_REPO_ID}', revision='${MODEL_REVISION}', token=token, \
local_dir='/opt/intent-classifier/model')" \
    && rm -rf /root/.cache/huggingface

# 추론 코드 + taxonomy 복사
COPY taxonomy.yaml /opt/intent-classifier/taxonomy.yaml
COPY serving/ /opt/intent-classifier/serving/

# 비루트 사용자
RUN groupadd --system --gid 1000 classifier \
    && useradd --system --uid 1000 --gid classifier --create-home classifier

USER classifier

ENV MODEL_DIR=/opt/intent-classifier/model \
    ABUSE_THRESHOLD=0.25 \
    LOG_LEVEL=INFO \
    PORT=3200

EXPOSE 3200

# 모델 로드는 시작 시 한 번만 (uvicorn worker=1 — 모델 메모리 중복 회피).
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "3200", "--workers", "1"]
