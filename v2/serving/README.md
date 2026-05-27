# mjuclaw-safety-serving

KcELECTRA-base fine-tuned 모델 기반 한국어 abuse 판별 FastAPI 추론 서버.

mjuclaw-router가 사용자 메시지를 OpenClaw로 forward하기 전에 호출한다.

```text
final == "abuse"     -> router가 정형 거절
final == "non_abuse" -> router가 OpenClaw로 forward
```

## 모델

- 베이스: `beomi/KcELECTRA-base`
- 라벨: `non_abuse`, `abuse`
- 최종 결정: `p_abuse >= ABUSE_THRESHOLD`
- 기본 threshold: `0.25`

## API

### `GET /healthz`

인증 없이 호출 가능. 모델 로드 확인용.

```json
{
  "ok": true,
  "device": "cpu",
  "task": "binary_safety",
  "classes": 2,
  "labels": ["non_abuse", "abuse"],
  "abuse_threshold": 0.25
}
```

### `POST /classify`

헤더: `Authorization: Bearer <CLASSIFIER_AUTH_TOKEN>`  
env가 비어있으면 인증을 생략한다.

요청:

```json
{"text": "오늘 학식 뭐야?", "top_k": 3}
```

응답:

```json
{
  "final": "non_abuse",
  "overridden_to_abuse": false,
  "p_abuse": 0.012,
  "top": [
    {"label": "non_abuse", "score": 0.988},
    {"label": "abuse", "score": 0.012}
  ],
  "latency_ms": 47.2
}
```

`top_k=3`처럼 binary label 수보다 큰 값을 보내도 서버는 실제 label 수로 clamp한다.
이는 기존 router가 `top_k: 3`을 보내는 contract를 깨지 않기 위함이다.

`overridden_to_abuse`는 top1이 `abuse`가 아니지만 `p_abuse >= ABUSE_THRESHOLD`라서
최종 `final`이 `abuse`가 된 경우다.

## 환경 변수

| key | default | 설명 |
|---|---|---|
| `MODEL_DIR` | `/opt/intent-classifier/model` | 모델 가중치/토크나이저 디렉토리 |
| `ABUSE_THRESHOLD` | `0.25` | abuse 임계값. 낮출수록 recall↑ precision↓ |
| `CLASSIFIER_AUTH_TOKEN` | (none) | 비어있으면 인증 비활성화. dev 외 환경에서는 16자 이상 권장 |
| `LOG_LEVEL` | `INFO` | uvicorn/app 로깅 레벨 |
| `PORT` | `3200` | uvicorn bind port |

## Docker

빌드 컨텍스트는 `intent-classifier/` 루트다.

```bash
docker build -f serving/Dockerfile -t mjuclaw-safety-serving:dev .
docker run --rm -p 3200:3200 \
  -e CLASSIFIER_AUTH_TOKEN=$(openssl rand -hex 32) \
  mjuclaw-safety-serving:dev
```

기본 HF 모델 repo는 Dockerfile의 `MODEL_REPO_ID` build arg로 정한다.

```bash
docker build -f serving/Dockerfile -t mjuclaw-safety-serving:dev \
  --build-arg MODEL_REPO_ID=<org-or-user>/mjuclaw-safety-classifier \
  --build-arg MODEL_REVISION=main \
  .
```

mjuclaw-setup의 docker-compose에서는 `classifier` 서비스로 등록되며, host port
매핑 없이 compose 내부 네트워크(router -> classifier:3200)로만 접근한다.
