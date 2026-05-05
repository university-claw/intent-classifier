# mjuclaw-intent-serving

KcELECTRA-base fine-tuned 모델(15-class) 기반 한국어 의도 분류 FastAPI 추론 서버.

mjuclaw-router가 사용자 메시지를 OpenClaw로 forward하기 전에 호출하여:
1. **abuse 차단** (현재 MVP-2 범위) — 분류 결과가 `abuse`이면 LLM 호출 0회로 정형 거절.
2. (향후) `service.*` 라우팅 분기 — mju-cli/mju-news 직접 호출로 대체.

## 모델

- 베이스: `beomi/KcELECTRA-base`
- 라벨: 15개 (taxonomy v1 — `taxonomy.yaml` 단일 source of truth)
- 학습 메트릭: macro F1 0.9348, abuse recall 0.7955 (+ threshold 보정 0.25)

## API

### `GET /healthz`
인증 없이 호출 가능. 모델 로드 확인용.
```json
{"ok": true, "device": "cpu", "classes": 15, "abuse_threshold": 0.25}
```

### `POST /classify`
헤더: `Authorization: Bearer <CLASSIFIER_AUTH_TOKEN>` (env가 비어있으면 인증 생략).

요청:
```json
{"text": "오늘 학식 뭐야?", "top_k": 3}
```

응답:
```json
{
  "final": "service.cafeteria.today",
  "overridden_to_abuse": false,
  "p_abuse": 0.012,
  "top": [
    {"label": "service.cafeteria.today", "score": 0.973},
    {"label": "chat", "score": 0.018},
    {"label": "service.news.recent", "score": 0.005}
  ],
  "latency_ms": 47.2
}
```

`overridden_to_abuse`는 top1이 abuse가 아니더라도 `p(abuse) >= ABUSE_THRESHOLD`이면
final을 abuse로 덮어씌운 경우다 (recall 보강).

## 환경 변수

| key | default | 설명 |
|---|---|---|
| `MODEL_DIR` | `/opt/intent-classifier/model` | 모델 가중치/토크나이저 디렉토리 |
| `ABUSE_THRESHOLD` | `0.25` | abuse override 임계값 (낮출수록 recall↑ precision↓) |
| `CLASSIFIER_AUTH_TOKEN` | (none) | 비어있으면 인증 비활성화 (dev only). 16자 이상 권장 |
| `LOG_LEVEL` | `INFO` | uvicorn/app 로깅 레벨 |
| `PORT` | `3200` | uvicorn bind port |

## Docker

빌드 컨텍스트는 `intent-classifier/` 루트 (model/, serving/, taxonomy.yaml 포함).

```bash
docker build -f serving/Dockerfile -t mjuclaw-intent-serving:dev .
docker run --rm -p 3200:3200 \
  -e CLASSIFIER_AUTH_TOKEN=$(openssl rand -hex 32) \
  mjuclaw-intent-serving:dev
```

mjuclaw-setup의 docker-compose에서 `classifier` 서비스로 등록되며, host port 매핑은
하지 않고 compose 내부 네트워크 (router → classifier:3200)로만 접근한다.
