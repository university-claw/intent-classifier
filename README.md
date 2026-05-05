# mjuclaw-intent-classifier

mjuclaw Discord 봇의 한국어 의도 분류 + abuse 차단 모델.

- **Base**: `beomi/KcELECTRA-base` (110M)
- **Classes 15**: `service.lms.{unsubmitted,due_assignments,unread_notices,incomplete_online,digest}`,
  `service.ucheck.attendance`, `service.msi.{grades,schedule}`,
  `service.library.{search,my_loans}`, `service.news.{recent,search}`,
  `service.cafeteria.today`, `chat`, `abuse`
- **Metrics (val 665)**: Macro F1 0.9348, Weighted F1 0.9346, Acc 0.9353,
  abuse recall 0.7955 (top != abuse 이어도 `p(abuse) ≥ 0.25`면 override → 0.90)
- **HuggingFace**: [`kbsooo/mjuclaw-intent-classifier`](https://huggingface.co/kbsooo/mjuclaw-intent-classifier)

## 디렉토리

```
serving/          # FastAPI 추론 서버 + Dockerfile (mjuclaw-classifier 컨테이너)
v1/               # 학습 / 분할 / ONNX export / HF upload 스크립트
taxonomy.yaml     # 라벨 정의 (single source of truth)
model/            # gitignore — Dockerfile이 HF에서 자동 download
data/             # gitignore — synth-data 레포가 source
```

## 빌드 & 실행 (mjuclaw-setup이 자동 호출)

```bash
docker build -t mjuclaw-classifier -f serving/Dockerfile .
docker run --rm -p 3200:3200 mjuclaw-classifier
```

빌드 단계에서 `huggingface_hub.snapshot_download` 가 모델 가중치(약 420MB)를
HF Hub에서 받아 image 안에 포함시킨다.

## 추론 호출

```bash
curl -X POST http://localhost:3200/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"내일 학식 뭐야?"}'
```

응답 예:
```json
{
  "final": "service.cafeteria.today",
  "pAbuse": 0.0008,
  "overriddenToAbuse": false,
  "scores": { ... },
  "latencyMs": 86.4
}
```

## 재학습

새 인텐트 추가 시 `taxonomy.yaml` 을 single source of truth 로 유지(label id 순서).

```bash
cd v1
pip install -r requirements.txt
python split.py
python train.py
python upload_hf.py        # HF Hub push
```

학습 데이터는 [`synth-data`](../synth-data) 레포의 jsonl을 사용한다.
