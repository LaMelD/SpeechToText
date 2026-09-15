# SpeechToText

회의 녹음 파일 하나를 넣으면 `[HH:MM:SS] 실명: 텍스트` 형식의 회의록을 만드는 CLI 파이프라인이다.
화자 분리(diarization)와 화자 식별(성문 매칭)은 로컬에서 돌리고, 텍스트 전사만 OpenAI `gpt-transcribe`에 보낸다.
클로바노트 음성기록을 대체하려고 만들었고, 혼자 CLI로 쓰는 용도다. LLM 요약·웹 UI·실시간 처리는 범위 밖이다.

같은 파이프라인을 diarization 백엔드만 바꿔 두 번 구현했다. 출력 파일 스키마가 같아서 결과를 그대로 비교할 수 있다.

| | `STT_LOCAL-pyannote/` | `STT_LOCAL-sherpa-onnx/` |
|---|---|---|
| diarization | pyannote.audio 4 (`pyannote/speaker-diarization-community-1`) | sherpa-onnx (pyannote segmentation-3.0 ONNX + 3D-Speaker CAM++) |
| 런타임 | torch, CPU 고정 | onnxruntime, torch 없음 (훨씬 가볍다) |
| 성문 임베딩 | WeSpeaker 256차원 | CAM++ 192차원 |
| 성문 매칭 임계값 기본값 | 0.65 | 0.60 |
| 필요한 인증 | HF 토큰 + 모델 약관 동의 | 없음 (모델 파일을 직접 내려받는다) |
| 추가 기능 | 없음 | 미등록 화자 묶기, `--participants`, `keywords.txt`, 클립 삭제 후 성문 재계산, 중복 등록 방지, 3초 미만 Unknown은 `불명`으로 접기 |
| 실측 | 회의 5건 331분: 화자 매칭 25/26, CER* 0.115~0.201 | 회의 3건 200분: 등록 화자 정확도 93.2% |

두 프로젝트의 성문 DB(`voiceprints/`)는 임베딩 모델이 달라 서로 호환되지 않는다. `model.txt` 가드가 섞임을 막으므로 각 프로젝트에서 따로 등록해야 한다.

## 디렉토리

```
SpeechToText/
├── README.md
├── .gitignore                     # 비밀·개인정보·모델·산출물은 커밋하지 않는다 (아래 "커밋하지 않는 것")
├── STT_LOCAL-pyannote/
│   ├── pyproject.toml             # pyannote.audio>=4, openai>=3
│   ├── stt_local/
│   │   ├── common.py              # 오디오 로드, diarize, WAV 슬라이싱, 성문 DB, 매칭, 턴 병합
│   │   ├── enroll.py              # 파이프라인 1: 성문 등록
│   │   └── transcribe.py          # 파이프라인 2: 전사
│   ├── test_stt_local.py
│   └── docs/superpowers/
│       ├── specs/                 # 설계 스펙 + 스파이크·엔드투엔드 검증 실측 (수치의 출처)
│       └── plans/                 # 구현 계획
└── STT_LOCAL-sherpa-onnx/
    ├── pyproject.toml             # sherpa-onnx>=1.13, numpy>=2, openai>=3
    ├── stt_local/                 # 위와 같은 구조. 튜닝 근거는 common.py 주석에 있다
    ├── test_stt_local.py
    └── keywords.txt               # STT 키워드. 레포에는 빈 파일만 두고 로컬에서 채운다 (transcribe가 자동으로 읽는다)
```

## 동작 원리

두 프로젝트 모두 다섯 단계다. 1~3은 로컬, 4만 API 호출이다.

1. **diarize** — 화자 구간과 화자별 centroid 임베딩을 얻는다. 결과는 `out/diarization.json`, `out/centroids.npy`에 캐시한다. 이 캐시는 속도가 아니라 정합성을 위한 것이다. 재실행 때 클러스터 라벨이 바뀌면 사용자가 고친 `speakers.json`이 어긋나기 때문이다.
2. **match** — 클러스터마다 성문 DB의 등록자와 코사인 유사도를 재서 가장 가까운 사람을 고른다(클러스터별 독립 argmax). 임계값 미만이면 `Unknown-1`, `Unknown-2`로 둔다. 한 사람이 여러 클러스터로 갈라지는 일이 흔해서 다대일을 허용하고, 헝가리안 1:1 배정은 쓰지 않는다. 결과는 `out/speakers.json`.
3. **merge** — 같은 화자의 인접 구간을 1초 이내면 합치고 0.3초 미만은 버린다. 다른 화자의 1초 이하 추임새는 건너뛰어 같은 화자를 2.5초 간격까지 묶는다(문맥이 길어져 정확도가 오른다). 병합 결과가 10분을 넘게 되는 병합은 하지 않는다(25MB 한도 대비).
4. **transcribe** — 턴마다 앞뒤 0.2초를 붙인 16kHz mono WAV를 메모리에서 잘라 `gpt-transcribe`에 보낸다. `languages=["ko"]`, `keywords`(사용자 지정 용어 + 매칭된 실명), 직전 12턴에서 뽑은 마지막 300자를 `prompt`로 넘긴다. 턴 목록을 8개 연속 구간으로 나눠 구간 안에서는 순차, 구간끼리는 병렬로 돌린다. 턴이 끝날 때마다 `out/turns.jsonl`에 한 줄씩 붙이므로 중단해도 재실행하면 이어서 한다. 실패한 턴은 `text: null`로 남기고 다음 실행에서 다시 시도한다.
5. **render** — 같은 이름의 인접 턴을 한 줄로 합쳐 `out/transcript.md`와 `out/transcript.json`을 쓴다. `speakers.json`에서 이름을 고친 뒤 같은 명령을 다시 실행하면 STT 없이 이 단계만 다시 한다.

### 성문 DB 형식

```
voiceprints/
  홍길동.npy   # float32, shape (세션 수, 임베딩 차원). 등록할 때마다 행이 추가된다
  김철수.npy
  model.txt    # 임베딩 모델 식별자. 다르면 실행을 거부한다
```

사람 점수는 그 사람 행들과의 코사인 유사도 최댓값이다. 회의실·온라인 등 녹음 조건별 벡터가 세션마다 쌓이는 구조다.

## 설치

공통: Python 3.12 이상, `ffmpeg`, 프로젝트 루트의 `.env`.

```
# .env (두 프로젝트 각각의 루트에 둔다. 커밋하지 않는다)
OPENAI_API_KEY=sk-...
```

### STT_LOCAL-pyannote

```bash
cd STT_LOCAL-pyannote
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
hf auth login        # 이후 https://hf.co/pyannote/speaker-diarization-community-1 에서 약관 동의
```

pyannote 4는 torchcodec으로 오디오를 읽어서 ffmpeg가 **shared 빌드**여야 한다. Windows는 `winget install BtbN.FFmpeg.LGPL.Shared.8.1`, macOS는 `brew install ffmpeg`.
현재 코드는 device를 CPU로 고정한다. GPU를 쓰려면 `common.py`의 `diarize()`에서 `"cpu"` 한 곳만 바꾼다.

### STT_LOCAL-sherpa-onnx

```bash
cd STT_LOCAL-sherpa-onnx
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
mkdir models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar -xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
mv 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx 3dspeaker_campplus_sv_zh_en_16k-common_advanced.onnx
```

HF 토큰은 필요 없다. ffmpeg는 디코딩에만 쓰므로 static 빌드도 된다.

## 사용법

### 1. 성문 등록 (`enroll`)

```bash
python -m stt_local.enroll 회의.m4a --out enroll_out/ [--num-speakers 5]
# enroll_out/SPEAKER_00/*.wav 를 듣고 폴더명을 실명으로 바꾼다. 섞인 클러스터는 그대로 둔다.
python -m stt_local.enroll --commit enroll_out/
```

클러스터마다 가장 긴 발화 6개를 WAV로 저장하고 centroid를 `centroid.npy`로 남긴다. `--commit`은 `SPEAKER_`로 시작하지 않는 폴더만 DB에 넣는다. 화자 수는 자동 추정이 기본이다(실측에서 지정보다 정확했다).

sherpa-onnx 판은 여기에 두 가지가 더 있다. 폴더 안에서 남의 목소리 클립을 지우고 commit하면 남은 클립으로 centroid를 다시 계산한다(클러스터 수를 늘려도 안 갈라지는 두 사람은 이 방법으로만 분리된다). 같은 폴더를 두 번 commit하면 `.committed` 마커가 막는다.

### 2. 전사 (`transcribe`)

```bash
# pyannote
python -m stt_local.transcribe 회의.m4a --out out/0902회의/ [--keywords "용어,제품명"] [--threshold 0.65]

# sherpa-onnx
python -m stt_local.transcribe 회의.m4a --out out/0902회의/ [--participants "홍길동,김철수"] [--keywords "용어"]
```

실행 전에 턴 수와 예상 비용을 출력한다. 끝나면 `out/` 안에 다음이 생긴다.

| 파일 | 내용 |
|---|---|
| `diarization.json`, `centroids.npy` | diarization 캐시. sherpa-onnx 판은 사용한 튜닝값(`config`)도 함께 남긴다 |
| `speakers.json` | `{"SPEAKER_00": {"name": "홍길동", "score": 0.91, "minutes": 14.2}, ...}`. 이름을 고쳐서 재실행할 수 있다. 1분 미만 클러스터는 신뢰도가 낮으니 `minutes`를 보고 의심한다 |
| `turns.jsonl` | 턴별 전사 결과. 재실행 시 이어서 하는 캐시 |
| `transcript.md` | `[00:12:34] 홍길동: 텍스트` 한 줄 한 턴 |
| `transcript.json` | transcript.md와 같은 행의 배열 (`start`, `end`, `name`, `text`) |

sherpa-onnx 판의 추가 입력:

- `keywords.txt` — 프로젝트 루트에 있으면 자동으로 읽는다. 한 줄에 하나, 쉼표 구분 가능, `#`은 주석. 매번 `--keywords`로 수십 개를 넘기는 대신 파일로 관리한다.
- `--participants` — 참석자 실명을 주면 성문 DB 후보를 그 사람들로 제한한다. 회의에 없는 사람이 임계값을 넘겨 끼어드는 오탐을 막는다. DB에 없는 이름은 경고만 하고 Unknown으로 처리한다.

## 튜닝 노브

**pyannote**: `--threshold`(기본 0.65. 같은 사람 세션 간 최소 0.74와 다른 사람 최대 0.58의 중간), `--min-speakers`/`--max-speakers`(기본 2~10).

**sherpa-onnx**: `--min-duration-on`(기본 1.5초)이 사실상 유일한 핵심 노브다. 낮추면 과분할, 높이면 0.4~0.9초 추임새를 버려서 merge의 추임새 흡수가 죽는다. 실회의 200분 재실측에서 1.0/1.5/2.0 모두 정확도 차이가 1% 안이라 1.5를 유지했다. 그 외 `--cluster-threshold`(0.8), `--threshold`(0.60. 0.65는 정확도 절벽 바로 위였다), `--threads`(물리 코어 수. 하이퍼스레딩까지 쓰면 오히려 느리다), `--clips`(centroid 계산에 쓸 화자별 긴 구간 수, 10). 같은 폴더의 `model.int8.onnx`는 쓰지 않는다. 12% 빨라지는 대신 발화 22%를 놓친다.

## 실측 요약

수치의 출처와 실험 과정은 `STT_LOCAL-pyannote/docs/superpowers/specs/`와 `STT_LOCAL-sherpa-onnx/stt_local/common.py` 주석에 있다. CER*은 클로바노트 출력 대비 불일치율(정렬 없는 근사치)이라 클로바 자체 오류도 포함한다.

**pyannote** (M3 Pro MPS 기준, 회의 5건 331분, 회의마다 그 회의를 뺀 성문 DB로 검증)

| 항목 | 값 |
|---|---|
| diarization RTF | 0.074 (50분 회의가 약 4분) |
| 화자 매칭 | 25/26 정답. Unknown 2건은 둘 다 옳은 판단(미등록자 거부 작동) |
| 전사 실패 | 0 / 2,972턴 |
| CER* | 0.115~0.201 (평균 0.148). 쉬운 회의는 클로바와 대등, 어려운 회의는 열세 |
| STT 비용 | 5건 합계 약 $1.43 (`gpt-transcribe` $0.0045/분) |

추임새 건너 병합과 직전 300자 prompt를 넣은 뒤 CER*이 평균 11% 내려갔고 호출 수는 절반이 됐다. 구간 첫 턴을 완성된 문맥으로 다시 전사하는 2패스는 비용만 10% 늘고 효과가 없어 되돌렸다.

**sherpa-onnx** (i7-9750H, 회의 3건 200분, 성문 DB 7명)

| 항목 | 값 |
|---|---|
| diarization RTF | 0.13 (6스레드. 1스레드 0.46, 12스레드 0.39) |
| 등록 화자 정확도 | 93.2% (`min_duration_on` 1.5, `threshold` 0.60) |
| 임계값 민감도 | 0.60 → 93.0%, 0.65 → 90.1%, 0.70 → 75.3% |

## 테스트

모델도 API도 부르지 않는다. numpy와 pytest만 있으면 된다.

```bash
cd STT_LOCAL-pyannote && .venv/bin/pytest
cd STT_LOCAL-sherpa-onnx && .venv/bin/pytest
```

## 커밋하지 않는 것

`.gitignore`가 막는 항목과 이유다. 이 레포를 clone하면 아래는 직접 만들어야 한다.

| 경로 | 이유 |
|---|---|
| `.env` | `OPENAI_API_KEY`, `HF_TOKEN` |
| `voiceprints/` | 실명이 붙은 성문 임베딩. 개인정보 |
| `enroll_out*/` | 실명 폴더 안의 화자 음성 클립 |
| `out*/` | 회의 전사문. 회의 내용 그 자체 |
| `spike/` | 실험 스크립트와 수집 데이터. 회의 텍스트가 섞여 있다 |
| `models/` | sherpa-onnx 모델 파일 34MB. 위 설치 절차로 내려받는다 |
| `*.wav`, `*.m4a`, `*.aac`, `*.mp3`, `*.npy` | 녹음 파일과 임베딩이 어디에 놓여도 새지 않게 |

설계 문서의 회의 참석자 실명은 `참석자A`~`참석자H`로 치환했다.
