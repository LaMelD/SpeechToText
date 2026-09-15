# STT_LOCAL 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 회의 녹음을 넣으면 pyannote로 화자를 나누고, 성문 DB로 실명을 붙이고, gpt-transcribe로 전사해 `[HH:MM:SS] 실명: 텍스트` 전사문을 내는 CLI 두 개(enroll, transcribe)를 만든다.

**Architecture:** 패키지 `stt_local` 안에 파일 3개. `common.py`는 오디오 로드, diarization 래퍼, WAV 슬라이싱, 성문 DB, 매칭, 턴 병합을 담는다. `enroll.py`는 클립 추출과 `--commit`, `transcribe.py`는 diarize → match → merge → 비동기 STT → render 흐름이다. 모델 호출이 없는 순수 로직은 `test_stt_local.py` 하나로 검증하고, 모델이 필요한 부분은 STT_BMT 녹음으로 수동 검증한다.

**Tech Stack:** Python 3.12, pyannote.audio 4.0.7 (torch 2.13, MPS), openai SDK 3.7 (`gpt-transcribe`), numpy, 시스템 ffmpeg. 테스트는 pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-stt-pipeline-design.md`

## Global Constraints

- 의존성은 `pyannote.audio>=4`, `openai` 두 개뿐. numpy는 pyannote가 끌고 온다. 그 외 추가 금지. (pytest는 dev 전용)
- Python 3.12. 가상환경은 프로젝트 루트 `.venv`에 이미 있다. 모든 명령은 `.venv/bin/python`으로 실행한다.
- 파일은 `stt_local/common.py`, `stt_local/enroll.py`, `stt_local/transcribe.py`, `test_stt_local.py` 네 개. 더 만들지 않는다.
- 오디오는 항상 16kHz mono float32. WAV 인코딩은 stdlib `wave`. ffmpeg subprocess 호출 금지 (pyannote의 `Audio`가 torchcodec으로 처리).
- 성문 DB는 `voiceprints/<이름>.npy` (float32, shape (세션 수, 256), 행마다 L2 정규화) + `voiceprints/model.txt`.
- 모델 ID: `pyannote/speaker-diarization-community-1`. STT 모델: `gpt-transcribe`, `languages=["ko"]`.
- 매칭은 클러스터별 독립 argmax, 사람 점수는 그 사람 행들과의 코사인 최댓값, 임계값 기본 0.65, 미달은 `Unknown-N`.
- 턴 병합: 같은 화자 간격 1.0초 이내 병합, 0.3초 미만 제거, 합쳐서 600초 초과면 병합 안 함.
- 슬라이싱 패딩 0.2초. STT 병렬 8. 실패 턴은 `text: null`로 기록하고 계속.
- 주석과 출력 메시지는 한국어. `ponytail:` 주석으로 의도적 단순화를 표시한다.
- 커밋 메시지 끝에 `Claude-Session: https://claude.ai/code/session_<id>` 줄을 붙인다.
- torch와 pyannote는 무거우므로 `common.py`에서 함수 안에서 지연 import한다. 테스트가 3초 안에 돌아야 한다.

---

### Task 1: 프로젝트 골격, 타임스탬프 포맷, WAV 슬라이싱

**Files:**
- Create: `pyproject.toml`
- Create: `stt_local/__init__.py` (빈 파일)
- Create: `stt_local/common.py`
- Create: `test_stt_local.py`

**Interfaces:**
- Produces: `SAMPLE_RATE: int = 16000`, `MODEL_ID: str`, `DB_DIR: Path = Path("voiceprints")`
- Produces: `fmt_ts(sec: float) -> str` "HH:MM:SS"
- Produces: `slice_wav(wave: np.ndarray, start: float, end: float, pad: float = 0.2) -> bytes` 16kHz mono 16bit WAV 바이트. `wave`는 shape (N,) float32 [-1, 1].

- [ ] **Step 1: pyproject.toml과 빈 패키지 만들기**

```toml
[project]
name = "stt-local"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pyannote.audio>=4", "openai>=3"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools]
packages = ["stt_local"]
```

```bash
mkdir -p stt_local && : > stt_local/__init__.py
.venv/bin/pip install -q pytest
```

- [ ] **Step 2: 실패하는 테스트 쓰기**

`test_stt_local.py`:

```python
import io
import wave as wave_mod

import numpy as np

from stt_local.common import SAMPLE_RATE, fmt_ts, slice_wav


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00"
    assert fmt_ts(754.9) == "00:12:34"
    assert fmt_ts(3725) == "01:02:05"


def test_slice_wav_pads_and_clamps():
    wave = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)  # 10초 무음
    data = slice_wav(wave, start=1.0, end=2.0, pad=0.2)
    with wave_mod.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == int(1.4 * SAMPLE_RATE)
    # 파일 시작에서 패딩이 잘림
    data = slice_wav(wave, start=0.1, end=1.0, pad=0.2)
    with wave_mod.open(io.BytesIO(data)) as w:
        assert w.getnframes() == int(1.2 * SAMPLE_RATE)
    # 파일 끝에서 패딩이 잘림
    data = slice_wav(wave, start=9.5, end=10.0, pad=0.2)
    with wave_mod.open(io.BytesIO(data)) as w:
        assert w.getnframes() == int(0.7 * SAMPLE_RATE)


def test_slice_wav_clips_amplitude():
    wave = np.full(SAMPLE_RATE, 2.0, dtype=np.float32)  # 범위 밖 값
    data = slice_wav(wave, 0, 1, pad=0)
    with wave_mod.open(io.BytesIO(data)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert pcm.max() == 32767
```

- [ ] **Step 3: 실패 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'stt_local.common'`

- [ ] **Step 4: 최소 구현**

`stt_local/common.py`:

```python
"""오디오 로드, diarization, WAV 슬라이싱, 성문 DB, 매칭, 턴 병합."""
import io
import wave as wave_mod
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL_ID = "pyannote/speaker-diarization-community-1"
DB_DIR = Path("voiceprints")


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def slice_wav(wave: np.ndarray, start: float, end: float, pad: float = 0.2) -> bytes:
    """wave[start-pad : end+pad]를 16kHz mono 16bit WAV 바이트로. 파일 경계에서 클램프."""
    a = max(0, round((start - pad) * SAMPLE_RATE))  # round: 9.5-0.2 같은 값의 부동소수 오차 방지
    b = min(len(wave), round((end + pad) * SAMPLE_RATE))
    pcm = (np.clip(wave[a:b], -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave_mod.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()
```

- [ ] **Step 5: 통과 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: 3 passed

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml stt_local/__init__.py stt_local/common.py test_stt_local.py
git commit -m "feat: 프로젝트 골격, fmt_ts, slice_wav

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 2: 턴 병합

**Files:**
- Modify: `stt_local/common.py`
- Modify: `test_stt_local.py`

**Interfaces:**
- Consumes: 없음
- Produces: `merge_turns(segs: list[dict], gap: float = 1.0, min_dur: float = 0.3, max_dur: float = 600.0) -> list[dict]`. 입력과 출력 모두 `{"start": float, "end": float, "speaker": str}`. 입력은 start 순 정렬을 가정하지 않는다(함수가 정렬). 출력은 start 순.

- [ ] **Step 1: 실패하는 테스트 쓰기**

`test_stt_local.py`에 추가:

```python
from stt_local.common import merge_turns


def seg(start, end, spk="A"):
    return {"start": start, "end": end, "speaker": spk}


def test_merge_same_speaker_within_gap():
    out = merge_turns([seg(0, 1), seg(1.5, 3), seg(4.5, 5)])
    assert out == [seg(0, 3), seg(4.5, 5)]  # 0.5초 간격은 합치고 1.5초는 안 합침


def test_merge_does_not_cross_speakers():
    out = merge_turns([seg(0, 1, "A"), seg(1.2, 2, "B"), seg(2.1, 3, "A")])
    assert out == [seg(0, 1, "A"), seg(1.2, 2, "B"), seg(2.1, 3, "A")]


def test_merge_drops_short_turns_after_merging():
    # 0.2초짜리 둘이 합쳐져 0.4초가 되면 살아남는다
    out = merge_turns([seg(0, 0.2), seg(0.3, 0.5), seg(10, 10.2)])
    assert out == [seg(0, 0.5)]


def test_merge_respects_max_dur():
    out = merge_turns([seg(0, 400), seg(400.5, 700)], max_dur=600)
    assert out == [seg(0, 400), seg(400.5, 700)]


def test_merge_sorts_input():
    out = merge_turns([seg(5, 6), seg(0, 1)])
    assert out == [seg(0, 1), seg(5, 6)]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v -k merge`
Expected: FAIL, `ImportError: cannot import name 'merge_turns'`

- [ ] **Step 3: 최소 구현**

`stt_local/common.py`에 추가:

```python
def merge_turns(segs: list[dict], gap: float = 1.0, min_dur: float = 0.3,
                max_dur: float = 600.0) -> list[dict]:
    """같은 화자의 인접 구간을 gap 이내면 합치고, 합친 뒤 min_dur 미만은 버린다.
    ponytail: max_dur 초과가 되는 병합은 안 한다 (가장 긴 공백에서 자르는 대신). 실측 최장 턴 1.3분."""
    turns: list[dict] = []
    for s in sorted(segs, key=lambda s: s["start"]):
        last = turns[-1] if turns else None
        if (last and last["speaker"] == s["speaker"]
                and s["start"] - last["end"] <= gap
                and s["end"] - last["start"] <= max_dur):
            last["end"] = max(last["end"], s["end"])
        else:
            turns.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"]})
    return [t for t in turns if t["end"] - t["start"] >= min_dur]
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
git add stt_local/common.py test_stt_local.py
git commit -m "feat: merge_turns

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 3: 성문 DB

**Files:**
- Modify: `stt_local/common.py`
- Modify: `test_stt_local.py`

**Interfaces:**
- Consumes: `MODEL_ID`, `DB_DIR`
- Produces: `db_add(db_dir: Path, name: str, vec: np.ndarray) -> None` (vec shape (256,), 정규화 후 행 추가, 처음이면 model.txt 생성)
- Produces: `db_load(db_dir: Path) -> dict[str, np.ndarray]` (이름 → (k, 256) 정규화 행렬. 디렉토리 없으면 빈 dict. 이름은 NFC)
- Produces: `check_model_tag(db_dir: Path) -> None` (model.txt가 있고 MODEL_ID와 다르면 `SystemExit`)

- [ ] **Step 1: 실패하는 테스트 쓰기**

`test_stt_local.py`에 추가:

```python
import pytest

from stt_local.common import MODEL_ID, check_model_tag, db_add, db_load


def test_db_add_and_load_accumulate_rows(tmp_path):
    db_add(tmp_path, "홍길동", np.array([3.0, 4.0]))
    db_add(tmp_path, "홍길동", np.array([0.0, 2.0]))
    db_add(tmp_path, "김철수", np.array([1.0, 0.0]))
    db = db_load(tmp_path)
    assert set(db) == {"홍길동", "김철수"}
    assert db["홍길동"].shape == (2, 2)
    np.testing.assert_allclose(db["홍길동"][0], [0.6, 0.8])
    np.testing.assert_allclose(db["홍길동"][1], [0.0, 1.0])
    assert db["홍길동"].dtype == np.float32
    assert (tmp_path / "model.txt").read_text().strip() == MODEL_ID


def test_db_load_missing_dir_is_empty(tmp_path):
    assert db_load(tmp_path / "없음") == {}


def test_db_rejects_other_model(tmp_path):
    (tmp_path / "model.txt").write_text("other-model\n")
    with pytest.raises(SystemExit):
        check_model_tag(tmp_path)
    with pytest.raises(SystemExit):
        db_add(tmp_path, "홍길동", np.array([1.0, 0.0]))


def test_db_add_rejects_zero_vector(tmp_path):
    with pytest.raises(ValueError):
        db_add(tmp_path, "홍길동", np.zeros(2))
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v -k db`
Expected: FAIL, `ImportError: cannot import name 'check_model_tag'`

- [ ] **Step 3: 최소 구현**

`stt_local/common.py` 상단 import에 `import sys`, `import unicodedata` 추가하고, 아래를 추가:

```python
def check_model_tag(db_dir: Path) -> None:
    tag = db_dir / "model.txt"
    if tag.exists() and tag.read_text().strip() != MODEL_ID:
        sys.exit(f"성문 DB 모델({tag.read_text().strip()})이 현재 모델({MODEL_ID})과 다릅니다. "
                 f"{db_dir}/를 지우고 재등록하세요.")


def db_add(db_dir: Path, name: str, vec: np.ndarray) -> None:
    """정규화한 vec을 db_dir/<name>.npy에 행으로 추가한다."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError(f"{name}: 영벡터는 등록할 수 없습니다")
    db_dir.mkdir(parents=True, exist_ok=True)
    check_model_tag(db_dir)
    tag = db_dir / "model.txt"
    if not tag.exists():
        tag.write_text(MODEL_ID + "\n")
    row = (vec / norm).astype(np.float32)[None]
    f = db_dir / f"{name}.npy"
    np.save(f, np.concatenate([np.load(f), row]) if f.exists() else row)


def db_load(db_dir: Path) -> dict[str, np.ndarray]:
    """이름 → (k, dim) 정규화 행렬. 디렉토리가 없으면 빈 dict."""
    if not db_dir.exists():
        return {}
    check_model_tag(db_dir)
    db = {}
    for f in sorted(db_dir.glob("*.npy")):
        rows = np.load(f).astype(np.float32)
        # macOS 파일명은 NFD라 이름을 NFC로 되돌린다
        db[unicodedata.normalize("NFC", f.stem)] = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    return db
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: 12 passed

- [ ] **Step 5: 커밋**

```bash
git add stt_local/common.py test_stt_local.py
git commit -m "feat: 성문 DB add/load/model tag

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 4: 매칭

**Files:**
- Modify: `stt_local/common.py`
- Modify: `test_stt_local.py`

**Interfaces:**
- Consumes: `db_load`의 반환 형식 (이름 → 정규화 (k, dim))
- Produces: `match(centroids: dict[str, np.ndarray], db: dict[str, np.ndarray], threshold: float = 0.65) -> dict[str, dict]`. 반환은 `{label: {"name": str, "score": float}}`. score는 소수 3자리. DB가 비었거나 centroid가 영벡터면 score 0.0에 Unknown.

- [ ] **Step 1: 실패하는 테스트 쓰기**

`test_stt_local.py`에 추가:

```python
from stt_local.common import match


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_match_argmax_allows_many_to_one():
    db = {"홍길동": np.stack([unit(1, 0)]), "김철수": np.stack([unit(0, 1)])}
    cent = {"SPEAKER_00": unit(1, 0.1), "SPEAKER_01": unit(1, -0.1), "SPEAKER_02": unit(0, 1)}
    out = match(cent, db, threshold=0.65)
    assert out["SPEAKER_00"]["name"] == "홍길동"
    assert out["SPEAKER_01"]["name"] == "홍길동"  # 같은 사람에게 둘 다 붙음
    assert out["SPEAKER_02"]["name"] == "김철수"
    assert out["SPEAKER_00"]["score"] > 0.99


def test_match_uses_max_over_sessions():
    db = {"홍길동": np.stack([unit(0, 1), unit(1, 0)])}
    out = match({"S": unit(1, 0)}, db)
    assert out["S"] == {"name": "홍길동", "score": 1.0}


def test_match_below_threshold_is_unknown_numbered():
    db = {"홍길동": np.stack([unit(1, 0)])}
    cent = {"SPEAKER_00": unit(0, 1), "SPEAKER_01": unit(1, 0), "SPEAKER_02": unit(0, -1)}
    out = match(cent, db, threshold=0.65)
    assert out["SPEAKER_00"]["name"] == "Unknown-1"
    assert out["SPEAKER_01"]["name"] == "홍길동"
    assert out["SPEAKER_02"]["name"] == "Unknown-2"


def test_match_empty_db_or_zero_centroid():
    assert match({"S": unit(1, 0)}, {}) == {"S": {"name": "Unknown-1", "score": 0.0}}
    db = {"홍길동": np.stack([unit(1, 0)])}
    assert match({"S": np.zeros(2, dtype=np.float32)}, db) == {"S": {"name": "Unknown-1", "score": 0.0}}
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v -k match`
Expected: FAIL, `ImportError: cannot import name 'match'`

- [ ] **Step 3: 최소 구현**

`stt_local/common.py`에 추가:

```python
def match(centroids: dict[str, np.ndarray], db: dict[str, np.ndarray],
          threshold: float = 0.65) -> dict[str, dict]:
    """클러스터별 독립 argmax. 사람 점수는 그 사람 행들과의 코사인 최댓값.
    다대일 허용: 한 사람이 여러 클러스터로 갈라져도 전부 같은 이름."""
    out, unknown = {}, 0
    for label, c in centroids.items():
        norm = np.linalg.norm(c)
        best, score = None, 0.0
        if norm > 0 and db:
            c = c / norm
            best, score = max(((name, float((rows @ c).max())) for name, rows in db.items()),
                              key=lambda kv: kv[1])
        if best is None or score < threshold:
            unknown += 1
            best = f"Unknown-{unknown}"
        out[label] = {"name": best, "score": round(score, 3)}
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: 16 passed

- [ ] **Step 5: 커밋**

```bash
git add stt_local/common.py test_stt_local.py
git commit -m "feat: match (독립 argmax + 임계값)

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 5: 환경 검사, 오디오 로드, diarization 래퍼

모델이 필요하므로 단위 테스트 대신 2분짜리 오디오로 수동 확인한다.

**Files:**
- Modify: `stt_local/common.py`

**Interfaces:**
- Produces: `load_env(path: Path = Path(".env")) -> None` (`.env`의 `KEY=VALUE`를 os.environ에 setdefault)
- Produces: `check_prereqs(need_openai: bool) -> None` (ffmpeg, HF 토큰, OPENAI_API_KEY 검사. 없으면 한 줄 메시지로 `sys.exit`)
- Produces: `load_audio(path: str) -> np.ndarray` (shape (N,) float32 16kHz mono)
- Produces: `diarize(wave: np.ndarray, num_speakers: int | None = None, min_speakers: int = 2, max_speakers: int = 10) -> tuple[list[dict], dict[str, np.ndarray]]`. 첫 번째는 exclusive 구간 `{"start","end","speaker"}` start 순 (소수 3자리), 두 번째는 `{label: centroid (256,)}` (labels 순서).

- [ ] **Step 1: 구현**

`stt_local/common.py` 상단 import에 `import os`, `import shutil` 추가하고, 아래를 추가:

```python
def load_env(path: Path = Path(".env")) -> None:
    """ponytail: python-dotenv 대신 KEY=VALUE 줄만 읽는다."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def check_prereqs(need_openai: bool) -> None:
    load_env()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg가 없습니다: brew install ffmpeg")
    from huggingface_hub import get_token
    if not get_token():
        sys.exit("HF 토큰이 없습니다: hf auth login 후 "
                 "https://hf.co/pyannote/speaker-diarization-community-1 에서 약관 동의")
    if need_openai and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없습니다: 프로젝트 루트 .env에 OPENAI_API_KEY=... 추가")


def load_audio(path: str) -> np.ndarray:
    from pyannote.audio import Audio  # 지연 import: torch가 무겁다
    waveform, _ = Audio(sample_rate=SAMPLE_RATE, mono="downmix")(path)
    return waveform.numpy()[0].astype(np.float32)


def diarize(wave: np.ndarray, num_speakers: int | None = None,
            min_speakers: int = 2, max_speakers: int = 10) -> tuple[list[dict], dict[str, np.ndarray]]:
    """exclusive 구간과 화자별 centroid. MPS 우선, 실패하면 CPU."""
    import torch
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(MODEL_ID)
    file = {"waveform": torch.from_numpy(wave)[None], "sample_rate": SAMPLE_RATE}
    kw = {"num_speakers": num_speakers} if num_speakers else \
         {"min_speakers": min_speakers, "max_speakers": max_speakers}
    devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    for i, dev in enumerate(devices):
        try:
            pipe.to(torch.device(dev))
            out = pipe(file, **kw)
            break
        except Exception as e:  # ponytail: MPS 미지원 연산이면 CPU로 한 번 더
            if i == len(devices) - 1:
                raise
            print(f"{dev} 실패({type(e).__name__}: {e}), cpu로 재시도", file=sys.stderr)
    segs = [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": spk}
            for s, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
    segs.sort(key=lambda s: s["start"])
    labels = out.speaker_diarization.labels()
    cent = {label: out.speaker_embeddings[i] for i, label in enumerate(labels)}
    return segs, cent
```

- [ ] **Step 2: 기존 테스트가 여전히 빠른지 확인**

Run: `time .venv/bin/python -m pytest test_stt_local.py -q`
Expected: 16 passed, 3초 이내 (torch가 import되지 않아야 함)

- [ ] **Step 3: 2분 오디오로 수동 확인**

```bash
.venv/bin/python -c "
from stt_local.common import check_prereqs, load_audio, diarize, SAMPLE_RATE
check_prereqs(need_openai=False)
w = load_audio('../STT_BMT/data/audio-2.aac')[:SAMPLE_RATE*120]
segs, cent = diarize(w)
print('segs', len(segs), 'labels', list(cent), 'dim', next(iter(cent.values())).shape)
print(segs[:3])
" 2>&1 | grep -vE "Warning|warn"
```
Expected: `segs` 수십 개, labels 2~4개, dim (256,), 구간이 start 순.

- [ ] **Step 4: 커밋**

```bash
git add stt_local/common.py
git commit -m "feat: load_env, check_prereqs, load_audio, diarize

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 6: enroll CLI

**Files:**
- Create: `stt_local/enroll.py`

**Interfaces:**
- Consumes: `check_prereqs`, `load_audio`, `diarize`, `slice_wav`, `db_add`, `DB_DIR` (common.py)
- Produces: `python -m stt_local.enroll AUDIO --out DIR [--num-speakers N] [--clips 6]` 와 `python -m stt_local.enroll --commit DIR [--db voiceprints]`

- [ ] **Step 1: 구현**

`stt_local/enroll.py`:

```python
"""파이프라인 1: 화자 클립 추출 → 사람이 폴더명을 실명으로 변경 → --commit으로 성문 DB에 추가.

    python -m stt_local.enroll 회의.m4a --out enroll_out/ [--num-speakers 5]
    python -m stt_local.enroll --commit enroll_out/
"""
import argparse
import unicodedata
from pathlib import Path

import numpy as np

from stt_local.common import DB_DIR, check_prereqs, db_add, diarize, load_audio, slice_wav


def extract(audio: str, out: Path, num_speakers: int | None, clips: int) -> None:
    wave = load_audio(audio)
    segs, cent = diarize(wave, num_speakers=num_speakers)
    for label, c in cent.items():
        mine = [s for s in segs if s["speaker"] == label]
        total = sum(s["end"] - s["start"] for s in mine) / 60
        if np.linalg.norm(c) == 0 or not mine:
            print(f"{label}: 발화 없음, 건너뜀")
            continue
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "centroid.npy", c)
        longest = sorted(mine, key=lambda s: s["end"] - s["start"], reverse=True)[:clips]
        for k, s in enumerate(sorted(longest, key=lambda s: s["start"]), 1):
            name = f"{k}_{int(s['start']):04d}-{int(s['end']):04d}.wav"
            (d / name).write_bytes(slice_wav(wave, s["start"], s["end"], pad=0))
        print(f"{label}: 발화 {total:.1f}분, 클립 {len(longest)}개 → {d}/")
    print(f"\n클립을 듣고 {out}/ 안의 폴더명을 실명으로 바꾼 뒤:\n"
          f"  python -m stt_local.enroll --commit {out}")


def commit(src: Path, db: Path) -> None:
    n = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        name = unicodedata.normalize("NFC", d.name)
        if name.startswith("SPEAKER_") or not (d / "centroid.npy").exists():
            continue
        db_add(db, name, np.load(d / "centroid.npy"))
        n += 1
        print(f"추가: {name}")
    print(f"{n}명을 {db}/에 추가했습니다")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="회의 오디오 파일")
    ap.add_argument("--out", type=Path, help="클립 출력 폴더")
    ap.add_argument("--num-speakers", type=int, default=None, help="화자 수 강제 (기본 자동)")
    ap.add_argument("--clips", type=int, default=6, help="클러스터당 클립 수")
    ap.add_argument("--commit", type=Path, metavar="DIR", help="이름 바꾼 폴더를 DB에 반영")
    ap.add_argument("--db", type=Path, default=DB_DIR)
    a = ap.parse_args()
    if a.commit:
        commit(a.commit, a.db)
    elif a.audio and a.out:
        check_prereqs(need_openai=False)
        extract(a.audio, a.out, a.num_speakers, a.clips)
    else:
        ap.error("AUDIO --out DIR 또는 --commit DIR 중 하나가 필요합니다")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 수동 확인 (audio-2, 약 3분 소요)**

```bash
.venv/bin/python -m stt_local.enroll ../STT_BMT/data/audio-2.aac --out enroll_out/ 2>&1 | grep -vE "Warning|warn"
ls enroll_out/*/
```
Expected: `enroll_out/SPEAKER_00/` ~ `SPEAKER_05/` 정도, 각 폴더에 `centroid.npy`와 wav 6개, 파일명 `1_0132-0141.wav` 형식. 출력에 클러스터별 발화 분.

- [ ] **Step 3: commit 경로 확인 (임시 DB로)**

```bash
mv enroll_out/SPEAKER_00 enroll_out/테스트사람
.venv/bin/python -m stt_local.enroll --commit enroll_out/ --db spike/vp_test
.venv/bin/python -c "import numpy as np; print(np.load('spike/vp_test/테스트사람.npy').shape)"
cat spike/vp_test/model.txt
rm -rf spike/vp_test
```
Expected: `추가: 테스트사람`, `1명을 ... 추가했습니다`, shape `(1, 256)`, model.txt에 MODEL_ID. SPEAKER_ 폴더는 건너뜀.

- [ ] **Step 4: 커밋**

```bash
git add stt_local/enroll.py
git commit -m "feat: enroll CLI (클립 추출, --commit)

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 7: transcribe CLI

**Files:**
- Create: `stt_local/transcribe.py`
- Modify: `test_stt_local.py`

**Interfaces:**
- Consumes: `check_prereqs`, `load_audio`, `diarize`, `slice_wav`, `db_load`, `match`, `merge_turns`, `fmt_ts`, `DB_DIR` (common.py)
- Produces: `render(turns: list[dict], texts: dict[int, str], speakers: dict[str, dict]) -> tuple[str, list[dict]]` (순수 함수, 테스트 대상). `turns[i]`의 텍스트는 `texts[i]`, 없으면 실패로 취급. 같은 이름의 인접 턴은 한 줄로 합침. 빈 텍스트 턴은 건너뜀.
- Produces: `load_texts(path: Path) -> dict[int, str]` (turns.jsonl에서 text가 null이 아닌 줄만)
- Produces: `python -m stt_local.transcribe AUDIO --out DIR [--min-speakers 2] [--max-speakers 10] [--threshold 0.65] [--keywords "a,b"] [--db voiceprints] [--concurrency 8]`

- [ ] **Step 1: 실패하는 테스트 쓰기**

`test_stt_local.py`에 추가:

```python
import json

from stt_local.transcribe import load_texts, render


def test_render_merges_adjacent_same_name_and_marks_failures():
    turns = [seg(0, 1, "SPEAKER_00"), seg(1.5, 2, "SPEAKER_01"), seg(2.5, 3, "SPEAKER_00"),
             seg(60, 61, "SPEAKER_02"), seg(62, 63, "SPEAKER_00")]
    speakers = {"SPEAKER_00": {"name": "홍길동"}, "SPEAKER_01": {"name": "홍길동"},
                "SPEAKER_02": {"name": "Unknown-1"}}
    texts = {0: "안녕하세요", 1: "반갑습니다", 2: "", 4: "네"}  # 2는 빈 텍스트, 3은 실패
    md, rows = render(turns, texts, speakers)
    assert md == ("[00:00:00] 홍길동: 안녕하세요 반갑습니다\n"
                  "[00:01:00] Unknown-1: [전사 실패]\n"
                  "[00:01:02] 홍길동: 네\n")
    assert rows[0] == {"start": 0, "end": 2, "name": "홍길동", "text": "안녕하세요 반갑습니다"}
    assert len(rows) == 3


def test_load_texts_skips_null_and_keeps_last(tmp_path):
    p = tmp_path / "turns.jsonl"
    p.write_text("\n".join([
        json.dumps({"i": 0, "text": "a"}),
        json.dumps({"i": 1, "text": None}),
        json.dumps({"i": 0, "text": "b"}),
    ]) + "\n")
    assert load_texts(p) == {0: "b"}
    assert load_texts(tmp_path / "없음.jsonl") == {}
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v -k "render or load_texts"`
Expected: FAIL, `ModuleNotFoundError: No module named 'stt_local.transcribe'`

- [ ] **Step 3: 구현**

`stt_local/transcribe.py`:

```python
"""파이프라인 2: diarize → 성문 매칭 → 턴 병합 → gpt-transcribe → 전사문.

    python -m stt_local.transcribe 회의.m4a --out out/0902회의/ [--keywords "용어,제품명"]

out/speakers.json에서 이름을 고친 뒤 같은 명령을 다시 실행하면 STT 없이 전사문만 다시 만든다.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from stt_local.common import (DB_DIR, check_prereqs, db_load, diarize, fmt_ts, load_audio,
                              match, merge_turns, slice_wav)

STT_MODEL = "gpt-transcribe"
PRICE_PER_MIN = 0.0045


def load_texts(path: Path) -> dict[int, str]:
    """turns.jsonl에서 text가 null이 아닌 줄만. 같은 i는 마지막 줄이 이긴다."""
    texts: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d["text"] is not None:
                    texts[d["i"]] = d["text"]
    return texts


def render(turns: list[dict], texts: dict[int, str], speakers: dict[str, dict]) -> tuple[str, list[dict]]:
    """같은 이름의 인접 턴을 한 줄로. 빈 텍스트는 건너뛰고, 없는 텍스트는 [전사 실패]."""
    rows: list[dict] = []
    for i, t in enumerate(turns):
        text = texts.get(i)
        if text == "":
            continue
        if text is None:
            text = "[전사 실패]"
        name = speakers[t["speaker"]]["name"]
        if rows and rows[-1]["name"] == name:
            rows[-1]["text"] += " " + text
            rows[-1]["end"] = t["end"]
        else:
            rows.append({"start": t["start"], "end": t["end"], "name": name, "text": text})
    md = "".join(f"[{fmt_ts(r['start'])}] {r['name']}: {r['text']}\n" for r in rows)
    return md, rows


async def transcribe_all(wave: np.ndarray, turns: list[dict], todo: list[int], keywords: list[str],
                         cache: Path, concurrency: int) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    done = 0
    with cache.open("a") as f:
        async def one(i: int) -> None:
            nonlocal done
            t = turns[i]
            async with sem:
                try:
                    r = await client.audio.transcriptions.create(
                        model=STT_MODEL,
                        file=("turn.wav", slice_wav(wave, t["start"], t["end"]), "audio/wav"),
                        languages=["ko"],
                        **({"keywords": keywords} if keywords else {}),
                    )
                    text = r.text.strip()
                except Exception as e:
                    print(f"턴 {i} 실패: {type(e).__name__}: {e}", file=sys.stderr)
                    text = None
            f.write(json.dumps({"i": i, "speaker": t["speaker"], "start": t["start"],
                                "end": t["end"], "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  전사 {done}/{len(todo)}")

        await asyncio.gather(*(one(i) for i in todo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--max-speakers", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--keywords", default="", help="쉼표로 구분한 용어. 매칭된 실명은 자동 추가")
    ap.add_argument("--db", type=Path, default=DB_DIR)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    check_prereqs(need_openai=True)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    wave = load_audio(a.audio)

    # 1. diarize (캐시: 재실행 때 클러스터 라벨이 바뀌면 speakers.json이 어긋난다)
    dj, cj = out / "diarization.json", out / "centroids.npy"
    if dj.exists() and cj.exists():
        d = json.loads(dj.read_text())
        segs, cent = d["segments"], dict(zip(d["labels"], np.load(cj)))
        print("diarization 캐시 사용")
    else:
        segs, cent = diarize(wave, min_speakers=a.min_speakers, max_speakers=a.max_speakers)
        if not cent:
            sys.exit("화자를 찾지 못했습니다")
        dj.write_text(json.dumps({"segments": segs, "labels": list(cent)}, ensure_ascii=False))
        np.save(cj, np.stack(list(cent.values())))

    # 2. match (speakers.json이 있으면 사용자가 고친 것으로 보고 그대로 씀)
    sj = out / "speakers.json"
    if sj.exists():
        speakers = json.loads(sj.read_text())
        print("speakers.json 사용")
    else:
        speakers = match(cent, db_load(a.db), a.threshold)
        for label, v in speakers.items():
            v["minutes"] = round(sum(s["end"] - s["start"] for s in segs if s["speaker"] == label) / 60, 1)
        sj.write_text(json.dumps(speakers, ensure_ascii=False, indent=1))
    for label, v in speakers.items():
        print(f"  {label} -> {v['name']} (유사도 {v.get('score', '?')}, {v.get('minutes', '?')}분)")

    # 3. merge + 4. transcribe (turns.jsonl 캐시)
    turns = merge_turns(segs)
    cache = out / "turns.jsonl"
    texts = load_texts(cache)
    todo = [i for i in range(len(turns)) if i not in texts]
    minutes = sum(turns[i]["end"] - turns[i]["start"] for i in todo) / 60
    print(f"턴 {len(turns)}개, 전사 필요 {len(todo)}개 ({minutes:.1f}분, 약 ${minutes * PRICE_PER_MIN:.2f})")
    if todo:
        names = sorted({v["name"] for v in speakers.values() if not v["name"].startswith("Unknown")})
        keywords = [k.strip() for k in a.keywords.split(",") if k.strip()] + names
        asyncio.run(transcribe_all(wave, turns, todo, keywords, cache, a.concurrency))
        texts = load_texts(cache)

    # 5. render
    md, rows = render(turns, texts, speakers)
    (out / "transcript.md").write_text(md)
    (out / "transcript.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    failed = sum(1 for i in range(len(turns)) if i not in texts)
    print(f"완료: {out / 'transcript.md'} (실패 {failed}턴{', 재실행하면 다시 시도' if failed else ''})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest test_stt_local.py -v`
Expected: 18 passed

- [ ] **Step 5: 커밋**

```bash
git add stt_local/transcribe.py test_stt_local.py
git commit -m "feat: transcribe CLI (diarize→match→merge→STT→render)

Claude-Session: https://claude.ai/code/session_<id>"
```

---

### Task 8: 엔드투엔드 검증 (audio-2, 클로바 대조)

스파이크 결과(`spike/out/*`)와 클로바 라벨로 성문 DB를 만들고(audio-2 제외), audio-2를 전사해 speakers.json의 이름이 클로바 참석자와 맞는지, 이름을 고치고 재실행하는 경로가 STT 없이 도는지 확인한다. STT 비용 약 $0.20.

**Files:**
- Create: `spike/bootstrap_db.py` (버릴 코드, gitignore 대상)
- Create: `.env` (사용자가 OPENAI_API_KEY를 넣어야 함. 없으면 이 태스크는 여기서 멈추고 사용자에게 요청)

**Interfaces:**
- Consumes: `db_add` (common.py), `spike/out/<stem>/{exclusive.json,labels.json,centroids.npy}`, `~/Documents/source/STT_BMT/result/clovernote/audio-N.txt`

- [ ] **Step 1: 성문 DB 부트스트랩 스크립트**

`spike/bootstrap_db.py`:

```python
"""throwaway: 스파이크 diarization 결과에 클로바 라벨을 붙여 voiceprints/를 만든다. audio-2는 제외(테스트용)."""
import json
import re
import shutil
from pathlib import Path

import numpy as np

from stt_local.common import DB_DIR, db_add

CLOVA = Path.home() / "Documents/source/STT_BMT/result/clovernote"
HEAD = re.compile(r"^(?P<spk>\S+)\s+(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{2})$")
SESSIONS = {"audio-1": "audio-1", "audio-3": "audio-3", "audio-4-auto": "audio-4", "audio-5-auto": "audio-5"}


def clova_blocks(path):
    blocks = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = HEAD.match(line.strip())
        if m:
            blocks.append([m["spk"], int(m["h"] or 0) * 3600 + int(m["m"]) * 60 + int(m["s"]), None])
    for a, b in zip(blocks, blocks[1:]):
        a[2] = b[1]
    blocks[-1][2] = blocks[-1][1] + 60
    return blocks


def label_clusters(segs, blocks):
    acc = {}
    for s in segs:
        for name, b0, b1 in blocks:
            ov = min(s["end"], b1) - max(s["start"], b0)
            if ov > 0:
                acc.setdefault(s["speaker"], {}).setdefault(name, 0.0)
                acc[s["speaker"]][name] += ov
    return {spk: (max(d, key=d.get), max(d.values()) / sum(d.values()), sum(d.values()))
            for spk, d in acc.items()}


shutil.rmtree(DB_DIR, ignore_errors=True)
for stem, clova in SESSIONS.items():
    d = Path("spike/out") / stem
    segs = json.loads((d / "exclusive.json").read_text())
    labels = json.loads((d / "labels.json").read_text())
    C = np.load(d / "centroids.npy")
    lab = label_clusters(segs, clova_blocks(CLOVA / f"{clova}.txt"))
    for i, label in enumerate(labels):
        name, purity, dur = lab.get(label, ("?", 0, 0))
        if purity >= 0.75 and dur >= 120:
            db_add(DB_DIR, name, C[i])
            print(f"{stem} {label} -> {name} (순도 {purity:.2f}, {dur / 60:.1f}분)")
        else:
            print(f"{stem} {label} 건너뜀: {name} 순도 {purity:.2f}, {dur / 60:.1f}분")
```

Run:
```bash
PYTHONPATH=. .venv/bin/python spike/bootstrap_db.py
ls voiceprints/
.venv/bin/python -c "import numpy as np, glob; [print(f, np.load(f).shape) for f in sorted(glob.glob('voiceprints/*.npy'))]"
```
Expected: 참석자A, 참석자C, 참석자D, 참석자E, 참석자F, 참석자G, 참석자H, 참석자B 등 8명 내외, 각 (1~4, 256). audio-4의 0.7분 조각 클러스터는 건너뜀.

- [ ] **Step 2: OPENAI_API_KEY 확인**

```bash
grep -c OPENAI_API_KEY .env
```
Expected: `1`. 아니면 사용자에게 `.env`에 키를 넣어달라고 요청하고 여기서 멈춘다.

- [ ] **Step 3: audio-2 전사 (약 3분 diarize + 2~3분 STT, 약 $0.20)**

```bash
.venv/bin/python -m stt_local.transcribe ../STT_BMT/data/audio-2.aac --out out/audio-2 2>&1 | grep -vE "Warning|warn"
head -20 out/audio-2/transcript.md
cat out/audio-2/speakers.json
```
Expected:
- speakers.json에 클러스터 6개 내외. 이름은 클로바 참석자 `참석자H 참석자F 참석자D 참석자A 참석자B 참석자E` 안에서 나오고 유사도 0.7 이상. Unknown이 있으면 minutes가 1분 미만인 조각이어야 함.
- transcript.md가 `[00:00:09] 참석자F: ...` 형식. 실패 턴 0개.
- turns.jsonl 줄 수 = 턴 수 (약 500).

- [ ] **Step 4: 재실행 경로 확인 (STT 없이)**

speakers.json에서 아무 클러스터의 name을 `테스트`로 바꾼 뒤:
```bash
.venv/bin/python -m stt_local.transcribe ../STT_BMT/data/audio-2.aac --out out/audio-2 2>&1 | grep -vE "Warning|warn"
grep -c "테스트:" out/audio-2/transcript.md
```
Expected: `diarization 캐시 사용`, `speakers.json 사용`, `전사 필요 0개`, transcript.md에 `테스트:` 줄이 있음. 확인 후 이름을 원래대로 되돌리고 한 번 더 실행해 복구.

- [ ] **Step 5: 클로바 대비 CER (참고 수치)**

```bash
.venv/bin/python -c "
import sys, json; sys.path.insert(0, '../STT_BMT')
from compare import parse_clovernote, cer, norm
from pathlib import Path
ref = ' '.join(s['text'] for s in parse_clovernote(Path('../STT_BMT/result/clovernote/audio-2.txt')))
hyp = ' '.join(r['text'] for r in json.loads(Path('out/audio-2/transcript.json').read_text()))
print('ref', len(norm(ref)), 'hyp', len(norm(hyp)), 'ratio', round(len(norm(hyp))/len(norm(ref)), 2), 'CER*', round(cer(ref, hyp), 3))
"
```
Expected: 길이비 0.8~1.2, CER* 0.3 이하. (정렬 없는 근사치. 0.4를 넘으면 스펙의 "prompt에 직전 턴" 개선 검토 대상으로 기록)

- [ ] **Step 6: 스펙에 검증 결과 기록 후 커밋**

`docs/superpowers/specs/2026-09-02-stt-pipeline-design.md` 맨 끝에 `### 7. 엔드투엔드 검증 (audio-2)` 섹션을 추가하고 speakers.json 결과, 실패 턴 수, 길이비, CER*을 표로 적는다.

```bash
git add docs/
git commit -m "docs: audio-2 엔드투엔드 검증 결과

Claude-Session: https://claude.ai/code/session_<id>"
```
