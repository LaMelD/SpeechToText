"""오디오 로드, diarization, WAV 슬라이싱, 성문 DB, 매칭, 턴 병합."""
import io
import os
import shutil
import sys
import unicodedata
import wave as wave_mod
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL_ID = "pyannote/speaker-diarization-community-1"
DB_DIR = Path("voiceprints")
THREADS = min(6, os.cpu_count() or 1)   # torch CPU 스레드. 코어를 다 먹지 않게 고정한다


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


def merge_turns(segs: list[dict], gap: float = 1.0, min_dur: float = 0.3,
                max_dur: float = 600.0, bridge_gap: float = 2.5, interjection: float = 1.0) -> list[dict]:
    """1패스: 같은 화자의 인접 구간을 gap 이내면 합치고, min_dur 미만은 버린다.
    2패스: 다른 화자의 interjection초 이하 추임새를 건너뛰어 같은 화자 턴을 bridge_gap까지 묶는다.
    건너뛴 추임새는 묶인 구간 안에 남아 주 화자 텍스트에 섞인다. (실측: 문맥이 길어져 심판 거리 0.19→0.12)
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
    turns = [t for t in turns if t["end"] - t["start"] >= min_dur]

    out: list[dict] = []
    i = 0
    while i < len(turns):
        cur = dict(turns[i])
        j = i + 1
        while j < len(turns):
            k = j
            while (k < len(turns) and turns[k]["speaker"] != cur["speaker"]
                   and turns[k]["end"] - turns[k]["start"] <= interjection):
                k += 1
            if (k < len(turns) and turns[k]["speaker"] == cur["speaker"]
                    and turns[k]["start"] - cur["end"] <= bridge_gap
                    and turns[k]["end"] - cur["start"] <= max_dur):
                cur["end"] = turns[k]["end"]
                j = k + 1
            else:
                break
        out.append(cur)
        i = j
    return out


def check_model_tag(db_dir: Path) -> None:
    tag = db_dir / "model.txt"
    if not tag.exists():
        return
    got = tag.read_text(encoding="utf-8").strip()
    if got != MODEL_ID:
        sys.exit(f"성문 DB 모델({got})이 현재 모델({MODEL_ID})과 다릅니다. "
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


def load_env(path: Path = Path(".env")) -> None:
    """ponytail: python-dotenv 대신 KEY=VALUE 줄만 읽는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def check_prereqs(need_openai: bool) -> None:
    load_env()
    if not shutil.which("ffmpeg"):
        # torchcodec이 FFmpeg DLL을 링크하므로 static(essentials) 빌드로는 안 된다
        sys.exit("ffmpeg가 없습니다: winget install BtbN.FFmpeg.LGPL.Shared.8.1 (shared 빌드) 후 새 셸에서 실행")
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
    """exclusive 구간과 화자별 centroid. CPU로 돌린다.
    ponytail: device 고정. GPU를 쓸 거면 아래 "cpu"만 바꾼다."""
    import torch
    from pyannote.audio import Pipeline

    torch.set_num_threads(THREADS)
    pipe = Pipeline.from_pretrained(MODEL_ID).to(torch.device("cpu"))
    file = {"waveform": torch.from_numpy(wave)[None], "sample_rate": SAMPLE_RATE}
    kw = ({"num_speakers": num_speakers} if num_speakers
          else {"min_speakers": min_speakers, "max_speakers": max_speakers})
    out = pipe(file, **kw)
    segs = [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": spk}
            for s, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
    segs.sort(key=lambda s: s["start"])
    labels = out.speaker_diarization.labels()
    cent = {label: out.speaker_embeddings[i] for i, label in enumerate(labels)}
    return segs, cent
