"""오디오 로드(ffmpeg), sherpa-onnx diarization, WAV 슬라이싱, 성문 DB, 매칭, 턴 병합.

STT_LOCAL-pyannote와 같은 출력 스키마를 유지한다 (diarization.json / speakers.json /
turns.jsonl / transcript.{md,json}). 두 파이프라인 결과를 그대로 비교하기 위한 것이므로
fmt_ts / slice_wav / merge_turns / db_* / match / render는 손대지 않고 그대로 가져왔다.
"""
import io
import os
import shutil
import subprocess
import sys
import unicodedata
import wave as wave_mod
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

MODELS = Path("models")
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
# ponytail: 같은 폴더의 model.int8.onnx는 쓰지 않는다. 실회의 3건 200분 실측에서 diarization이 12% 빨라지지만
#  검출 발화가 22% 줄고(클로바 블록 누락 20→130개, 371→5,139자) 미등록 화자 오인식이 4%→24%로 는다.
#  속도 이득은 int8 연산이 아니라 놓친 구간의 임베딩을 건너뛴 데서 온다. 임베딩 모델에는 int8이 없다.
EMB_MODEL = MODELS / "3dspeaker_campplus_sv_zh_en_16k-common_advanced.onnx"

# 성문 벡터 공간을 결정하는 것은 임베딩 모델이다. pyannote(256dim)와 호환되지 않으므로
# model.txt 가드가 섞인 DB를 막는다. campplus는 192dim.
MODEL_ID = "sherpa-onnx/3dspeaker_campplus_sv_zh_en_16k-common_advanced"

DB_DIR = Path("voiceprints")

# 실측(i7-9750H, 6물리/12논리코어): 6스레드 RTF 0.13x, 1스레드 0.46x, 12스레드 0.39x.
# 논리코어의 절반(=물리코어)이 최적. HT까지 쓰면 오히려 퇴화한다.
DEFAULT_THREADS = min(6, os.cpu_count() or 1)   # 코어가 6개보다 적으면 그만큼만

MODEL_HELP = """모델을 내려받으세요 (프로젝트 루트에서):
  mkdir models; cd models
  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar -xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
  mv 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx 3dspeaker_campplus_sv_zh_en_16k-common_advanced.onnx"""


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
        tag.write_text(MODEL_ID + "\n", encoding="utf-8")
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
          threshold: float = 0.60) -> dict[str, dict]:
    """클러스터별 독립 argmax. 사람 점수는 그 사람 행들과의 코사인 최댓값.
    다대일 허용: 한 사람이 여러 클러스터로 갈라져도 전부 같은 이름.

    ponytail: threshold 0.60은 campplus 192dim 실측값이다. audio-3/4/5(143분, 클로바노트 정답)에서
    0.65 → 90.1%, 0.60 → 93.0%, 0.70 → 75.3%. 0.65는 절벽 바로 위였다.
    등록 인물을 놓치는 쪽(낮은 점수 탈락)과 미등록 인물을 오인식하는 쪽이 0.60 근처에서 갈린다:
    0.575면 미등록 Unknown 처리가 81.9%로 떨어지고 0.60에서 96.3%로 회복한다."""
    out, unmatched = {}, []
    for label, c in centroids.items():
        norm = np.linalg.norm(c)
        best, score = None, 0.0
        if norm > 0 and db:
            c = c / norm
            best, score = max(((name, float((rows @ c).max())) for name, rows in db.items()),
                              key=lambda kv: kv[1])
        if best is None or score < threshold:
            unmatched.append(label)
            best = None
        out[label] = {"name": best, "score": round(score, 3)}

    for i, group in enumerate(group_unknown(centroids, unmatched, threshold), 1):
        for label in group:
            out[label]["name"] = f"Unknown-{i}"
    return out


def group_unknown(centroids: dict[str, np.ndarray], labels: list[str],
                  threshold: float) -> list[list[str]]:
    """DB에 없는 클러스터끼리 묶어 같은 사람에게 같은 번호를 준다.
    과분할된 클러스터 수십 개가 미등록 1명을 Unknown-17/18/20/21로 흩어놓는 걸 막는다.

    그룹 대표는 소속 벡터의 평균(centroid linkage)이다. 단일 연결로 하면 사슬처럼 이어붙어
    서로 다른 사람이 한 그룹이 된다. norm이 큰(=구간들이 일관된) 클러스터부터 그룹을 연다.
    ponytail: 묶는 기준도 DB 매칭과 같은 threshold다. 같은 화자인지를 재는 척도가 같기 때문."""
    groups: list[list[str]] = []
    reps: list[np.ndarray] = []
    for label in sorted(labels, key=lambda l: (-float(np.linalg.norm(centroids[l])), l)):
        v = centroids[label]
        n = np.linalg.norm(v)
        if n == 0:                      # 영벡터는 비교 불가, 항상 새 그룹
            groups.append([label]); reps.append(None)
            continue
        v = v / n
        best, sim = None, threshold
        for j, r in enumerate(reps):
            if r is None:
                continue
            s = float(r @ v)
            if s >= sim:
                best, sim = j, s
        if best is None:
            groups.append([label]); reps.append(v)
        else:
            groups[best].append(label)
            m = np.mean([centroids[l] / np.linalg.norm(centroids[l]) for l in groups[best]], axis=0)
            reps[best] = m / np.linalg.norm(m)
    return groups


def filter_db(db: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """참석자를 지정하면 성문 DB 후보를 그 사람들로 제한한다. 미지정이면 그대로.
    회의에 없는 사람이 임계값을 넘겨 끼어드는 오탐을 막는다 (실측: 3개 회의에서 1.0분).
    DB에 없는 이름은 경고만 하고 무시한다 — 미등록 참석자는 Unknown으로 나와야 정상."""
    if not names:
        return db
    wanted = {unicodedata.normalize("NFC", n) for n in names}
    missing = wanted - set(db)
    if missing:
        print(f"성문 DB에 없는 참석자(Unknown으로 처리): {', '.join(sorted(missing))}", file=sys.stderr)
    return {k: v for k, v in db.items() if k in wanted}


def load_keywords(path: Path = Path("keywords.txt")) -> list[str]:
    """프로젝트 루트 keywords.txt의 STT 키워드. 한 줄에 하나, 쉼표 구분도 허용, #은 주석.
    CLI --keywords로 매번 25개를 넘기는 대신 파일로 관리한다."""
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        out += [k.strip() for k in line.split(",") if k.strip()]
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
    """pyannote판과 달리 HF 토큰이 필요 없다 (sherpa-onnx 모델은 직접 내려받는다)."""
    load_env()
    if not shutil.which("ffmpeg"):
        # 여기서는 디코딩만 쓰므로 static 빌드도 되지만, 안내는 pyannote판과 통일한다
        sys.exit("ffmpeg가 없습니다: winget install BtbN.FFmpeg.LGPL.Shared.8.1 후 새 셸에서 실행")
    for f in (SEG_MODEL, EMB_MODEL):
        if not f.exists():
            sys.exit(f"모델이 없습니다: {f}\n{MODEL_HELP}")
    if need_openai and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없습니다: 프로젝트 루트 .env에 OPENAI_API_KEY=... 추가")


def load_audio(path: str) -> np.ndarray:
    """ffmpeg로 16kHz mono float32 디코딩.
    ponytail: pyannote.audio(torch 의존)나 soundfile+librosa 대신 ffmpeg 파이프.
    ffmpeg는 어차피 필수 의존이고, 이 프로젝트는 torch를 아예 안 쓴다."""
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        sys.exit(f"ffmpeg 디코딩 실패({path}): {p.stderr.decode('utf-8', 'replace').strip()}")
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _extractor(threads: int):
    import sherpa_onnx
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=threads))


def embed(ex, wave: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, round(start * SAMPLE_RATE))
    b = min(len(wave), round(end * SAMPLE_RATE))
    s = ex.create_stream()
    s.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(wave[a:b]))
    s.input_finished()
    return np.array(ex.compute(s), dtype=np.float32)


def cluster_centroids(wave: np.ndarray, segs: list[dict], threads: int = DEFAULT_THREADS,
                      clips: int = 10) -> dict[str, np.ndarray]:
    """화자별 centroid. sherpa-onnx diarization은 클러스터 centroid를 노출하지 않으므로
    같은 임베딩 모델로 직접 계산한다. 각 벡터를 정규화한 뒤 평균해야 긴 구간이 독식하지 않는다.

    ponytail: 전 구간이 아니라 가장 긴 clips개만 쓴다. 짧은 구간 임베딩이 불안정하다는 건
    실측됐다(min_duration_on 0.3→3.0에서 클러스터 7개→3개). 정확도가 부족하면 clips를 올린다."""
    ex = _extractor(threads)
    out: dict[str, np.ndarray] = {}
    for label in sorted({s["speaker"] for s in segs}):
        mine = sorted((s for s in segs if s["speaker"] == label),
                      key=lambda s: s["end"] - s["start"], reverse=True)[:clips]
        vecs = []
        for s in mine:
            v = embed(ex, wave, s["start"], s["end"])
            n = np.linalg.norm(v)
            if n > 0:
                vecs.append(v / n)
        out[label] = (np.mean(vecs, axis=0).astype(np.float32) if vecs
                      else np.zeros(ex.dim, dtype=np.float32))
    return out


def centroid_from_clips(paths: list[Path], threads: int = DEFAULT_THREADS) -> np.ndarray:
    """폴더에 남은 클립들로 centroid를 다시 계산한다. 클립을 듣고 남의 목소리를 지운 뒤 커밋하면
    그 클립이 성문에서 빠진다. 클러스터 수를 늘려도 안 갈라지는 두 사람(실측: audio-1 참석자E/참석자A)은
    이 방법으로만 분리된다. 클립을 안 지웠으면 diarize가 만든 centroid와 코사인 1.0000으로 일치한다."""
    ex = _extractor(threads)
    vecs = []
    for p in paths:
        with wave_mod.open(str(p), "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
                sys.exit(f"{p}: 16kHz mono WAV가 아닙니다")
            wave = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767
        v = embed(ex, wave, 0.0, len(wave) / SAMPLE_RATE)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(ex.dim, dtype=np.float32)


def diarize(wave: np.ndarray, num_speakers: int | None = None, cluster_threshold: float = 0.8,
            min_duration_on: float = 1.5, min_duration_off: float = 0.5,
            threads: int = DEFAULT_THREADS, clips: int = 10,
            progress: bool = True) -> tuple[list[dict], dict[str, np.ndarray]]:
    """구간과 화자별 centroid. CPU 전용(onnxruntime).

    min_duration_on 기본 1.5초는 실측값이다. 90초 클립(정답 3명)에서
    0.3 → 7명 / 1.0 → 5명 / 1.5 → 4명 / 3.0 → 3명.
    3.0이 화자 수는 정확하지만 0.4~0.9초 추임새를 버려서 merge_turns의 추임새 흡수가 죽는다.
    1.5는 과분할을 남기지만 match()가 다대일을 허용하므로 성문 DB가 흡수할 수 있다.
    실회의 3건 200분 재실측(클로바노트 정답, 성문 DB 7명, threshold 0.60):
      1.0 → 등록 정확도 92.7% / 클로바 블록 누락 179자 / STT 180분
      1.5 → 93.2% / 371자 / 169분
      2.0 → 93.7% / 559자 / 159분
    올릴수록 정확도가 오르는 건 어려운 짧은 구간을 버린 효과다. 회의록엔 놓친 질문이 더 아프고
    3초 미만 파편은 render가 '불명'으로 접으므로 1.5를 유지한다.
    ponytail: 이 값이 이 파이프라인의 유일한 튜닝 노브다. 결과가 이상하면 여기부터 만진다."""
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL), window_shift_ratio=0.1),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=cluster_threshold),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not cfg.validate():
        sys.exit(f"sherpa-onnx config 검증 실패. 모델 경로를 확인하세요.\n{MODEL_HELP}")

    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if sd.sample_rate != SAMPLE_RATE:
        sys.exit(f"모델 샘플레이트({sd.sample_rate})가 {SAMPLE_RATE}과 다릅니다")

    def cb(done: int, total: int) -> int:
        if total and (done % max(1, total // 10) == 0 or done == total):
            print(f"  diarize {done * 100 // total}%")
        return 0

    res = sd.process(wave, callback=cb if progress else None).sort_by_start_time()
    segs = [{"start": round(r.start, 3), "end": round(r.end, 3),
             "speaker": f"SPEAKER_{r.speaker:02d}"} for r in res]
    if not segs:
        return [], {}
    return segs, cluster_centroids(wave, segs, threads, clips)
