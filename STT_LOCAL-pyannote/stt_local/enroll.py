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
