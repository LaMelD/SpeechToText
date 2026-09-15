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
PROMPT_CHARS = 300   # 직전 텍스트를 prompt로 넘길 길이
PROMPT_TURNS = 12    # 문맥으로 볼 직전 턴 수


def load_texts(path: Path, turns: list[dict]) -> dict[int, str]:
    """turns.jsonl에서 text가 null이 아니고 현재 turns의 인덱스/화자/시간과 일치하는 줄만.
    같은 i는 마지막 줄이 이긴다. 맞지 않는 줄은 무시하고 개수를 출력한다 (다른 diarization 결과 잔재)."""
    texts: dict[int, str] = {}
    mismatched = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d["text"] is None:
                    continue
                i = d["i"]
                if (0 <= i < len(turns) and d["speaker"] == turns[i]["speaker"]
                        and d["start"] == turns[i]["start"] and d["end"] == turns[i]["end"]):
                    texts[i] = d["text"]
                else:
                    mismatched += 1
    if mismatched:
        print(f"turns.jsonl에서 현재 diarization과 맞지 않는 {mismatched}줄을 무시합니다 (다시 전사)")
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


def prompt_context(known: dict[int, str], i: int) -> str:
    """턴 i 직전 PROMPT_TURNS개 중 이미 전사된 텍스트를 이어 붙인 마지막 PROMPT_CHARS자."""
    prev = [known[j] for j in range(max(0, i - PROMPT_TURNS), i) if known.get(j)]
    return " ".join(prev)[-PROMPT_CHARS:]


async def transcribe_all(wave: np.ndarray, turns: list[dict], todo: list[int], keywords: list[str],
                         cache: Path, concurrency: int, texts: dict[int, str]) -> None:
    """todo를 concurrency개 연속 구간으로 나눠 구간 안에서는 직전 텍스트를 prompt로 넘기며 순차,
    구간끼리는 병렬로 전사한다. 구간의 첫 턴만 문맥이 없다. (실측: prompt 문맥이 심판 거리 0.19→0.12)"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    known = dict(texts)
    done = 0
    n = -(-len(todo) // max(1, concurrency))
    chunks = [todo[k:k + n] for k in range(0, len(todo), n)]
    with cache.open("a", encoding="utf-8") as f:
        async def run_chunk(idx: list[int]) -> None:
            nonlocal done
            for i in idx:
                t = turns[i]
                prompt = prompt_context(known, i)
                try:
                    r = await client.audio.transcriptions.create(
                        model=STT_MODEL,
                        file=("turn.wav", slice_wav(wave, t["start"], t["end"]), "audio/wav"),
                        languages=["ko"],
                        **({"keywords": keywords} if keywords else {}),
                        **({"prompt": prompt} if prompt else {}),
                    )
                    text = r.text.strip()
                    known[i] = text
                except Exception as e:
                    print(f"턴 {i} 실패: {type(e).__name__}: {e}", file=sys.stderr)
                    text = None
                f.write(json.dumps({"i": i, "speaker": t["speaker"], "start": t["start"],
                                    "end": t["end"], "text": text}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"  전사 {done}/{len(todo)}")

        await asyncio.gather(*(run_chunk(c) for c in chunks))


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
    sj = out / "speakers.json"
    if dj.exists() and cj.exists():
        d = json.loads(dj.read_text(encoding="utf-8"))
        if d.get("n_samples") != len(wave):
            sys.exit(f"{dj}는 다른 오디오의 diarization 결과입니다. --out을 바꾸거나 {out}/를 지우세요")
        segs, cent = d["segments"], dict(zip(d["labels"], np.load(cj)))
        print("diarization 캐시 사용")
    else:
        if sj.exists():
            print("diarization을 새로 하므로 speakers.json을 다시 만듭니다")
            sj.unlink()
        segs, cent = diarize(wave, min_speakers=a.min_speakers, max_speakers=a.max_speakers)
        if not cent:
            sys.exit("화자를 찾지 못했습니다")
        dj.write_text(json.dumps({"segments": segs, "labels": list(cent), "n_samples": len(wave)}, ensure_ascii=False), encoding="utf-8")
        np.save(cj, np.stack(list(cent.values())))

    # 2. match (speakers.json이 있으면 사용자가 고친 것으로 보고 그대로 씀)
    if sj.exists():
        speakers = json.loads(sj.read_text(encoding="utf-8"))
        print("speakers.json 사용")
    else:
        speakers = match(cent, db_load(a.db), a.threshold)
        for label, v in speakers.items():
            v["minutes"] = round(sum(s["end"] - s["start"] for s in segs if s["speaker"] == label) / 60, 1)
        sj.write_text(json.dumps(speakers, ensure_ascii=False, indent=1), encoding="utf-8")
    for label, v in speakers.items():
        print(f"  {label} -> {v['name']} (유사도 {v.get('score', '?')}, {v.get('minutes', '?')}분)")

    # 3. merge + 4. transcribe (turns.jsonl 캐시)
    turns = merge_turns(segs)
    cache = out / "turns.jsonl"
    texts = load_texts(cache, turns)
    todo = [i for i in range(len(turns)) if i not in texts]
    minutes = sum(turns[i]["end"] - turns[i]["start"] for i in todo) / 60
    print(f"턴 {len(turns)}개, 전사 필요 {len(todo)}개 ({minutes:.1f}분, 약 ${minutes * PRICE_PER_MIN:.2f})")
    if todo:
        names = sorted({v["name"] for v in speakers.values() if not v["name"].startswith("Unknown")})
        keywords = [k.strip() for k in a.keywords.split(",") if k.strip()] + names
        asyncio.run(transcribe_all(wave, turns, todo, keywords, cache, a.concurrency, texts))
        texts = load_texts(cache, turns)

    # 5. render
    md, rows = render(turns, texts, speakers)
    (out / "transcript.md").write_text(md, encoding="utf-8")
    (out / "transcript.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = sum(1 for i in range(len(turns)) if i not in texts)
    print(f"완료: {out / 'transcript.md'} (실패 {failed}턴{', 재실행하면 다시 시도' if failed else ''})")


if __name__ == "__main__":
    main()
