import io
import json
import wave as wave_mod

import numpy as np
import pytest

from stt_local.common import (SAMPLE_RATE, fmt_ts, slice_wav, merge_turns, MODEL_ID, check_model_tag,
                              db_add, db_load, match, group_unknown, filter_db, load_keywords)
from stt_local.transcribe import load_texts, render


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


def seg(start, end, spk="A"):
    return {"start": start, "end": end, "speaker": spk}


def test_merge_same_speaker_within_gap():
    out = merge_turns([seg(0, 1), seg(1.5, 3), seg(4.5, 5), seg(8, 9)])
    assert out == [seg(0, 5), seg(8, 9)]  # 2.5초 이내는 묶고 3초는 안 묶음


def test_merge_does_not_cross_long_other_speaker_turn():
    # 다른 화자 턴이 1초를 넘으면 건너뛰지 않는다
    out = merge_turns([seg(0, 1, "A"), seg(1.2, 2.5, "B"), seg(2.6, 3, "A")])
    assert out == [seg(0, 1, "A"), seg(1.2, 2.5, "B"), seg(2.6, 3, "A")]


def test_merge_bridges_short_interjection():
    # B의 0.8초 추임새를 건너뛰어 A를 하나로 묶는다. 추임새 오디오는 A 구간 안에 남는다
    out = merge_turns([seg(0, 1, "A"), seg(1.2, 2, "B"), seg(2.1, 3, "A")])
    assert out == [seg(0, 3, "A")]


def test_merge_bridge_skips_several_interjections():
    out = merge_turns([seg(0, 1, "A"), seg(1.1, 1.5, "B"), seg(1.6, 2.0, "C"), seg(2.2, 3, "A")])
    assert out == [seg(0, 3, "A")]


def test_merge_bridge_respects_gap_and_max_dur():
    # 같은 화자라도 사이가 2.5초를 넘으면 안 묶고, 추임새는 그대로 남는다
    out = merge_turns([seg(0, 1, "A"), seg(1.2, 2, "B"), seg(3.6, 4, "A")])
    assert out == [seg(0, 1, "A"), seg(1.2, 2, "B"), seg(3.6, 4, "A")]
    out = merge_turns([seg(0, 400, "A"), seg(400.5, 401, "B"), seg(401.5, 700, "A")], max_dur=600)
    assert out == [seg(0, 400, "A"), seg(400.5, 401, "B"), seg(401.5, 700, "A")]


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


def test_match_merges_similar_unknowns():
    """미등록 1명이 여러 클러스터로 쪼개져도 Unknown 번호는 하나여야 한다."""
    db = {"홍길동": np.stack([unit(1, 0)])}
    cent = {"SPEAKER_00": unit(0, 1), "SPEAKER_01": unit(0.15, 1), "SPEAKER_02": unit(-0.1, 1)}
    out = match(cent, db, threshold=0.65)
    assert {v["name"] for v in out.values()} == {"Unknown-1"}


def test_match_keeps_different_unknowns_apart():
    db = {"홍길동": np.stack([unit(1, 0)])}
    cent = {"SPEAKER_00": unit(0, 1), "SPEAKER_01": unit(0, -1)}
    out = match(cent, db, threshold=0.65)
    assert out["SPEAKER_00"]["name"] != out["SPEAKER_01"]["name"]


def test_group_unknown_centroid_linkage_blocks_chaining():
    """사슬 연결 방지: A-B가 가깝고 B-C가 가까워도 A-C가 멀면 한 그룹이 안 된다."""
    import math
    def rot(deg):
        r = math.radians(deg)
        return np.array([math.cos(r), math.sin(r)], dtype=np.float32)
    cent = {"A": rot(0), "B": rot(40), "C": rot(80)}   # 이웃끼리 cos40=0.77, 양끝 cos80=0.17
    groups = group_unknown(cent, list(cent), threshold=0.65)
    assert not any({"A", "C"} <= set(g) for g in groups)


def test_match_empty_db_or_zero_centroid():
    assert match({"S": unit(1, 0)}, {}) == {"S": {"name": "Unknown-1", "score": 0.0}}
    db = {"홍길동": np.stack([unit(1, 0)])}
    assert match({"S": np.zeros(2, dtype=np.float32)}, db) == {"S": {"name": "Unknown-1", "score": 0.0}}


def test_render_merges_adjacent_same_name_and_marks_failures():
    turns = [seg(0, 1, "SPEAKER_00"), seg(1.5, 2, "SPEAKER_01"), seg(2.5, 3, "SPEAKER_00"),
             seg(60, 64, "SPEAKER_02"), seg(65, 66, "SPEAKER_00")]   # Unknown은 3초 이상이어야 번호 유지
    speakers = {"SPEAKER_00": {"name": "홍길동"}, "SPEAKER_01": {"name": "홍길동"},
                "SPEAKER_02": {"name": "Unknown-1"}}
    texts = {0: "안녕하세요", 1: "반갑습니다", 2: "", 4: "네"}  # 2는 빈 텍스트, 3은 실패
    md, rows = render(turns, texts, speakers)
    assert md == ("[00:00:00] 홍길동: 안녕하세요 반갑습니다\n"
                  "[00:01:00] Unknown-1: [전사 실패]\n"
                  "[00:01:05] 홍길동: 네\n")
    assert rows[0] == {"start": 0, "end": 2, "name": "홍길동", "text": "안녕하세요 반갑습니다"}
    assert len(rows) == 3


def test_filter_db_keeps_only_participants(capsys):
    db = {"홍길동": np.stack([unit(1, 0)]), "김철수": np.stack([unit(0, 1)]), "이영희": np.stack([unit(1, 1)])}
    out = filter_db(db, ["홍길동", "이영희"])
    assert set(out) == {"홍길동", "이영희"}
    assert filter_db(db, []) is db                       # 미지정이면 그대로


def test_filter_db_warns_on_name_not_in_db(capsys):
    db = {"홍길동": np.stack([unit(1, 0)])}
    out = filter_db(db, ["홍길동", "없는사람"])
    assert set(out) == {"홍길동"}
    assert "없는사람" in capsys.readouterr().err


def test_load_keywords_reads_lines_and_skips_comments(tmp_path):
    f = tmp_path / "keywords.txt"
    f.write_text("# 도메인 용어\n리니어\n\nGit, 깃\n  커밋  \n", encoding="utf-8")
    assert load_keywords(f) == ["리니어", "Git", "깃", "커밋"]


def test_load_keywords_missing_file_is_empty(tmp_path):
    assert load_keywords(tmp_path / "nope.txt") == []


def test_render_folds_short_unknown_into_generic_label():
    """3초 미만 Unknown 파편은 번호 없이 [불명]으로 접고, 인접한 것끼리 합친다. 긴 Unknown은 번호 유지."""
    turns = [seg(0, 5, "SPEAKER_00"), seg(5, 7, "SPEAKER_01"), seg(7, 9, "SPEAKER_02"),
             seg(10, 20, "SPEAKER_03")]
    speakers = {"SPEAKER_00": {"name": "홍길동"}, "SPEAKER_01": {"name": "Unknown-1"},
                "SPEAKER_02": {"name": "Unknown-2"}, "SPEAKER_03": {"name": "Unknown-3"}}
    texts = {0: "안녕하세요", 1: "어", 2: "응", 3: "저는 처음 뵙는 사람입니다"}
    md, rows = render(turns, texts, speakers, unknown_min=3.0)
    assert md == ("[00:00:00] 홍길동: 안녕하세요\n"
                  "[00:00:05] 불명: 어 응\n"
                  "[00:00:10] Unknown-3: 저는 처음 뵙는 사람입니다\n")


def test_commit_skips_folder_already_committed(tmp_path, capsys):
    from stt_local.enroll import commit
    src, db = tmp_path / "enroll_out", tmp_path / "voiceprints"
    d = src / "홍길동"; d.mkdir(parents=True)
    np.save(d / "centroid.npy", np.array([1.0, 0.0]))
    commit(src, db)
    commit(src, db)                                       # 두 번째는 건너뛰어야 한다
    assert db_load(db)["홍길동"].shape[0] == 1
    assert (d / ".committed").exists()
    assert "건너" in capsys.readouterr().out


def test_load_texts_skips_null_and_keeps_last(tmp_path):
    turns = [seg(0, 1, "S0"), seg(2, 3, "S1")]
    p = tmp_path / "turns.jsonl"
    p.write_text("\n".join([
        json.dumps({"i": 0, "speaker": "S0", "start": 0, "end": 1, "text": "a"}),
        json.dumps({"i": 1, "speaker": "S1", "start": 2, "end": 3, "text": None}),
        json.dumps({"i": 0, "speaker": "S0", "start": 0, "end": 1, "text": "b"}),
        json.dumps({"i": 0, "speaker": "S0", "start": 5, "end": 1, "text": "stale"}),  # 다른 diarization 잔재
    ]) + "\n")
    assert load_texts(p, turns) == {0: "b"}
    assert load_texts(tmp_path / "없음.jsonl", turns) == {}


from stt_local.transcribe import prompt_context


def test_prompt_context_uses_recent_known_text_only():
    known = {0: "가" * 200, 1: "나" * 200, 3: "다"}
    ctx = prompt_context(known, 4)                 # 빠진 턴(2)은 건너뛰고 마지막 300자만
    assert ctx.endswith("다") and len(ctx) == 300
    assert prompt_context({0: "멀리", 13: "가까이"}, 14) == "가까이"   # 12턴 창 밖은 제외
    assert prompt_context(known, 0) == ""          # 앞 턴이 없으면 빈 문맥
    assert prompt_context({}, 5) == ""


# --- sherpa-onnx판에서 새로 쓴 로직: centroid를 직접 계산하는 부분 ---

from stt_local import common
from stt_local.common import cluster_centroids


class FakeStream:
    def __init__(self):
        self.n = 0

    def accept_waveform(self, sr, samples):
        self.n = len(samples)

    def input_finished(self):
        pass


class FakeExtractor:
    """구간 길이에 따라 다른 벡터를 돌려주는 가짜 임베딩 추출기."""
    dim = 2

    def __init__(self):
        self.seen = []

    def create_stream(self):
        return FakeStream()

    def compute(self, s):
        dur = round(s.n / SAMPLE_RATE, 1)
        self.seen.append(dur)
        return {3.0: [3.0, 0.0], 1.0: [0.0, 1.0], 2.0: [0.0, 0.0]}.get(dur, [0.1, 0.1])


def test_cluster_centroids_normalizes_before_averaging(monkeypatch):
    fake = FakeExtractor()
    monkeypatch.setattr(common, "_extractor", lambda threads: fake)
    wave = np.zeros(SAMPLE_RATE * 25, dtype=np.float32)
    segs = [seg(0, 3, "S0"), seg(10, 11, "S0"), seg(20, 20.5, "S0")]
    cent = cluster_centroids(wave, segs, clips=2)
    # 긴 2개만 쓴다 (0.5초는 제외)
    assert fake.seen == [3.0, 1.0]
    # 정규화 후 평균: ([1,0] + [0,1]) / 2. 정규화를 안 했다면 [1.5, 0.5]가 됐을 것
    np.testing.assert_allclose(cent["S0"], [0.5, 0.5])


def test_centroid_from_clips_normalizes_and_skips_zero(monkeypatch, tmp_path):
    """클립 wav로 centroid 재계산: 정규화 후 평균, 영벡터 클립은 제외."""
    from stt_local.common import centroid_from_clips
    fake = FakeExtractor()
    monkeypatch.setattr(common, "_extractor", lambda threads: fake)
    for name, sec in (("a.wav", 3), ("b.wav", 1), ("c.wav", 2)):   # 2초는 FakeExtractor가 영벡터
        (tmp_path / name).write_bytes(slice_wav(np.zeros(SAMPLE_RATE * sec, dtype=np.float32), 0, sec, pad=0))
    cent = centroid_from_clips(sorted(tmp_path.glob("*.wav")))
    assert fake.seen == [3.0, 1.0, 2.0]
    np.testing.assert_allclose(cent, [0.5, 0.5])


def test_commit_recomputes_centroid_from_remaining_clips(monkeypatch, tmp_path, capsys):
    """클립을 지우고 커밋하면 남은 클립으로 다시 계산하고, centroid.npy는 무시한다."""
    from stt_local.enroll import commit
    fake = FakeExtractor()
    monkeypatch.setattr(common, "_extractor", lambda threads: fake)
    src, db = tmp_path / "enroll_out", tmp_path / "voiceprints"
    d = src / "홍길동"; d.mkdir(parents=True)
    np.save(d / "centroid.npy", np.array([9.0, 9.0]))                      # 오염된 옛 centroid
    (d / "1_0000-0003.wav").write_bytes(slice_wav(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), 0, 3, pad=0))
    commit(src, db)
    np.testing.assert_allclose(db_load(db)["홍길동"][0], [1.0, 0.0])     # 3초 클립 → [3,0] 정규화


def test_cluster_centroids_zero_vectors_give_zero_centroid(monkeypatch):
    monkeypatch.setattr(common, "_extractor", lambda threads: FakeExtractor())
    wave = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    cent = cluster_centroids(wave, [seg(0, 2, "S1")], clips=6)
    np.testing.assert_allclose(cent["S1"], [0.0, 0.0])
