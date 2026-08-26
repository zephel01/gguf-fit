"""gguf-fetch のテスト。**ネットワークには一切出ない**.

ヘッダのパーサは純粋関数なので合成したバイト列で突ける。HTTP が要る部分は
``http.server`` を localhost に立てて ``HF_ENDPOINT`` を向ける。Hugging Face が
落ちていても、CI にネットワークが無くても、このテストは同じ結果を出す。

固定しているのは主に3つ:

  * **量子化ラベルの取り出し** — ファイル名の解釈を間違えると別の量子化を
    落とす。実物で見た名前を並べて固定してある
  * **分割 GGUF をまとめること** — ``-00001-of-00003`` を3本と数えると
    「3量子化ある」ことになり、サイズも3分の1で判定してしまう
  * **Range を無視するサーバで止まること** — 気づかずに 21 GB 受け取るのを
    防ぐ最後の砦
"""

from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from gguf_fit import _config, _ggufhdr, _hardware, fetch
from gguf_fit._messages import pad, width
from gguf_fit.plan import calibration_mismatch

# --- 合成 GGUF ------------------------------------------------------------

T_UINT32, T_FLOAT32, T_STRING, T_ARRAY = 4, 6, 8, 9


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(meta: list[tuple[str, int, bytes]],
               tensors: list[tuple[str, int]], trailer: int = 4096,
               dims: tuple[int, int] = (8, 8)) -> bytes:
    """テストで使う最小の GGUF。``trailer`` は「本体のつもり」の埋め草.

    ``dims`` はテンソルの形。**パラメータ数がここで決まる**ので、bpw を見る
    テストでは実物に近い大きさ (BIG_DIMS) を渡すこと。
    """
    out = [b"GGUF", struct.pack("<I", 3),
           struct.pack("<Q", len(tensors)), struct.pack("<Q", len(meta))]
    for key, vtype, payload in meta:
        out += [_gstr(key), struct.pack("<I", vtype), payload]
    for name, code in tensors:
        out += [_gstr(name), struct.pack("<I", 2),
                struct.pack("<QQ", *dims), struct.pack("<I", code),
                struct.pack("<Q", 0)]
    return b"".join(out) + b"\0" * trailer


#: 11本のテンソルで合計 22.5B パラメータ。ファイルサイズ 20〜35 GB に対して
#: bpw が 7〜12 に落ちるので、実物と同じ範囲で判定を通せる
BIG_DIMS = (4096, 500_000)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


#: 11層中3層だけが attn_k/attn_v を持つハイブリッド注意。残りは attn_qkv
MODEL_META = [
    ("general.architecture", T_STRING, _gstr("testarch")),
    ("testarch.block_count", T_UINT32, _u32(6)),
    ("testarch.context_length", T_UINT32, _u32(262144)),
    ("testarch.attention.head_count_kv", T_UINT32, _u32(4)),
    ("testarch.attention.key_length", T_UINT32, _u32(128)),
    ("testarch.attention.value_length", T_UINT32, _u32(128)),
    ("testarch.rope.freq_base", T_FLOAT32, struct.pack("<f", 10000.0)),
    # 語彙。**読み飛ばされることを確かめるために大きめに入れる**
    ("tokenizer.ggml.tokens", T_ARRAY,
     _u32(T_STRING) + struct.pack("<Q", 500) + b"".join(_gstr(f"tok{i}")
                                                        for i in range(500))),
    ("tokenizer.chat_template", T_STRING, _gstr("{% if think %}...{% endif %}")),
]
MODEL_TENSORS = (
    [(f"blk.{i}.attn_k.weight", 12) for i in range(3)]
    + [(f"blk.{i}.attn_v.weight", 12) for i in range(3)]
    + [(f"blk.{i}.attn_qkv.weight", 12) for i in range(3, 6)]
    + [("token_embd.weight", 12), ("output.weight", 14)]
)


# --- ヘッダのパーサ -------------------------------------------------------

def test_parses_metadata_and_tensor_types():
    header = _ggufhdr.parse_header(build_gguf(MODEL_META, MODEL_TENSORS))
    assert header.version == 3
    assert header.metadata["general.architecture"] == "testarch"
    assert header.metadata["testarch.block_count"] == 6
    assert ("blk.0.attn_k.weight", "Q4_K") in header.tensors
    assert ("output.weight", "Q6_K") in header.tensors
    assert len(header.tensors) == len(MODEL_TENSORS)


def test_want_filter_skips_the_vocabulary():
    """語彙を持ち歩かない。**位置は正しく進む**ので後続が読めていること."""
    data = build_gguf(MODEL_META, MODEL_TENSORS)
    header = _ggufhdr.parse_header(data, ("general.architecture", ".block_count"))
    assert set(header.metadata) == {"general.architecture", "testarch.block_count"}
    assert len(header.tensors) == len(MODEL_TENSORS)


def test_truncated_says_how_many_bytes_are_needed():
    data = build_gguf(MODEL_META, MODEL_TENSORS, trailer=0)
    with pytest.raises(_ggufhdr.TruncatedGGUF) as excinfo:
        _ggufhdr.parse_header(data[:200])
    # 「足りない」は「壊れている」ではない。追加取得の目安が出ること
    assert excinfo.value.need > 200
    assert _ggufhdr.parse_header(data[:excinfo.value.need] + data[excinfo.value.need:])


def test_bad_magic_is_a_value_error_not_truncation():
    """取り直しても直らないものを TruncatedGGUF にすると、無限に取りにいく."""
    with pytest.raises(ValueError, match="GGUF"):
        _ggufhdr.parse_header(b"NOPE" + b"\0" * 64)


def test_implausible_counts_are_rejected():
    bad = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 2 ** 60) \
        + struct.pack("<Q", 1)
    with pytest.raises(ValueError, match="implausible"):
        _ggufhdr.parse_header(bad)


def test_unknown_tensor_type_is_named_not_guessed():
    """未知の型を F32 などに丸めると、量子化の内訳が静かに嘘になる."""
    assert _ggufhdr.type_name(250) == "TYPE_250"
    assert _ggufhdr.type_name(14) == "Q6_K"


# --- ファイル名の解釈 -----------------------------------------------------

@pytest.mark.parametrize(("filename", "expected"), [
    ("Ornith-1.5-35B-Q4_K_M.gguf", "Q4_K_M"),
    ("Ornith-1.5-35B-BF16.gguf", "BF16"),
    ("Qwen3.8-27B-UD-Q5_K_XL.gguf", "UD-Q5_K_XL"),
    ("Qwen3.8-27B-IQ4_XS.gguf", "IQ4_XS"),
    ("Qwen3.8-27B-IQ4_NL.gguf", "IQ4_NL"),
    ("model-Q6_K.gguf", "Q6_K"),
    ("model-Q8_0.gguf", "Q8_0"),
    ("model-Q4_0.gguf", "Q4_0"),
    ("model-Q3_K_S.gguf", "Q3_K_S"),
    ("gpt-oss-20b-MXFP4_MOE.gguf", "MXFP4_MOE"),
    ("model-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
    ("sub/dir/model-Q5_K_M.gguf", "Q5_K_M"),
    # 量子化らしきものが無ければファイル名そのもの。**当て推量はしない**
    ("something-else.gguf", "something-else"),
])
def test_quant_label(filename, expected):
    assert fetch.quant_label(filename) == expected


def test_version_number_is_not_mistaken_for_a_quantization():
    assert fetch.quant_label("Llama-3.1-8B-Q4_K_M.gguf") == "Q4_K_M"


def test_strip_shard():
    assert fetch.strip_shard("model-00002-of-00003") == ("model", 2)
    assert fetch.strip_shard("model") == ("model", None)


SIBLINGS = [
    {"rfilename": "README.md", "size": 100},
    {"rfilename": "M-Q4_K_M.gguf", "size": 4_000_000_000},
    {"rfilename": "M-Q8_0-00001-of-00002.gguf", "size": 5_000_000_000},
    {"rfilename": "M-Q8_0-00002-of-00002.gguf", "size": 3_000_000_000},
    {"rfilename": "mmproj-M-F16.gguf", "size": 900_000_000},
    {"rfilename": "mmproj-M-Q8_0.gguf", "size": 500_000_000},
    # 実物 (unsloth/Qwen3.8-27B-GGUF) にあった形。**本体ではない**のに
    # ラベルが Q4_0 で衝突し、一番小さいので代表に選ばれていた
    {"rfilename": "MTP/mtp-M-Q4_0.gguf", "size": 1_280_000_000},
    {"rfilename": "M-Q4_0.gguf", "size": 14_950_000_000},
]


def test_subdirectory_gguf_is_not_a_candidate():
    """MTP の draft を本体と並べると、ラベルも代表もそれに食われる."""
    body, _projs, extra = fetch.group_files(SIBLINGS)
    assert [c.files[0] for c in extra] == ["MTP/mtp-M-Q4_0.gguf"]
    assert "MTP/mtp-M-Q4_0.gguf" not in [f for c in body for f in c.files]
    # ラベルが一意であること。--pick Q4_0 が2つ当たってはいけない
    labels = [c.label for c in body]
    assert len(labels) == len(set(labels))
    assert "Q4_0" in labels


#: 実物 (unsloth/Qwen3.8-Flash-Next-GGUF)。ルートには README と
#: .gitattributes しか無く、**本体のシャードは丸ごと量子化名のフォルダの中**。
#: これを弾いていたので body が空になり「GGUF が無い」で止まっていた。
QUANT_DIR_SIBLINGS = [
    {"rfilename": ".gitattributes", "size": 1_800},
    {"rfilename": "README.md", "size": 65_100},
    {"rfilename": "UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf", "size": 10_900_000},
    {"rfilename": "UD-IQ1_S/M-UD-IQ1_S-00002-of-00003.gguf", "size": 50_000_000_000},
    {"rfilename": "UD-IQ1_S/M-UD-IQ1_S-00003-of-00003.gguf", "size": 22_500_000_000},
]


def test_quant_named_directory_is_a_candidate():
    """量子化名のフォルダに入っている本体を**取り落とさない**."""
    body, projs, extra = fetch.group_files(QUANT_DIR_SIBLINGS)
    assert len(body) == 1
    assert body[0].label == "UD-IQ1_S"
    assert len(body[0].files) == 3          # 3本まとめて1候補
    assert body[0].size_bytes == 72_510_900_000
    assert projs == [] and extra == []


def test_quant_directory_label_comes_from_the_directory():
    """ファイル名側に量子化が入っていない置き方がある.

    ``Q4_K_M/model.gguf`` のラベルを ``model`` にしてしまうと
    ``--pick Q4_K_M`` が当たらない。
    """
    body, _p, _e = fetch.group_files([
        {"rfilename": "Q4_K_M/model.gguf", "size": 18_000_000_000},
        {"rfilename": "BF16/model-00001-of-00002.gguf", "size": 45_000_000_000},
        {"rfilename": "BF16/model-00002-of-00002.gguf", "size": 45_000_000_000},
    ])
    assert sorted(c.label for c in body) == ["BF16", "Q4_K_M"]
    assert [c.label for c in fetch.match_pick(body, "Q4_K_M")] == ["Q4_K_M"]


def test_non_quant_directories_are_still_not_candidates():
    """**退行防止。**量子化ディレクトリを許しても MTP/ imatrix/ は本体でない."""
    body, _projs, extra = fetch.group_files([
        *QUANT_DIR_SIBLINGS,
        {"rfilename": "MTP/mtp-M-Q4_0.gguf", "size": 1_280_000_000},
        {"rfilename": "imatrix/imatrix.gguf", "size": 13_000_000},
        {"rfilename": "original/M.gguf", "size": 50_000_000_000},
    ])
    assert [c.label for c in body] == ["UD-IQ1_S"]
    assert sorted(c.files[0] for c in extra) == [
        "MTP/mtp-M-Q4_0.gguf", "imatrix/imatrix.gguf", "original/M.gguf",
    ]


def test_a_draft_inside_a_quant_directory_is_not_a_candidate():
    """量子化フォルダの中に置かれていても、draft は本体ではない."""
    body, _projs, extra = fetch.group_files([
        *QUANT_DIR_SIBLINGS,
        {"rfilename": "UD-IQ1_S/mtp-M.gguf", "size": 1_280_000_000},
        {"rfilename": "UD-Q4_K_XL/MTP/mtp.gguf", "size": 1_000_000_000},
    ])
    assert [c.label for c in body] == ["UD-IQ1_S"]
    assert sorted(c.files[0] for c in extra) == [
        "UD-IQ1_S/mtp-M.gguf", "UD-Q4_K_XL/MTP/mtp.gguf",
    ]


def test_mmproj_inside_a_quant_directory_is_still_mmproj():
    _body, projs, extra = fetch.group_files([
        *QUANT_DIR_SIBLINGS,
        {"rfilename": "UD-IQ1_S/mmproj-F16.gguf", "size": 900_000_000},
    ])
    assert [c.files[0] for c in projs] == ["UD-IQ1_S/mmproj-F16.gguf"]
    assert extra == []


@pytest.mark.parametrize("name", [
    "UD-IQ1_S", "Q4_K_M", "Q8_0", "IQ4_XS", "BF16", "F16", "MXFP4_MOE", "TQ1_0",
])
def test_quant_dir_re_accepts_quant_names(name):
    assert fetch.QUANT_DIR_RE.fullmatch(name)


@pytest.mark.parametrize("name", [
    "MTP", "imatrix", "original", "drafts", "docs", "M-Q4_K_M", "Q4_K_M-extra",
])
def test_quant_dir_re_rejects_everything_else(name):
    """**部分一致で拾わないこと。**``M-Q4_K_M/`` のようなフォルダは
    量子化ごとの置き場ではなくモデルごとの置き場なので、本体とは限らない。
    """
    assert not fetch.QUANT_DIR_RE.fullmatch(name)


def test_colliding_labels_get_spelled_out():
    body, _p, _e = fetch.group_files([
        {"rfilename": "a-Q4_0.gguf", "size": 10},
        {"rfilename": "b-Q4_0.gguf", "size": 20},
    ])
    assert sorted(c.label for c in body) == ["a-Q4_0", "b-Q4_0"]


def test_a_tensor_poor_file_is_not_accepted_as_the_representative():
    """実物: MTP の draft は 65層に対してテンソル 18本。本体は 866本."""
    full = _rec(20.0)
    assert fetch.looks_like_the_main_model(full)
    assert not fetch.looks_like_the_main_model({**full, "n_tensors": 2})
    assert not fetch.looks_like_the_main_model({**full, "kv_cache": None})


def test_extras_are_excluded_from_judging_but_reachable():
    """候補にしない = 要らない、ではない。draft は本体と組にして使う."""
    _body, _projs, extras = fetch.group_files([
        *SIBLINGS, {"rfilename": "imatrix/imatrix.gguf", "size": 1000},
    ])
    assert fetch.pick_extras(extras, "none") == []
    # mtp は draft/MTP と読めるものだけ。imatrix は付けない
    assert [c.files[0] for c in fetch.pick_extras(extras, "mtp")] == \
        ["MTP/mtp-M-Q4_0.gguf"]
    assert len(fetch.pick_extras(extras, "all")) == 2


@pytest.mark.parametrize("name", [
    "MTP/mtp-M-Q4_0.gguf", "M-draft-Q4_0.gguf", "drafts/M.gguf",
])
def test_mtp_file_is_recognised(name):
    assert fetch.MTP_FILE_RE.search(name)


def test_imatrix_is_not_mistaken_for_a_draft():
    assert not fetch.MTP_FILE_RE.search("imatrix/imatrix.gguf")


def test_a_file_too_small_to_be_weights_is_not_a_candidate():
    """実物: unsloth は imatrix_unsloth.gguf (13 MiB) を**ルートに**置いている.

    置き場所では弾けない。27B のパラメータ数に対して 0.004 bpw なので、
    重みではありえない。--spread の下端としてこれが選ばれていた。
    """
    params_27b = 27_320_000_000
    assert fetch.implausible_as_model(13 * 1024 ** 2, params_27b)
    # 実在する一番軽い量子化 (IQ1_S 1.81 bpw) は残す
    assert not fetch.implausible_as_model(5_770_000_000 * 1.074, params_27b)
    assert not fetch.implausible_as_model(50_900_000_000 * 1.074, params_27b)  # BF16
    # パラメータ数が読めていなければ判断しない
    assert not fetch.implausible_as_model(13 * 1024 ** 2, 0)


def test_the_imatrix_file_never_reaches_the_download_list(
        hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "40", "--fit", "--spread",
        "--top", "3", "--mmproj", "none", "--json"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert "imatrix_unsloth.gguf" not in out["selected"]
    assert "imatrix_unsloth" not in [c["label"] for c in out["candidates"]]


def test_group_files_merges_shards_and_splits_mmproj():
    body, projs, _extra = fetch.group_files(SIBLINGS)
    assert [c.label for c in body] == ["Q4_K_M", "Q8_0", "Q4_0"]
    q8 = body[1]
    # 分割は1つの候補。**サイズは合計**でなければ「載る」と誤判定する
    assert len(q8.files) == 2
    assert q8.size_bytes == 8_000_000_000
    assert [c.size_bytes for c in projs] == [500_000_000, 900_000_000]
    assert all(c.mmproj for c in projs)


def test_mmproj_auto_takes_the_smallest_one():
    _body, projs, _extra = fetch.group_files(SIBLINGS)
    assert [c.files for c in fetch.pick_mmproj(projs, "auto")] == \
        [("mmproj-M-Q8_0.gguf",)]
    assert len(fetch.pick_mmproj(projs, "all")) == 2
    assert fetch.pick_mmproj(projs, "none") == []


# --- bpw / 絞り込み --------------------------------------------------------

def test_bpw_is_the_measured_bit_width_not_the_name():
    """検算: 16 bit の重み n 個は n*2 バイト → ちょうど 16.00 bpw.

    実測でも BF16 は Qwen3.8-27B で 16.00、Ornith-1.5-35B で 16.01 になった。
    ここがずれていたらパラメータ数の数え方が間違っている。
    """
    assert fetch.bits_per_weight(2000, 1000) == 16.0
    assert fetch.bits_per_weight(1000, 0) is None
    # 実物の値: UD-Q6_K_XL は名前に 6 と入っているが 7.41 bpw だった
    assert round(fetch.bits_per_weight(25_296_000_000, 27_320_000_000), 2) == 7.41


def test_parse_header_counts_parameters():
    header = _ggufhdr.parse_header(build_gguf(MODEL_META, MODEL_TENSORS))
    # build_gguf は全テンソルを 8x8 で書く
    assert header.n_params == 64 * len(MODEL_TENSORS)


def test_spread_picks_across_the_range_not_the_top_three():
    """実物の --top 3 は 8.2 / 8.5 / 9.2 bpw を返した。3本とも同じビット帯."""
    verdicts = []
    for i, size in enumerate([8, 12, 16, 20, 24]):
        rec = {**_rec(float(size)), "n_params": 20_000_000_000}
        verdicts.append(fetch.Verdict(
            _cand(f"q{i}", float(size)), rec, "f16", 0, 0, 0, True, "ctx"))
    top = [v.cand.label for v in fetch.choose(verdicts, 3)]
    spread = [v.cand.label for v in fetch.choose(verdicts, 3, spread=True)]
    assert top == ["q4", "q3", "q2"]          # 上から3本 = 隣どうし
    assert spread == ["q4", "q2", "q0"]       # 端と真ん中 (大きい順のまま)
    assert spread[0] != spread[-1]


def test_min_bpw_drops_the_light_end():
    verdicts = []
    for i, size in enumerate([2, 12, 24]):
        rec = {**_rec(float(size)), "n_params": 20_000_000_000}
        verdicts.append(fetch.Verdict(
            _cand(f"q{i}", float(size)), rec, "f16", 0, 0, 0, True, "ctx"))
    # 2 GB / 20B params = 0.8 bpw。1bit 量子化は比較の相手にならない
    assert [v.cand.label for v in fetch.choose(verdicts, 3, min_bpw=4.0)] == \
        ["q2", "q1"]


def test_only_and_exclude_are_globs_over_label_and_filename():
    body, _p, _e = fetch.group_files(SIBLINGS)
    assert [c.label for c in fetch.filter_candidates(body, ["Q8*"], None)] == ["Q8_0"]
    assert [c.label for c in fetch.filter_candidates(body, None, ["Q4*"])] == ["Q8_0"]
    # 大文字小文字は無視する
    assert fetch.filter_candidates(body, ["q4_k_m"], None)[0].label == "Q4_K_M"
    assert fetch.filter_candidates(body, ["nope*"], None) == []


def test_calibration_measured_on_another_model_is_flagged():
    """実際に起きた: Qwen で測った 69.1 KB/token が Ornith の計画に使われ、
    最大 ctx が3倍近く低く出た。**数字は静かに出るので気づけない**."""
    qwen_like = {**_rec(20.0)}
    qwen_like["kv_cache"] = {**qwen_like["kv_cache"], "bytes_per_token_f16": 69632}
    ornith_like = {**_rec(20.0)}
    ornith_like["kv_cache"] = {**ornith_like["kv_cache"], "bytes_per_token_f16": 22528}
    cfg = {"kv_f16_bytes": 70720, "kv_derived_f16_bytes": 69632,
           "kv_measured_on": "Qwen3.8-27B-Q5_K_M.gguf"}

    assert calibration_mismatch(qwen_like, cfg) is None       # 本人には言わない
    warned = calibration_mismatch(ornith_like, cfg, "ja")
    assert warned and "Qwen3.8-27B-Q5_K_M.gguf" in warned

    # 記録が無い古い設定では黙る。比べる相手が無いのに警告するのは当て推量
    assert calibration_mismatch(ornith_like, {"kv_f16_bytes": 70720}) is None
    # 較正していなければ関係ない
    assert calibration_mismatch(ornith_like, {"kv_derived_f16_bytes": 69632}) is None


def test_pick_prefers_an_exact_label():
    body, _p, _e = fetch.group_files(SIBLINGS)
    assert [c.label for c in fetch.match_pick(body, "q8_0")] == ["Q8_0"]
    # 部分一致は当たったもの全部。Q4 は Q4_K_M と Q4_0 の両方に当たる
    assert [c.label for c in fetch.match_pick(body, "Q4")] == ["Q4_K_M", "Q4_0"]
    # 完全一致があればそちらが勝つ
    assert [c.label for c in fetch.match_pick(body, "Q4_0")] == ["Q4_0"]
    assert fetch.match_pick(body, "IQ2_XXS") == []


# --- 判定 -----------------------------------------------------------------

def _rec(size_gb: float) -> dict:
    header = _ggufhdr.parse_header(build_gguf(MODEL_META, MODEL_TENSORS))
    return fetch.record_from_header(header, "m.gguf", int(size_gb * 1e9))


def _cand(label: str, size_gb: float) -> fetch.Candidate:
    return fetch.Candidate(label, (f"m-{label}.gguf",), int(size_gb * 1e9), False)


def test_evaluate_uses_the_kv_cache_not_just_the_file_size():
    rec = _rec(20.0)
    assert rec["kv_cache"]["kv_bearing_layers"] == 3      # 6層のうち3層だけ
    assert rec["kv_cache"]["bytes_per_token_f16"] == 3 * 4 * (128 + 128) * 2
    v = fetch.evaluate(_cand("Q4_K_M", 20.0), rec, 24.0, 1.0, "auto", 16384)
    assert v.fits and v.kv_mode == "f16"
    assert v.ctx_f16 > 0

    # ファイルだけで予算を食い切る (26.0 GB = 24.2 GiB)
    tight = fetch.evaluate(_cand("Q8_0", 26.0), _rec(26.0), 24.0, 1.0, "auto", 16384)
    assert not tight.fits
    assert tight.ctx_f16 == 0


def test_the_candidate_size_wins_over_the_records_size():
    """レコードは代表1本の複製。**差し替え忘れを判定に持ち込ませない**."""
    small = _rec(20.0)                       # 20 GB のつもりのレコード
    v = fetch.evaluate(_cand("Q8_0", 26.0), small, 24.0, 1.0, "f16", 16384)
    assert not v.fits                        # 判定するのは 26 GB のほう


def test_min_ctx_is_what_makes_it_count_as_fitting():
    """載っても ctx が伸びないなら、それは「載る」と言わない."""
    rec = _rec(24.4)
    loose = fetch.evaluate(_cand("Q5", 24.4), rec, 24.0, 1.0, "f16", 4096)
    strict = fetch.evaluate(_cand("Q5", 24.4), rec, 24.0, 1.0, "f16", 131072)
    assert loose.fits
    assert not strict.fits
    assert loose.ctx_f16 == strict.ctx_f16   # 同じ数字で、線の引き方だけが違う


def test_without_a_header_the_verdict_says_it_is_size_only():
    v = fetch.evaluate(_cand("Q4", 20.0), None, 24.0, 1.0, "auto", 16384)
    assert v.reason == "size"
    assert v.fits          # 20 + 1 <= 24。KV は入っていない
    assert v.kv_mode is None


def test_no_budget_means_unknown_not_no():
    """予算が分からないのに「入らない」と書くと、嘘をつくことになる."""
    v = fetch.evaluate(_cand("Q4", 20.0), _rec(20.0), None, 1.0, "auto", 16384)
    assert v.reason == "unknown"
    assert not v.fits


def test_choose_takes_the_largest_that_fit():
    verdicts = [
        fetch.Verdict(_cand("Q4", 10.0), None, "f16", 0, 0, 0, True, "ctx"),
        fetch.Verdict(_cand("Q5", 14.0), None, "f16", 0, 0, 0, True, "ctx"),
        fetch.Verdict(_cand("Q6", 18.0), None, "f16", 0, 0, 0, True, "ctx"),
        fetch.Verdict(_cand("Q8", 25.0), None, "f16", 0, 0, 0, False, "ctx"),
    ]
    assert [v.cand.label for v in fetch.choose(verdicts, 2)] == ["Q6", "Q5"]
    assert [v.cand.label for v in fetch.choose(verdicts, 9)] == ["Q6", "Q5", "Q4"]


def test_rec_for_size_keeps_kv_but_drops_the_quant_mix():
    """量子化の内訳は代表1本のもの。持ち越すと Q8 の行に Q4 の内訳が出る."""
    base = _rec(20.0)
    clone = fetch.rec_for_size(base, _cand("Q8_0", 35.0))
    assert clone["kv_cache"] == base["kv_cache"]
    assert clone["context_length"] == base["context_length"]
    assert clone["size_gb"] == 35.0
    assert "weight_mix" not in clone
    assert "dominant_weight_type" not in clone


# --- hf download の組み立て -----------------------------------------------

def test_download_command_passes_file_names_positionally(tmp_path):
    cmd = fetch.download_command("hf", "org/repo", ["a.gguf", "b.gguf"],
                                 tmp_path, "main")
    assert cmd[:3] == ["hf", "download", "org/repo"]
    assert "a.gguf" in cmd and "b.gguf" in cmd
    assert "--local-dir" in cmd
    assert "--revision" not in cmd          # main のときは付けない


def test_download_command_carries_a_non_default_revision(tmp_path):
    cmd = fetch.download_command("hf", "org/repo", ["a.gguf"], tmp_path, "v2")
    assert cmd[-2:] == ["--revision", "v2"]


# --- 表示幅 ---------------------------------------------------------------

def test_japanese_counts_as_two_columns():
    assert width("abc") == 3
    assert width("入らない") == 8
    assert len(pad("入らない", 12)) == len("入らない") + 4


# --- ローカル HTTP でのやりとり -------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Hugging Face の必要な部分だけを真似る."""

    blobs: ClassVar[dict[str, bytes]] = {}
    api: ClassVar[dict] = {}

    def log_message(self, *_args):  # テスト出力を汚さない
        pass

    def do_GET(self):
        if self.path.startswith("/api/models/"):
            body = json.dumps(self.api).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        name = self.path.rsplit("/", 1)[-1]
        blob = self.blobs.get(name)
        if blob is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng and "ignore" not in name:
            start, end = rng.removeprefix("bytes=").split("-")
            chunk = blob[int(start):int(end) + 1]
            self.send_response(206)
        else:
            # Range を無視して全部返すサーバの再現
            chunk = blob
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def hf_server(monkeypatch):
    gguf = build_gguf(MODEL_META, MODEL_TENSORS, dims=BIG_DIMS)
    _Handler.blobs = {
        "M-Q4_K_M.gguf": gguf,
        "M-Q8_0.gguf": gguf,
        "mmproj-M-F16.gguf": gguf,
        "mtp-M-Q4_0.gguf": gguf,
        "imatrix_unsloth.gguf": gguf,
        "ignore-range.gguf": gguf,
    }
    _Handler.api = {"siblings": [
        {"rfilename": "M-Q4_K_M.gguf", "size": 20_000_000_000},
        {"rfilename": "M-Q8_0.gguf", "size": 35_000_000_000},
        {"rfilename": "mmproj-M-F16.gguf", "size": 900_000_000},
        {"rfilename": "MTP/mtp-M-Q4_0.gguf", "size": 1_280_000_000},
        # 実物 (unsloth) と同じく**ルートに**置いてある imatrix。
        # 置き場所では弾けないので bpw で弾く
        {"rfilename": "imatrix_unsloth.gguf", "size": 13_000_000},
        {"rfilename": "README.md", "size": 10},
    ]}
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HF_TOKEN", "")
    yield server
    server.shutdown()


def test_read_range_refuses_a_server_that_ignores_range(hf_server):
    """ここで止めないと、21 GB を黙って受け取ることになる."""
    url = fetch.file_url("org/repo", "main", "ignore-range.gguf")
    with pytest.raises(OSError, match="Range"):
        fetch.read_range(url, 100, 200)


def test_fetch_header_asks_for_more_when_the_first_chunk_is_short(hf_server):
    url = fetch.file_url("org/repo", "main", "M-Q4_K_M.gguf")
    header, used = fetch.fetch_header(url, first=64)
    assert header.metadata["general.architecture"] == "testarch"
    assert used >= 64                 # 追加で取りにいった


def test_fetch_header_gives_up_instead_of_downloading_the_model(hf_server):
    """上限を切っておかないと、壊れたファイルで本体を全部落としにいく."""
    url = fetch.file_url("org/repo", "main", "M-Q4_K_M.gguf")
    with pytest.raises(OSError, match="did not end"):
        fetch.fetch_header(url, first=16, limit=64)


def _no_gpu_machine():
    """GPU が1枚も無いマシン.

    **本物の Hardware を使う。**スタブを自作すると、_hardware に生えたメソッドを
    fetch が呼び始めたときにテストだけが素通りする（実際 budget_warnings を
    足したときにそれが起きた）。
    """
    return _hardware.Hardware(gpus=[], ram_gib=64.0, physical_cores=8,
                              logical_cores=16, unified_memory=False)


# --- gguf-plan と食い違わせない -------------------------------------------

def test_llama_server_resolution_is_shared_with_gguf_plan():
    """片方だけが ROCm ビルドを見ると、同じマシンで予算の数字が食い違う."""
    # 単数形しか書いていない人も、複数形で書いた人も、両方拾う
    assert _config.resolve_llama_servers(None, {}).value == ["llama-server"]
    assert _config.resolve_llama_servers(
        None, {"llama_server": "/opt/cuda/llama-server"}
    ).value == ["/opt/cuda/llama-server"]
    assert _config.resolve_llama_servers(
        None, {"llama_servers": ["/a/llama-server", "/b/llama-server"]}
    ).value == ["/a/llama-server", "/b/llama-server"]
    # CLI は設定より強い
    assert _config.resolve_llama_servers(["/cli/llama-server"], {"llama_server": "/x"}
                                         ).value == ["/cli/llama-server"]


def test_repeated_flags_and_commas_flatten_the_same_way():
    assert _config.split_repeated(["a", "b,c"]) == ["a", "b", "c"]
    assert _config.split_repeated([]) is None
    assert _config.split_repeated(None) is None


def _mixed_backend_machine():
    """実機: CUDA の 5090 / 3090 と、ROCm から見える APU 8060S が同居.

    容量だけで選ぶと **96 GiB の APU** が勝つが、生成速度では 5090 に負ける。
    ここで黙って APU を選ぶと、利用者は理由を知らないまま遅いほうで回す。
    """
    return _hardware.Hardware(
        gpus=[
            _hardware.Gpu(0, "NVIDIA GeForce RTX 5090", 31.4, 0.4,
                          device_id="CUDA0", free_gib=31.0),
            _hardware.Gpu(1, "NVIDIA GeForce RTX 3090", 23.6, 0.3,
                          device_id="CUDA1", free_gib=23.3),
            _hardware.Gpu(2, "AMD Radeon 8060S Graphics", 96.0, 0.2,
                          device_id="ROCm0", free_gib=95.8),
        ],
        ram_gib=128.0, physical_cores=16, logical_cores=32, unified_memory=False)


def test_mixed_backends_are_flagged_before_any_transfer(
        hf_server, monkeypatch, tmp_path, capsys):
    """gguf-plan が言うことを gguf-fetch が黙っていてはいけない."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect",
                        lambda _b: _mixed_backend_machine())
    monkeypatch.setattr("sys.argv", ["gguf-fetch", "org/repo", "--json"])
    assert fetch.main() == 0
    captured = capsys.readouterr()
    assert "ROCm" in captured.err and "CUDA" in captured.err
    assert "8060S" in captured.err
    # 予算は一番大きいデバイスから取られている
    assert json.loads(captured.out)["vram_gib"] == 96.0


def test_enclosing_git_repo_walks_up(tmp_path):
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "a" / "b" / "models"
    assert fetch.enclosing_git_repo(deep) == tmp_path.resolve()


def test_enclosing_git_repo_handles_a_dot_git_file(tmp_path):
    """worktree や submodule では .git は**ファイル**。ディレクトリ決め打ちは駄目."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert fetch.enclosing_git_repo(tmp_path / "models") == tmp_path.resolve()


def test_warns_when_models_would_land_in_a_source_checkout(
        hf_server, monkeypatch, tmp_path, capsys):
    """--dir の付け忘れで 72GB がチェックアウトに落ちる。.gitignore が隠すので
    git status では気づけない."""
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch.subprocess, "run", _never_called)
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--dry-run"])
    assert fetch.main() == 0
    assert "git" in capsys.readouterr().err


def test_no_such_warning_outside_a_repo(hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch.subprocess, "run", _never_called)
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--dry-run",
        "--dir", str(tmp_path / "models")])
    assert fetch.main() == 0
    assert "models_dir" not in capsys.readouterr().err


def test_dry_run_still_shows_the_command_when_the_disk_is_too_small(
        hf_server, monkeypatch, tmp_path, capsys):
    """「落とさずコマンドだけ見せろ」に、空き容量を理由に黙るのは答えではない."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch.subprocess, "run", _never_called)
    monkeypatch.setattr(fetch, "disk_free_gib", lambda _p: 1.0)
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--dry-run",
        "--dir", str(tmp_path / "models")])
    assert fetch.main() == 1              # 足りないことは終了コードに残す
    out = capsys.readouterr().out
    assert "hf download org/repo" in out  # それでもコマンドは出す


def test_a_real_run_stops_when_the_disk_is_too_small(
        hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch.subprocess, "run", _never_called)
    monkeypatch.setattr(fetch, "disk_free_gib", lambda _p: 1.0)
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "-y",
        "--dir", str(tmp_path / "models")])
    assert fetch.main() == 1
    assert "hf download" not in capsys.readouterr().out


def test_show_config_says_where_the_budget_came_from(monkeypatch, tmp_path, capsys):
    """予算はデバイスの容量から取る。**device を伏せたら説明になっていない**."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch._hardware, "render", lambda _hw: "detected hardware\n  GPU  none")
    monkeypatch.setattr("sys.argv", ["gguf-fetch", "--show-config"])
    assert fetch.main() == 0
    out = capsys.readouterr().out
    for key in ("lang", "vram", "overhead", "device", "llama_servers",
                "models_dir", "hf_bin"):
        assert key in out
    assert "detected hardware" in out


def test_end_to_end_json(hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)          # リポジトリの gguf-fit.toml を拾わせない
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--json"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)

    assert [c["label"] for c in out["candidates"]] == ["Q4_K_M", "Q8_0"]
    q4, q8 = out["candidates"]
    assert q4["fits"] and q4["basis"] == "ctx"
    assert not q8["fits"]
    # mmproj は既定で付く
    assert out["selected"] == ["M-Q4_K_M.gguf", "mmproj-M-F16.gguf"]
    # 判定に使ったのはヘッダぶんだけ。**35 GB は触っていない**
    assert 0 < out["header_bytes_transferred"] < 1_000_000


def test_pick_downloads_only_that_one(hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--pick", "Q8_0", "--json", "--mmproj", "none"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == ["M-Q8_0.gguf"]


def test_extras_mtp_rides_along_with_the_model(hf_server, monkeypatch,
                                               tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--top", "1",
        "--mmproj", "none", "--extras", "mtp", "--json"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert "MTP/mtp-M-Q4_0.gguf" in out["selected"]
    # 判定の表には出さない。本体ではないので「載るか」の話に混ぜない
    assert "Q4_0" not in [c["label"] for c in out["candidates"]
                          if c["files"] == ["MTP/mtp-M-Q4_0.gguf"]]
    assert any(e["mtp"] for e in out["extras"])


def test_pick_can_name_a_file_that_is_not_a_candidate(hf_server, monkeypatch,
                                                      tmp_path, capsys):
    """「要るなら --pick で名指しして」と案内する以上、当たらないと嘘になる."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--pick", "mtp", "--mmproj", "none", "--json"])
    assert fetch.main() == 0
    assert json.loads(capsys.readouterr().out)["selected"] == \
        ["MTP/mtp-M-Q4_0.gguf"]


def test_all_takes_every_gguf_but_not_the_readme(hf_server, monkeypatch,
                                                 tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr("sys.argv", ["gguf-fetch", "org/repo", "--all", "--json"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == ["M-Q4_K_M.gguf", "M-Q8_0.gguf", "mmproj-M-F16.gguf"]


def test_dry_run_prints_the_command_and_downloads_nothing(
        hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _no_gpu_machine())
    monkeypatch.setattr(fetch.subprocess, "run", _never_called)
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--vram", "24", "--fit", "--dry-run",
        "--dir", str(tmp_path)])
    assert fetch.main() == 0
    out = capsys.readouterr().out
    assert "hf download org/repo M-Q4_K_M.gguf" in out
    assert "--local-dir" in out


def _never_called(*_args, **_kwargs):
    raise AssertionError("--dry-run must not run hf download")
