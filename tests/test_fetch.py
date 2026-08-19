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

from gguf_fit import _ggufhdr, fetch
from gguf_fit._messages import pad, width

# --- 合成 GGUF ------------------------------------------------------------

T_UINT32, T_FLOAT32, T_STRING, T_ARRAY = 4, 6, 8, 9


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(meta: list[tuple[str, int, bytes]],
               tensors: list[tuple[str, int]], trailer: int = 4096) -> bytes:
    """テストで使う最小の GGUF。``trailer`` は「本体のつもり」の埋め草."""
    out = [b"GGUF", struct.pack("<I", 3),
           struct.pack("<Q", len(tensors)), struct.pack("<Q", len(meta))]
    for key, vtype, payload in meta:
        out += [_gstr(key), struct.pack("<I", vtype), payload]
    for name, code in tensors:
        out += [_gstr(name), struct.pack("<I", 2),
                struct.pack("<QQ", 8, 8), struct.pack("<I", code),
                struct.pack("<Q", 0)]
    return b"".join(out) + b"\0" * trailer


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
]


def test_group_files_merges_shards_and_splits_mmproj():
    body, projs = fetch.group_files(SIBLINGS)
    assert [c.label for c in body] == ["Q4_K_M", "Q8_0"]
    q8 = body[1]
    # 分割は1つの候補。**サイズは合計**でなければ「載る」と誤判定する
    assert len(q8.files) == 2
    assert q8.size_bytes == 8_000_000_000
    assert [c.size_bytes for c in projs] == [500_000_000, 900_000_000]
    assert all(c.mmproj for c in projs)


def test_mmproj_auto_takes_the_smallest_one():
    _body, projs = fetch.group_files(SIBLINGS)
    assert [c.files for c in fetch.pick_mmproj(projs, "auto")] == \
        [("mmproj-M-Q8_0.gguf",)]
    assert len(fetch.pick_mmproj(projs, "all")) == 2
    assert fetch.pick_mmproj(projs, "none") == []


def test_pick_prefers_an_exact_label():
    body, _ = fetch.group_files(SIBLINGS)
    assert [c.label for c in fetch.match_pick(body, "q8_0")] == ["Q8_0"]
    assert [c.label for c in fetch.match_pick(body, "Q4")] == ["Q4_K_M"]
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
    gguf = build_gguf(MODEL_META, MODEL_TENSORS)
    _Handler.blobs = {
        "M-Q4_K_M.gguf": gguf,
        "M-Q8_0.gguf": gguf,
        "mmproj-M-F16.gguf": gguf,
        "ignore-range.gguf": gguf,
    }
    _Handler.api = {"siblings": [
        {"rfilename": "M-Q4_K_M.gguf", "size": 20_000_000_000},
        {"rfilename": "M-Q8_0.gguf", "size": 35_000_000_000},
        {"rfilename": "mmproj-M-F16.gguf", "size": 900_000_000},
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


class _FakeHw:
    def suggested_device(self):
        return None

    def suggested_vram_gib(self, _device=None):
        return None

    def suggested_threads(self):
        return None


def test_end_to_end_json(hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)          # リポジトリの gguf-fit.toml を拾わせない
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _FakeHw())
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
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _FakeHw())
    monkeypatch.setattr("sys.argv", [
        "gguf-fetch", "org/repo", "--pick", "Q8_0", "--json", "--mmproj", "none"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == ["M-Q8_0.gguf"]


def test_all_takes_every_gguf_but_not_the_readme(hf_server, monkeypatch,
                                                 tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _FakeHw())
    monkeypatch.setattr("sys.argv", ["gguf-fetch", "org/repo", "--all", "--json"])
    assert fetch.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == ["M-Q4_K_M.gguf", "M-Q8_0.gguf", "mmproj-M-F16.gguf"]


def test_dry_run_prints_the_command_and_downloads_nothing(
        hf_server, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _FakeHw())
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
