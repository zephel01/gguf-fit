"""Hugging Face から GGUF を落とす。**落とす前に、載るかどうかを決める**.

    gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF              # 一覧と判定だけ
    gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --fit        # 載るものを上から N 本
    gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --pick Q5_K_M
    gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --all

``hf download`` そのものは既に十分よくできている。足りないのは
**「このリポジトリの12本のうち、自分の 24 GiB に載るのはどれか」**という判断で、
それは普通、全部落としてから `gguf-probe` で調べることになる。
21 GB を5本落として4本消す、というのがいちばんありがちな失敗。

このコマンドは順番を入れ替える。

  1. HF の API でファイル一覧とサイズを取る（**ここは数 KB**）
  2. 代表1本の GGUF ヘッダだけを HTTP Range で取る（**実測 12.0 MiB**）
  3. `gguf-plan` と同じ式で「載るか / 最大 ctx はいくつか」を出す
  4. 決まったものだけを ``hf download`` に渡す

実測: Ornith-1.5-35B（5量子化・合計 172 GB）の判定に要した転送は **12.0 MiB**。

--- なぜ代表1本でいいのか -------------------------------------------------

KV キャッシュの単価は**層の構造**で決まる。量子化を変えても、テンソルの
**名前**は変わらない（型が変わるだけ）。つまり同じリポジトリの Q4 と Q8 は
KV/token も native ctx も MTP の有無も同じで、**違うのはファイルサイズだけ**。
だから代表1本のヘッダを読めば、残りはサイズを差し替えるだけで判定できる。

この仮定を隠さないために、出力には**どのファイルから読んだか**を書く。
仮定を置きたくないときは ``--probe all`` で全部のヘッダを読む
（本数ぶん転送が増える）。``--probe none`` ならヘッダを読まず、
**ファイルサイズだけ**の粗い判定になる（そうと分かる形で出す）。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

from . import _hardware
from ._config import (
    drop_detectable,
    load_config,
    render_show_config,
    resolve,
    resolve_llama_servers,
    split_repeated,
)
from ._ggufhdr import Header, TruncatedGGUF, parse_header
from ._messages import DEFAULT_LANG, pad, t, width
from .plan import (
    DEFAULT_OVERHEAD_GIB,
    GIB,
    KvRates,
    budget_warnings,
    calibration_mismatch,
    max_ctx,
)
from .probe import summarize_tensors

#: HF のエンドポイント。ミラーを使う人がいるので環境変数を見る
#: （``huggingface_hub`` 自身も同じ名前を見る）
DEFAULT_ENDPOINT = "https://huggingface.co"

#: ヘッダ取得の最初の1回。実測 Ornith-1.5-35B のヘッダは 10.48 MiB だったので、
#: 1回で当たるところに置いてある。足りなければ倍にして取り直す
HEADER_FIRST_BYTES = 12 * 1024 * 1024
#: ヘッダ取得の上限。ここを超えたら「ヘッダが異常に大きい」として諦める
HEADER_MAX_BYTES = 192 * 1024 * 1024

#: ``--fit`` で落とす本数の既定。1本だけだと比較ができず、全部だと意味が無い
DEFAULT_TOP = 3
#: ``--fit`` が「載る」と認める最低の ctx。これを割るなら載っても使い道が薄い
DEFAULT_MIN_CTX = 16384

#: GGUF のヘッダから拾うメタデータ。``gguf-probe`` と同じ対応表
_META_KEYS: tuple[tuple[str, str], ...] = (
    (".context_length", "context_length"),
    (".block_count", "block_count"),
    (".attention.head_count", "head_count"),
    (".attention.head_count_kv", "head_count_kv"),
    (".attention.key_length", "key_length"),
    (".attention.value_length", "value_length"),
    (".embedding_length", "embedding_length"),
    (".rope.freq_base", "rope_freq_base"),
    (".rope.scaling.type", "rope_scaling_type"),
    (".rope.scaling.factor", "rope_scaling_factor"),
    (".attention.sliding_window", "sliding_window"),
)
_WANT_META: tuple[str, ...] = (
    *(k for k, _ in _META_KEYS),
    "general.architecture", "general.file_type", "tokenizer.chat_template",
)

#: 分割 GGUF の連番。``-00001-of-00003.gguf``
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")

#: ファイル名から量子化ラベルを取り出す。**長いものを先に並べること**
#: （``Q4_K_XL`` を ``Q4_K`` で切ってしまわないように）。
#: 実物で確かめた形: Q4_K_M / Q6_K / Q8_0 / IQ4_XS / IQ4_NL / UD-Q5_K_XL /
#: BF16 / F16 / MXFP4_MOE / TQ1_0
_QUANT_ALT = (
    r"(?:UD-)?(?:"
    r"IQ\d+_[A-Z]+(?:_[A-Z]+)?"
    r"|Q\d+_[KS](?:_[A-Z]+)?"
    r"|Q\d+_\d+"
    r"|MXFP\d+(?:_MOE)?"
    r"|NVFP\d+"
    r"|TQ\d+_\d+"
    r"|BF16|F16|F32"
    r")"
)
QUANT_RE = re.compile(
    r"(?:^|[-_.])"
    r"(?P<label>" + _QUANT_ALT + r")(?=$|[-_.])",
    re.IGNORECASE,
)

#: **ディレクトリ名そのもの**が量子化ラベルか。``fullmatch`` で使う。
#: 実物: unsloth の新しい置き方は量子化ごとにフォルダを切る
#: (``UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf``)。
#: ``MTP`` / ``imatrix`` / ``original`` は**一致しない**ので、
#: 「本体ではないサブディレクトリ」との区別がこれ1本で付く。
QUANT_DIR_RE = re.compile(_QUANT_ALT, re.IGNORECASE)

#: ビジョン投影（mmproj）。本体とは別枠で扱う
MMPROJ_RE = re.compile(r"(^|/)mmproj", re.IGNORECASE)

#: 投機デコード用の draft / MTP ファイル。本体と**組にして**使うもの。
#: 実物: unsloth の ``MTP/mtp-Qwen3.8-27B-Q4_0.gguf``
MTP_FILE_RE = re.compile(r"(^|[/\-_])(mtp|draft)", re.IGNORECASE)


# --------------------------------------------------------------------------
# ファイル名の解釈（純粋。ネットワークに触らない）
# --------------------------------------------------------------------------

def strip_shard(stem: str) -> tuple[str, int | None]:
    """``foo-00002-of-00003`` を ``("foo", 2)`` にする。連番でなければ ``None``."""
    m = SHARD_RE.search(stem)
    if not m:
        return stem, None
    return stem[:m.start()], int(m.group(1))


def quant_label(filename: str) -> str:
    """ファイル名から量子化ラベルを取る。取れなければファイル名そのもの.

    **最後に出てきたものを採る。**``Llama-3.1-8B-Q4_K_M`` の ``3.1`` を
    量子化と読み違えないため（数字だけの並びは候補に入れていないが、
    将来ゆるめたときに効く）。
    """
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    stem, _ = strip_shard(stem)
    matches = list(QUANT_RE.finditer(stem))
    if not matches:
        return stem
    return matches[-1].group("label")


def is_mmproj(filename: str) -> bool:
    return bool(MMPROJ_RE.search(filename))


class Candidate(NamedTuple):
    """落とす単位。分割されていれば複数ファイルで1つ."""

    label: str
    files: tuple[str, ...]
    size_bytes: int
    mmproj: bool

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB


def uniquify(cands: list[Candidate]) -> list[Candidate]:
    """ラベルが衝突したら、ファイル名で区別できるところまで伸ばす.

    ``--pick Q4_0`` が2つ当たる状態を残さない。表に同じ名前の行が2つ並ぶのも
    「どちらの話をしているのか」が言えなくなるので許さない。
    """
    seen: dict[str, int] = {}
    for c in cands:
        seen[c.label] = seen.get(c.label, 0) + 1
    out = []
    for cand in cands:
        if seen[cand.label] == 1:
            out.append(cand)
            continue
        stem = cand.files[0].rsplit("/", 1)[-1]
        out.append(cand._replace(
            label=stem[:-5] if stem.lower().endswith(".gguf") else stem))
    return out


def group_files(
    siblings: list[dict],
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """API のファイル一覧を「本体の候補」「mmproj」「その他」に仕分ける.

    純粋関数。``siblings`` は ``{"rfilename": ..., "size": ...}`` の並び。
    分割 GGUF は1つの候補にまとめ、サイズは合計する。

    **サブディレクトリの GGUF は本体の候補にしない。**実物で踏んだ:
    unsloth/Qwen3.8-27B-GGUF には ``MTP/mtp-Qwen3.8-27B-Q4_0.gguf`` (1.28 GiB)
    が置いてある。これは投機デコード用の**別ファイル**で本体ではないのに、

      * ``Q4_0`` というラベルが本物の ``Qwen3.8-27B-Q4_0.gguf`` (14.95 GiB) と
        衝突し、``--pick Q4_0`` が2つ当たる
      * 一番小さいので**代表として選ばれ**、その 4.0 KB/token (KV層 1/65) が
        全行に当たった。本物は 68.0 KB/token (17/65) で、**17倍ずれる**

    表は「それらしく」出る。だから気づかない。

    **ただし「サブディレクトリ = 本体ではない」は言い過ぎだった。**実物:
    unsloth/Qwen3.8-Flash-Next-GGUF は本体のシャード3本を丸ごと
    ``UD-IQ1_S/`` に入れている (ルートには README と .gitattributes しか無い)。
    これを一律に弾くと body が空になり、``fetch_no_gguf`` で止まる —
    **ファイルは目の前にあるのにダウンロードできない**。

    そこで、**ディレクトリ名そのものが量子化ラベルなら本体**とみなす
    (``UD-IQ1_S/`` ``Q4_K_M/`` ``BF16/``)。``MTP`` ``imatrix`` ``original``
    は量子化名として読めないので、これまでどおり extras に落ちる。
    量子化ディレクトリの中に入っていても ``mtp``/``draft`` と読める
    ファイルは本体にしない。
    """
    groups: dict[str, dict[str, Any]] = {}
    for entry in siblings:
        name = entry.get("rfilename") or ""
        if not name.lower().endswith(".gguf"):
            continue
        size = int(entry.get("size") or 0)
        stem = name[:-5]
        base, _shard = strip_shard(stem)
        slot = groups.setdefault(base, {"files": [], "size": 0, "name": name})
        slot["files"].append(name)
        slot["size"] += size

    body: list[Candidate] = []
    proj: list[Candidate] = []
    extra: list[Candidate] = []
    for base, slot in groups.items():
        files = tuple(sorted(slot["files"]))
        mm = is_mmproj(base)
        dirname, _, stem = base.rpartition("/")
        # ラベルは**ディレクトリ名を優先**する。量子化ごとにフォルダを切る
        # 置き方では、ファイル名側に量子化が入っていないことがある
        # (``Q4_K_M/model.gguf`` の quant_label は "model" になってしまう)
        promoted = (
            bool(dirname)
            and QUANT_DIR_RE.fullmatch(dirname) is not None
            and MTP_FILE_RE.search("/" + stem) is None
        )
        label = dirname if (promoted and not mm) else quant_label(base)
        cand = Candidate(label, files, slot["size"], mm)
        if dirname and not promoted:
            extra.append(cand)      # MTP/ imatrix/ original/ ... 本体ではない
        elif mm:
            proj.append(cand)
        else:
            body.append(cand)
    for group in (body, proj, extra):
        group.sort(key=lambda c: c.size_bytes)
    return uniquify(body), proj, extra


def looks_like_the_main_model(rec: dict) -> bool:
    """代表として KV の単価を借りてよいレコードか.

    本体には**層ごとに何本もテンソルがある**。65層のモデルで 18本しか無ければ、
    それは本体ではない (実物: MTP の draft は 18本、本体は 866本)。
    ``n_tensors >= block_count`` は緩い線だが、桁で外れているものは確実に弾ける。
    """
    if not rec.get("is_language_model") or not rec.get("kv_cache"):
        return False
    blocks = rec.get("block_count") or 0
    return rec["n_tensors"] >= max(int(blocks), 1)


def pick_mmproj(projs: list[Candidate], mode: str) -> list[Candidate]:
    """mmproj をどれだけ付けるか。``auto`` は**1本だけ（最小）**.

    無いとマルチモーダルが使えないので既定で付ける。ただし F16 版と Q8 版が
    両方置いてあるリポジトリがあり、全部落とすと数 GB の無駄になる。
    """
    if mode == "none" or not projs:
        return []
    if mode == "all":
        return list(projs)
    return [projs[0]]


def pick_extras(extras: list[Candidate], mode: str) -> list[Candidate]:
    """サブディレクトリのファイルをどれだけ付けるか。既定は**付けない**.

    ``MTP/`` や ``imatrix/`` に入っているものは**本体ではない**ので、載るか
    どうかの判定には出さない（代表に選ぶと KV の単価が 17倍ずれる）。
    ただし「要らない」わけではない:

      * ``MTP/mtp-*.gguf`` は投機デコードで本体と**組にして**使うもの。
        本体だけ落としても使えないし、あとから気づいて取りに戻ることになる
      * ``imatrix/`` は自分で量子化し直すときに要る

    既定を ``none`` にしてあるのは、**何に使うファイルかはこちらには
    分からない**から。要ると分かっている人が ``--extras mtp`` と言う。
    """
    if mode == "none" or not extras:
        return []
    if mode == "all":
        return list(extras)
    return [c for c in extras if MTP_FILE_RE.search(c.files[0])]


# --------------------------------------------------------------------------
# HTTP（Hugging Face の公開 API と Range 取得）
# --------------------------------------------------------------------------

def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def hf_token() -> str | None:
    """トークンを環境変数か ``huggingface-cli login`` の保存先から取る."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    home = os.environ.get("HF_HOME")
    root = Path(home) if home else Path.home() / ".cache" / "huggingface"
    token_file = root / "token"
    try:
        return token_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    head = {"User-Agent": "gguf-fit"}
    token = hf_token()
    if token:
        head["Authorization"] = f"Bearer {token}"
    head.update(headers or {})
    return urllib.request.Request(url, headers=head)


def api_url(repo: str, revision: str) -> str:
    quoted = urllib.parse.quote(revision, safe="")
    return f"{endpoint()}/api/models/{repo}/revision/{quoted}?blobs=true"


def file_url(repo: str, revision: str, filename: str) -> str:
    quoted = urllib.parse.quote(revision, safe="")
    return f"{endpoint()}/{repo}/resolve/{quoted}/{urllib.parse.quote(filename)}"


def repo_info(repo: str, revision: str, timeout: float = 30.0) -> dict:
    """リポジトリのファイル一覧を取る。**数 KB で済む**."""
    with urllib.request.urlopen(_request(api_url(repo, revision)),
                                timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_range(url: str, start: int, end: int, timeout: float = 60.0) -> bytes:
    """``[start, end]`` バイトだけ取る（両端を含む。HTTP の流儀）."""
    req = _request(url, {"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status == 200 and start > 0:
            # Range を無視して全部返してくるサーバがある。**気づかずに
            # 21 GB 受け取らない**ために、ここで止める
            raise OSError("server ignored the Range request")
        return resp.read()


def fetch_header(url: str, first: int = HEADER_FIRST_BYTES,
                 limit: int = HEADER_MAX_BYTES,
                 reader=read_range) -> tuple[Header, int]:
    """先頭だけ取ってヘッダを読む。足りなければ**足りない分だけ**追加で取る.

    戻り値は ``(ヘッダ, 実際に転送したバイト数)``。転送量を返すのは、
    「判定にいくら使ったか」を出力に書けるようにするため。
    """
    buf = b""
    want = first
    while True:
        chunk = reader(url, len(buf), want - 1)
        if not chunk:
            raise OSError(f"no data from {url}")
        buf += chunk
        try:
            return parse_header(buf, _WANT_META), len(buf)
        except TruncatedGGUF as exc:
            want = max(want * 2, exc.need)
            if want > limit:
                raise OSError(
                    f"GGUF header did not end within {limit} bytes") from exc


def is_metadata_only_shard(header: Header) -> bool:
    """テンソルを1本も持たないシャードか.

    実物: unsloth/Qwen3.8-Flash-Next-GGUF の ``-00001-of-00003.gguf`` は
    10.9 MB あるがテンソルは**0本**で、中身はアーキテクチャのメタデータと
    tokenizer だけ。重みは 2本目以降に入っている。
    """
    return not header.tensors


def merge_shard_headers(headers: list[Header]) -> Header:
    """分割 GGUF の各シャードのヘッダを**1本ぶんに見えるよう**束ねる.

    **純粋関数。**ネットワークには触らない。

    分割 GGUF は「メタデータを持つシャード」と「テンソルを持つシャード」が
    別々になっていることがある (実物: ``-00001-of-00003`` はメタデータ 11キー
    でテンソル0本、``-00002`` は 595本でメタデータ0キー、``-00003`` は 629本)。
    片方だけ見ても本体の姿にならない:

      * テンソルが無いと **KV層が数えられない**。``block_count`` で代用しては
        いけない (この実物は 48層中 KV を持つのは 12層だけ。代用すると
        KV/token が**4倍**に出る — README のバグ #1 そのもの)
      * ``n_params`` はシャードごとの合計なので、**足さないと bpw が出ない**

    そこでテンソルは連結、``n_params`` は合算、メタデータは**持っている
    シャードのものを採る**（先に出てきたものを優先し、後のもので埋める）。
    ``version`` と ``header_bytes`` は先頭シャードのものを残す。
    """
    if not headers:
        raise ValueError("no headers to merge")
    if len(headers) == 1:
        return headers[0]
    metadata: dict[str, Any] = {}
    for header in headers:
        for key, value in header.metadata.items():
            metadata.setdefault(key, value)
    return headers[0]._replace(
        metadata=metadata,
        tensors=[t for header in headers for t in header.tensors],
        n_params=sum(header.n_params for header in headers),
    )


# --------------------------------------------------------------------------
# ヘッダ -> gguf-probe と同じ形のレコード
# --------------------------------------------------------------------------

def record_from_header(header: Header, filename: str, size_bytes: int,
                       url: str = "") -> dict:
    """``gguf-probe`` の出力と**同じ形**の dict にする.

    同じ形にしておけば ``gguf-plan`` の関数（``max_ctx`` など）がそのまま使え、
    見積りの式が2か所に分かれない。数字が食い違う事故はたいていこれで起きる。
    """
    meta = header.metadata
    rec: dict[str, Any] = {
        "file": filename.rsplit("/", 1)[-1],
        "path": url,
        "size_gb": round(size_bytes / 1e9, 2),
        "architecture": meta.get("general.architecture"),
        "file_type": meta.get("general.file_type"),
        "n_tensors": len(header.tensors),
    }
    rec["is_language_model"] = rec["architecture"] not in ("clip", "mmproj", None)
    rec["n_params"] = header.n_params
    for suffix, key in _META_KEYS:
        for name, value in meta.items():
            if name.endswith(suffix):
                rec[key] = value
                break
    rec.update(summarize_tensors(header.tensors, rec))
    template = meta.get("tokenizer.chat_template")
    if isinstance(template, str):
        rec["chat_template_len"] = len(template)
        rec["chat_template_has_think"] = "think" in template.lower()
    return rec


def rec_for_size(rec: dict, cand: Candidate) -> dict:
    """代表1本から読んだレコードを、別の量子化に**サイズだけ差し替えて**使う.

    KV/token も native ctx も MTP の有無も、量子化を変えても動かない
    （テンソルの型が変わるだけで、名前と層構造は同じ）。動くのはサイズだけ。
    """
    clone = dict(rec)
    clone["file"] = cand.files[0].rsplit("/", 1)[-1]
    clone["size_gb"] = round(cand.size_bytes / 1e9, 2)
    # n_params は残す。**量子化で変わらない**ので、これがあれば各行の bpw が出る
    # 量子化の配分は**その1本のもの**なので、持ち越さない。
    # 持ち越すと Q8_0 の行に Q4_K の内訳が出る
    for key in ("quant_mix", "weight_mix", "dominant_weight_type",
                "notable_tensors", "mixed_roles", "mtp_only_roles"):
        clone.pop(key, None)
    return clone


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------

def bits_per_weight(size_bytes: int, n_params: int) -> float | None:
    """**実測の** bpw。ファイルサイズ ÷ パラメータ数。名前ではなく実体.

    このリポジトリの出発点は「ファイル名はビット幅について嘘をつく」だった。
    ``UD-Q6_K_XL`` は名前に 6 と入っているが実測 **7.41 bpw**、``UD-Q8_K_XL``
    は **9.21** で ``Q8_0`` (8.51) より重い。名前で並べると隣に見えるものが、
    実体では 1.5 段違う。

    検算になるのが ``BF16``。定義上ちょうど 16.00 になるはずで、実際に
    Qwen3.8-27B で 16.00、Ornith-1.5-35B で 16.01 が出た。ここがずれていたら
    パラメータ数の数え方を間違えている。
    """
    if not n_params:
        return None
    return size_bytes * 8 / n_params


#: 量子化として成立しうる bpw の範囲。実在する一番軽い量子化でも
#: IQ1_S が 1.81 bpw、一番重い F32 で 32.0。**桁で外れているものは重みではない**。
#: 実物で踏んだ: unsloth/Qwen3.8-27B-GGUF のルートに置いてある
#: ``imatrix_unsloth.gguf`` (13 MiB) は 27B のパラメータ数に対して 0.004 bpw。
#: サブディレクトリではなくルートにあるので、置き場所では弾けない。
PLAUSIBLE_BPW = (0.5, 33.0)


def implausible_as_model(size_bytes: int, n_params: int) -> bool:
    """このサイズで「その parameter 数のモデル」はありえないか.

    ``imatrix`` や語彙だけのファイルを量子化と並べないための線。
    **名前ではなく数字で弾く。**パラメータ数が読めていなければ判断しない
    （分からないものを「違う」と言わない）。
    """
    bpw = bits_per_weight(size_bytes, n_params)
    if bpw is None:
        return False
    return not PLAUSIBLE_BPW[0] <= bpw <= PLAUSIBLE_BPW[1]


class Verdict(NamedTuple):
    """1つの候補に対する判定."""

    cand: Candidate
    rec: dict | None
    kv_mode: str | None
    ctx: int
    ctx_f16: int
    ctx_q8: int
    fits: bool
    reason: str  # "ctx" / "size" / "unknown"

    @property
    def bpw(self) -> float | None:
        if not self.rec:
            return None
        return bits_per_weight(self.cand.size_bytes, self.rec.get("n_params") or 0)


def evaluate(cand: Candidate, rec: dict | None, vram: float | None, overhead: float,
             kv_mode: str, min_ctx: int,
             calibrated: KvRates | None = None) -> Verdict:
    """この量子化は載るか。載るなら KV の型と ctx はいくつか.

    ``rec`` が ``None``（``--probe none``）のときは**ファイルサイズだけ**で
    判断する。KV のぶんが読めないので、そうと分かる ``reason="size"`` を返す。
    予算そのものが分からないとき（GPU を検出できず ``--vram`` も無いとき）は
    ``reason="unknown"``。**判定できないことを「入らない」と書かない。**
    """
    if vram is None:
        return Verdict(cand, rec, None, 0, 0, 0, False, "unknown")
    if rec is None or not rec.get("kv_cache"):
        fits = cand.size_gib + overhead <= vram
        return Verdict(cand, rec, None, 0, 0, 0, fits, "size")

    # **サイズは候補のものを使う。**レコードは代表1本から複製してくるので、
    # 差し替え忘れると「Q4 のサイズで Q8 を判定する」が静かに起きる。
    # 判定しているのは cand なのだから、ここで必ず一致させる
    rec = {**rec, "size_gb": round(cand.size_bytes / 1e9, 2)}
    f16 = max_ctx(rec, vram, "f16", overhead, calibrated)
    q8 = max_ctx(rec, vram, "q8_0", overhead, calibrated)
    modes = ("f16", "q8_0") if kv_mode == "auto" else (kv_mode,)
    per_mode = {"f16": f16, "q8_0": q8}
    for mode in modes:
        if per_mode[mode] >= min_ctx:
            return Verdict(cand, rec, mode, per_mode[mode], f16, q8, True, "ctx")
    best = max(modes, key=lambda m: per_mode[m])
    return Verdict(cand, rec, best, per_mode[best], f16, q8, False, "ctx")


def choose(verdicts: list[Verdict], top: int, spread: bool = False,
           min_bpw: float | None = None) -> list[Verdict]:
    """落とすものを選ぶ。**載るものの中で大きいほうから**.

    大きい = ビット数が多い = 品質が高い、というのがここでの並べ方。
    ぴったり1本に絞らないのは、境界付近は実測しないと分からないため
    （このツールが出すのは見積りで、測定ではない）。

    ``spread`` を立てると、上から N 本ではなく **bpw の幅を取って** N 本選ぶ。
    実物で必要になった: unsloth/Qwen3.8-27B の ``--top 3`` は 26.1 / 27.0 /
    29.3 GiB を返す。8.2 / 8.5 / 9.2 bpw で、**3本とも同じビット帯**。
    83 GiB 落として比較できるのは 1段ぶんしかない。
    """
    fitting = [v for v in verdicts if v.fits]
    if min_bpw is not None:
        fitting = [v for v in fitting if (v.bpw or 0) >= min_bpw]
    fitting.sort(key=lambda v: v.cand.size_bytes, reverse=True)
    if not spread or top >= len(fitting):
        return fitting[:top]
    return _spread(fitting, top)


def _spread(fitting: list[Verdict], top: int) -> list[Verdict]:
    """``fitting`` (大きい順) から bpw を散らして ``top`` 本選ぶ.

    一番上は必ず入れる（予算内で最良のものは知りたい）。残りは bpw の
    範囲を等分した目標値にいちばん近いものを、重複しないように取る。
    bpw が読めていないときはサイズで代用する（順序は同じ）。
    """
    def axis(v: Verdict) -> float:
        return v.bpw if v.bpw is not None else v.cand.size_bytes

    high, low = axis(fitting[0]), axis(fitting[-1])
    if high == low:
        return fitting[:top]
    chosen = [0]
    for step in range(1, top):
        target = high - (high - low) * step / (top - 1)
        best = min((i for i in range(len(fitting)) if i not in chosen),
                   key=lambda i: abs(axis(fitting[i]) - target))
        chosen.append(best)
    return [fitting[i] for i in sorted(chosen)]


def filter_candidates(cands: list[Candidate], only: list[str] | None,
                      exclude: list[str] | None) -> list[Candidate]:
    """``--only`` / ``--exclude`` で候補を絞る。**名前の一致であって品質ではない**.

    ``--only 'UD-Q*_K_*'`` のようなグロブを、ラベルとファイル名の両方に当てる。
    大文字小文字は無視する。

    このツールは「名前はビット幅について嘘をつく」と言っている側なので、
    ここで絞ったものを「良いもの」と呼ばない。**手で候補を減らすための道具**
    であって、品質の判断ではない。中身で絞りたいなら ``--min-bpw`` を使う。
    """
    def hit(cand: Candidate, patterns: list[str]) -> bool:
        names = [cand.label.lower(), *(f.rsplit("/", 1)[-1].lower()
                                       for f in cand.files)]
        return any(fnmatch.fnmatch(n, p.lower())
                   for p in patterns for n in names)

    out = list(cands)
    if only:
        out = [c for c in out if hit(c, only)]
    if exclude:
        out = [c for c in out if not hit(c, exclude)]
    return out


def match_pick(cands: list[Candidate], pick: str) -> list[Candidate]:
    """``--pick`` の突き合わせ。ラベル一致を優先し、駄目ならファイル名の部分一致."""
    needle = pick.lower()
    exact = [c for c in cands if c.label.lower() == needle]
    if exact:
        return exact
    return [c for c in cands
            if needle in c.label.lower() or any(needle in f.lower() for f in c.files)]


# --------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------

def render_table(verdicts: list[Verdict], lang: str, chosen: set[str]) -> str:
    rows = []
    for v in sorted(verdicts, key=lambda x: x.cand.size_bytes):
        if v.reason == "unknown":
            mark = t("fetch_unknown", lang)
            ctx_f16 = ctx_q8 = t("fetch_unknown", lang)
        elif v.reason == "size":
            mark = t("fetch_mark_maybe", lang) if v.fits else t("fetch_mark_no", lang)
            ctx_f16 = ctx_q8 = t("fetch_unknown", lang)
        else:
            mark = t("fetch_mark_yes", lang) if v.fits else t("fetch_mark_no", lang)
            ctx_f16 = f"{v.ctx_f16:,}" if v.ctx_f16 else t("fetch_mark_no", lang)
            ctx_q8 = f"{v.ctx_q8:,}" if v.ctx_q8 else t("fetch_mark_no", lang)
        if v.cand.label in chosen:
            mark = t("fetch_mark_take", lang)
        bpw = f"{v.bpw:.2f}" if v.bpw else t("fetch_unknown", lang)
        rows.append((v.cand.label, v.cand.size_gib, bpw, ctx_f16, ctx_q8, mark))

    # 日本語の見出しと「入らない」が混ざるので、文字数ではなく**表示幅**で詰める。
    # 見出し自身も幅に数える (en の "quantization" は中身より長い)
    first = max([*(width(r[0]) for r in rows),
                 width(t("col_quant", lang))]) + 2
    head = (pad(t("col_quant", lang), first) + pad(t("col_filesize", lang), 9, True)
            + pad(t("col_bpw", lang), 7, True)
            + pad(t("col_maxctx_f16", lang), 16, True)
            + pad(t("col_maxctx_q8", lang), 17, True)
            + "   " + t("fetch_col_verdict", lang))
    lines = [head, "-" * width(head)]
    for label, gib, bpw, f16, q8, mark in rows:
        lines.append(pad(label, first) + f"{gib:>8.2f}G" + pad(bpw, 7, True)
                     + pad(f16, 16, True) + pad(q8, 17, True) + "   " + mark)
    return "\n".join(lines)


def hf_binary(explicit: str | None = None) -> str | None:
    """``hf`` を探す。古い環境では ``huggingface-cli`` の名前で入っている."""
    for name in ([explicit] if explicit else ["hf", "huggingface-cli"]):
        found = shutil.which(name)
        if found:
            return found
    return None


def download_command(binary: str, repo: str, files: list[str], dest: Path,
                     revision: str) -> list[str]:
    """実行する ``hf download`` を組み立てる.

    ファイル名を位置引数で渡す。``--include`` のパターンにしないのは、
    ワイルドカードの解釈が増えると「何が落ちてくるか」が読めなくなるため。
    ここで渡した名前のものだけが落ちてくる。
    """
    cmd = [binary, "download", repo, *files, "--local-dir", str(dest)]
    if revision != "main":
        cmd += ["--revision", revision]
    return cmd


def enclosing_git_repo(path: Path) -> Path | None:
    """``path`` が git の作業ツリーの中なら、そのルートを返す.

    モデルをソースのチェックアウトに落とすのは、たいてい ``--dir`` を
    付け忘れただけ（``models_dir`` の既定はカレントなので、リポジトリの中で
    実行するとそうなる）。しかも **``.gitignore`` に ``*.gguf`` が入っていると
    ``git status`` は綺麗なまま**なので、置いたことに気づく機会が無い。
    数十GB が黙って居座る。

    ``git`` は呼ばない。``.git`` を上に辿るだけ（ファイルのこともある —
    worktree や submodule ではファイル）。
    """
    try:
        current = path.expanduser().resolve()
    except OSError:  # pragma: no cover - 解決できないパスは黙って諦める
        return None
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def disk_free_gib(path: Path) -> float | None:
    """書き先の空き容量。取れなければ ``None``（**0 ではない**）."""
    probe_path = path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        return shutil.disk_usage(probe_path).free / GIB
    except OSError:
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=t("desc_fetch", DEFAULT_LANG))
    ap.add_argument("repo", nargs="?", help=t("help_repo", DEFAULT_LANG))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fit", action="store_true", help=t("help_fit", DEFAULT_LANG))
    mode.add_argument("--pick", default=None, help=t("help_fetch_pick", DEFAULT_LANG))
    mode.add_argument("--all", action="store_true", dest="all_",
                      help=t("help_all", DEFAULT_LANG))
    ap.add_argument("--top", type=int, default=DEFAULT_TOP,
                    help=t("help_top", DEFAULT_LANG) + f" (default {DEFAULT_TOP})")
    ap.add_argument("--min-ctx", type=int, default=DEFAULT_MIN_CTX, dest="min_ctx",
                    help=t("help_min_ctx", DEFAULT_LANG) + f" (default {DEFAULT_MIN_CTX})")
    ap.add_argument("--min-bpw", type=float, default=None, dest="min_bpw",
                    help=t("help_min_bpw", DEFAULT_LANG))
    ap.add_argument("--spread", action="store_true",
                    help=t("help_spread", DEFAULT_LANG))
    ap.add_argument("--only", action="append", default=None,
                    help=t("help_only", DEFAULT_LANG))
    ap.add_argument("--exclude", action="append", default=None,
                    help=t("help_exclude", DEFAULT_LANG))
    ap.add_argument("--dir", default=None, dest="models_dir",
                    help=t("help_dir", DEFAULT_LANG))
    ap.add_argument("--revision", default="main", help=t("help_revision", DEFAULT_LANG))
    ap.add_argument("--mmproj", choices=["auto", "all", "none"], default="auto",
                    help=t("help_mmproj", DEFAULT_LANG))
    ap.add_argument("--extras", choices=["none", "mtp", "all"], default="none",
                    help=t("help_extras", DEFAULT_LANG))
    ap.add_argument("--probe", choices=["one", "all", "none"], default="one",
                    help=t("help_probe_mode", DEFAULT_LANG))
    ap.add_argument("--kv", choices=["f16", "q8_0", "auto"], default="auto",
                    help=t("help_kv", DEFAULT_LANG))
    ap.add_argument("--vram", type=float, default=None, help=t("help_vram", DEFAULT_LANG))
    ap.add_argument("--overhead", type=float, default=None,
                    help=t("help_overhead", DEFAULT_LANG))
    ap.add_argument("--yes", "-y", action="store_true",
                    help=t("help_yes", DEFAULT_LANG))
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help=t("help_dry_run", DEFAULT_LANG))
    ap.add_argument("--json", action="store_true", help=t("help_fetch_json", DEFAULT_LANG))
    ap.add_argument("--hf-bin", default=None, dest="hf_bin",
                    help=t("help_hf_bin", DEFAULT_LANG))
    ap.add_argument("--llama-server", action="append", default=None,
                    dest="llama_server", help=t("help_llama_server", DEFAULT_LANG))
    ap.add_argument("--lang", default=None, choices=["en", "ja"],
                    help=t("help_lang", DEFAULT_LANG))
    ap.add_argument("--config", default=None,
                    help="path to a gguf-fit.toml (overrides the search)")
    ap.add_argument("--show-config", action="store_true", dest="show_config",
                    help="print the settings in effect and where each came from")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore vram/threads/device in the config file and take "
                         "them from the hardware again")
    return ap


def _probe_targets(body: list[Candidate], mode: str,
                   picked: list[Candidate]) -> list[Candidate]:
    """ヘッダを読む順番を決める。既定は**代表1本**だが、候補は順に並べる.

    代表は**大きいほうから**試す。中途半端なファイル (draft、壊れかけ、
    置き忘れ) はたいてい小さいので、小さいほうから選ぶと真っ先にそれを掴む。
    実際に掴んだ (MTP の 1.28 GiB を 27B 本体の代表にしていた)。
    """
    if mode == "none" or not body:
        return []
    if mode == "all":
        return body
    if picked:
        # --pick が付いているならその1本。判定したいのはそれなので
        return [picked[0], *sorted(body, key=lambda c: c.size_bytes, reverse=True)]
    return sorted(body, key=lambda c: c.size_bytes, reverse=True)


def _load_records(repo: str, revision: str, targets: list[Candidate],
                  lang: str, first_valid_only: bool = False,
                  ) -> tuple[dict[str, dict], int]:
    """対象のヘッダを読む。戻り値は ``({ラベル: rec}, 転送バイト数)``.

    ``first_valid_only`` は「代表を1本決める」モード。**本体に見えないものは
    採らずに次を試す。**ここで検証しないと、リポジトリに紛れている別物の
    KV/token が全行に当たる。

    先頭シャードが**メタデータだけでテンソルを持っていない**ときは、残りの
    シャードのヘッダも読んで束ねる (``merge_shard_headers``)。読まないと
    KV層が数えられず、bpw も出ない。読む量は増えるので、**増えたことは
    出力に書く** (転送量は戻り値に乗る)。
    """
    recs: dict[str, dict] = {}
    transferred = 0
    for cand in targets:
        url = file_url(repo, revision, cand.files[0])
        try:
            header, used = fetch_header(url)
        except (OSError, ValueError) as exc:
            print("!! " + t("fetch_header_failed", lang,
                            file=cand.files[0], err=exc), file=sys.stderr)
            continue
        transferred += used
        if is_metadata_only_shard(header) and len(cand.files) > 1:
            rest = cand.files[1:]
            print(t("fetch_metadata_only_shard", lang, file=cand.files[0],
                    n=len(rest)), file=sys.stderr)
            headers = [header]
            failed = False
            for name in rest:
                try:
                    more, more_used = fetch_header(file_url(repo, revision, name))
                except (OSError, ValueError) as exc:
                    print("!! " + t("fetch_header_failed", lang,
                                    file=name, err=exc), file=sys.stderr)
                    failed = True
                    break
                transferred += more_used
                headers.append(more)
            if failed:
                # 1本でも欠けると KV層を数え落とす。**中途半端な数字は出さない**
                continue
            header = merge_shard_headers(headers)
        rec = record_from_header(header, cand.files[0], cand.size_bytes, url)
        if first_valid_only and not looks_like_the_main_model(rec):
            print("!! " + t("fetch_not_the_model", lang, file=cand.files[0],
                            n=rec["n_tensors"], blocks=rec.get("block_count") or 0),
                  file=sys.stderr)
            continue
        recs[cand.label] = rec
        if first_valid_only:
            break
    return recs, transferred


def _selection(args, body: list[Candidate], verdicts: list[Verdict],
               lang: str, extras: list[Candidate] | None = None,
               ) -> list[Candidate] | None:
    """3つのモードから、落とす候補を決める。``None`` は「まだ決めない」."""
    if args.all_:
        return list(body)
    if args.pick:
        # **--pick は extras も探す。**候補表に出ていないものを名指しで取りたい
        # ことがある (MTP の draft など)。ここを body だけにしていたので、
        # 「要るなら --pick で」という案内が嘘になっていた
        hits = match_pick(body + list(extras or []), args.pick)
        if not hits:
            sys.exit(t("fetch_pick_none", lang, pick=args.pick,
                       names=", ".join(c.label for c in body)))
        return hits
    if args.fit:
        chosen = choose(verdicts, max(1, args.top), args.spread, args.min_bpw)
        if not chosen:
            return []
        return [v.cand for v in chosen]
    return None


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    if args.refresh:
        cfg = drop_detectable(cfg)
    # llama-server の探し方は gguf-plan と**同じ関数**を通す。ここが食い違うと、
    # 同じマシンで gguf-fetch と gguf-plan が違う GPU を前提に予算を出す
    r_llama = resolve_llama_servers(split_repeated(args.llama_server), cfg)
    hw = _hardware.detect(r_llama.value)
    r_lang = resolve("lang", args.lang, cfg, DEFAULT_LANG)
    r_device = resolve("device", None, cfg, detected=hw.suggested_device())
    device = str(r_device.value) if r_device.value else None
    r_vram = resolve("vram", args.vram, cfg, detected=hw.suggested_vram_gib(device))
    r_overhead = resolve("overhead", args.overhead, cfg, DEFAULT_OVERHEAD_GIB)
    r_dir = resolve("models_dir", args.models_dir, cfg, ".")
    r_hf_bin = resolve("hf_bin", args.hf_bin, cfg)
    lang = r_lang.value

    if args.show_config:
        # **device と検出結果も出す。**予算はデバイスの容量から取っているので、
        # そこを伏せると「48.0 GiB がどこから来たのか」に答えられていない
        print(render_show_config(
            {"lang": r_lang, "vram": r_vram, "overhead": r_overhead,
             "device": r_device, "llama_servers": r_llama,
             "models_dir": r_dir, "hf_bin": r_hf_bin}, cfg_path))
        print()
        print(_hardware.render(hw))
        if not hw.gpus:
            print()
            print(t("hint_no_devices", lang))
        return 0
    if not args.repo:
        ap.error("repo is required (for example: ornith-ai/Ornith-1.5-35B-A3B-GGUF)")

    overhead = float(r_overhead.value)
    vram = float(r_vram.value) if r_vram.value is not None else None
    if vram is None and (args.fit or not (args.pick or args.all_)):
        print("# " + t("hint_no_devices", lang), file=sys.stderr)
        ap.error("could not detect any GPU, so --vram is required "
                 "(or set vram in the config file)")

    # 予算の取り方が怪しいときは、**転送を始める前に**言う。gguf-plan と同じ
    # 文面を同じ関数から出す (片方だけが黙っていると、同じマシンで違う予算に
    # なった理由が分からない)
    for line in budget_warnings(hw, device, vram, r_vram.source,
                                r_device.source, lang):
        print(line, file=sys.stderr)

    try:
        info = repo_info(args.repo, args.revision)
    except (OSError, ValueError) as exc:
        sys.exit(t("fetch_repo_failed", lang, repo=args.repo, err=exc))

    body, projs, extras = group_files(info.get("siblings") or [])
    if not body:
        sys.exit(t("fetch_no_gguf", lang, repo=args.repo))

    if args.only or args.exclude:
        kept = filter_candidates(body, args.only, args.exclude)
        dropped = len(body) - len(kept)
        if not kept:
            sys.exit(t("fetch_filter_empty", lang,
                       patterns=", ".join(args.only or args.exclude or []),
                       names=", ".join(c.label for c in body)))
        print(t("fetch_filtered", lang, dropped=dropped, kept=len(kept)))
        body = kept

    picked = match_pick(body, args.pick) if args.pick else []
    targets = _probe_targets(body, args.probe, picked)
    recs, transferred = _load_records(args.repo, args.revision, targets, lang,
                                      first_valid_only=args.probe == "one")

    #: ヘッダを1本しか読んでいないときは、その1本を全部に当てる。
    #: **どの1本から来た数字かは必ず出力に書く**
    shared = next(iter(recs.values())) if recs else None

    # **ルートに置いてある非モデルを候補から外す。**置き場所では弾けないので
    # bpw で見る。実物: unsloth の imatrix_unsloth.gguf (13 MiB) が
    # --spread の下端として選ばれていた
    n_params = (shared or {}).get("n_params") or 0
    if n_params:
        not_models = [c for c in body if implausible_as_model(c.size_bytes, n_params)]
        if not_models:
            body = [c for c in body if c not in not_models]
            extras = extras + not_models
            print(t("fetch_not_weights", lang, n=len(not_models),
                    names=", ".join(c.files[0] for c in not_models)),
                  file=sys.stderr)
        if not body:
            sys.exit(t("fetch_no_gguf", lang, repo=args.repo))
    # 較正値が別のモデルで測ったものなら、**表を出す前に**言う。
    # 数字はそれらしく出てしまうので、出てから気づく手立てが無い
    if shared is not None:
        mismatch = calibration_mismatch(shared, cfg, lang)
        if mismatch:
            print(mismatch, file=sys.stderr)
    verdicts: list[Verdict] = []
    for cand in body:
        rec = recs.get(cand.label)
        if rec is None and shared is not None and args.probe == "one":
            rec = rec_for_size(shared, cand)
        verdicts.append(evaluate(cand, rec, vram, overhead, args.kv,
                                 args.min_ctx, KvRates.from_config(cfg)))

    selected = _selection(args, body, verdicts, lang, extras)
    files: list[str] = []
    if selected is not None:
        # --pick で extras を名指ししたぶんを二重に足さない
        wanted_extras = [c for c in pick_extras(extras, args.extras)
                         if c not in selected]
        chosen_all = selected + pick_mmproj(projs, args.mmproj) + wanted_extras
        files = [f for c in chosen_all for f in c.files]
        total = sum(c.size_bytes for c in chosen_all)
    else:
        chosen_all, total = [], 0

    if args.json:
        print(json.dumps({
            "repo": args.repo,
            "revision": args.revision,
            "vram_gib": vram,
            "overhead_gib": overhead,
            "header_bytes_transferred": transferred,
            "candidates": [
                {"label": v.cand.label, "files": list(v.cand.files),
                 "size_bytes": v.cand.size_bytes, "fits": v.fits,
                 "basis": v.reason, "kv": v.kv_mode,
                 "bpw": round(v.bpw, 3) if v.bpw else None,
                 "max_ctx_f16": v.ctx_f16, "max_ctx_q8_0": v.ctx_q8}
                for v in sorted(verdicts, key=lambda x: x.cand.size_bytes)
            ],
            "mmproj": [{"label": c.label, "files": list(c.files),
                        "size_bytes": c.size_bytes} for c in projs],
            "extras": [{"label": c.label, "files": list(c.files),
                        "size_bytes": c.size_bytes,
                        "mtp": bool(MTP_FILE_RE.search(c.files[0]))}
                       for c in extras],
            "selected": files,
            "selected_bytes": total,
        }, ensure_ascii=False, indent=2))
        return 0

    print(t("fetch_header_line", lang, repo=args.repo, rev=args.revision))
    if vram is not None:
        print(t("fetch_budget", lang, vram=vram, overhead=overhead,
                kv=args.kv, min_ctx=args.min_ctx))
    print()
    chosen_labels = {c.label for c in (selected or [])}
    print(render_table(verdicts, lang, chosen_labels))
    print()

    if transferred and recs:
        source = ", ".join(sorted(recs))
        print(t("fetch_kv_source" if args.probe == "one" else "fetch_kv_source_all",
                lang, files=source, mib=transferred / (1024 * 1024)))
    elif transferred:
        # 取りに行ったが1つも使えなかった。**「ヘッダから読みました」とは
        # 書かない。**書いていたので、ファイル名が空欄のまま出ていた
        print(t("fetch_no_usable_header", lang, mib=transferred / (1024 * 1024)))
    else:
        print(t("fetch_size_only", lang))
    if projs:
        print(t("fetch_mmproj_found", lang, n=len(projs),
                gib=projs[0].size_gib, name=projs[0].files[0]))
    if extras:
        # サブディレクトリの GGUF は本体ではない。判定には出さないが、
        # **黙って捨てない**。要ると分かっている人が取れるように言う
        taken = pick_extras(extras, args.extras)
        gib = sum(c.size_bytes for c in taken) / GIB
        key = "fetch_extras_taken" if taken else "fetch_extras_available"
        print(t(key, lang, n=len(extras), taken=len(taken), gib=gib,
                names=", ".join(c.files[0] for c in extras[:3])))

    if selected is None:
        print()
        print(t("fetch_next", lang, repo=args.repo))
        return 0
    if not files:
        print()
        print(t("fetch_nothing_fits", lang, vram=vram or 0.0))
        return 1

    dest = Path(str(r_dir.value)).expanduser() / args.repo.split("/")[-1]
    print()
    print(t("fetch_plan", lang, n=len(files), gib=total / GIB))
    for cand in chosen_all:
        for name in cand.files:
            print(f"  {name}")
    print(f"  -> {dest}")

    repo_root = enclosing_git_repo(dest)
    if repo_root is not None:
        print(t("fetch_inside_repo", lang, dest=dest, root=repo_root),
              file=sys.stderr)

    free = disk_free_gib(dest)
    short_on_disk = False
    if free is not None:
        need = total / GIB
        short_on_disk = free < need
        if short_on_disk:
            print(t("fetch_disk_short", lang, free=free, need=need))
            # **--dry-run では止めない。**「落とさずコマンドだけ見せろ」と
            # 言われているのに、空き容量を理由に何も出さないのでは答えに
            # ならない。空けてから流すのはこちらの都合ではなく利用者の判断
            if not args.dry_run:
                return 1
        else:
            print(t("fetch_disk_ok", lang, free=free))

    binary = hf_binary(str(r_hf_bin.value) if r_hf_bin.value else None)
    cmd = download_command(binary or "hf", args.repo, files, dest, args.revision)
    print()
    print(shlex.join(cmd))

    if args.dry_run:
        # 容量が足りないことは伝わっているので、終了コードには残す
        return 1 if short_on_disk else 0
    if binary is None:
        print()
        print(t("fetch_no_hf", lang), file=sys.stderr)
        return 127
    if not args.yes:
        try:
            answer = input("\n" + t("fetch_confirm", lang)).strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print(t("fetch_cancelled", lang))
            return 0

    dest.mkdir(parents=True, exist_ok=True)
    # 引数はリストで渡している (shell=False)。ファイル名は API が返したものだけ
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return result.returncode

    # 落としたら次は測る番。**ここでの数字はまだ見積り**なので、そう言っておく
    print()
    print(t("fetch_done", lang, dir=dest))
    print(f"  gguf-probe --json --out gguf.json {dest}/*.gguf")
    print(f"  gguf-plan gguf.json --pick {selected[0].label}" if selected
          else "  gguf-plan gguf.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
