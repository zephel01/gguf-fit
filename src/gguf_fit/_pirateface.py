"""Pirate Face (pirateface.co) から GGUF を落とすための部品.

Hugging Face からモデルが消えることがある。Pirate Face は、その GGUF を
BitTorrent の swarm で配っている索引サイトで、各モデルのページに

  * 元の Hugging Face リビジョン (commit SHA)
  * リポジトリ全体の magnet (infohash)
  * ファイルごとの名前・おおよそのサイズ・SHA-256

が載っている。ここはそれを読むだけの純粋寄りの部品で、``fetch.py`` から使う。

**実測でわかっていること** (2026-10 時点) と、そこから決めたこと:

  * 公開されている JSON API は無い。一覧はページ (Next.js の RSC ペイロード) の
    中にしか無い → ``parse_page`` でそこを読む。**レイアウトが変わったら
    読めなくなる**ので、読めなかったときは黙らず ``PageError`` で止める
  * ``ws=`` の web-seed は HF の同一リビジョンへの 302 だった。HF が
    モデルを消せば web-seed も死ぬ → HF 非依存で落とせるのは swarm だけ
  * サイズは「5.6 GB」のように丸められている (0.1 GB 刻み)。適合判定の
    粗い見積りには足りるが、**正確な値ではない**と出力で言う
  * SHA-256 はファイルごとに載っている → 落としたあとの照合に使う

magnet からのファイル選択は、aria2c の ``--select-file`` を使う。
その番号は torrent の中のファイル順なので、まず ``--bt-metadata-only`` で
.torrent だけ取り、ここで読んで番号を決める (``torrent_files``)。
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_ENDPOINT = "https://pirateface.co"

#: ページ1枚で数百 KB。暴走して巨大な応答を抱え込まないための上限
PAGE_MAX_BYTES = 8 * 1024 * 1024

#: ``--bt-metadata-only`` を待つ秒数。peer が居ないときに居座らない
METADATA_TIMEOUT_S = 180

_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


class PageError(Exception):
    """ページを読めない、または読み取れる形をしていない."""


class PageFile(NamedTuple):
    name: str
    size_bytes: int  # 丸められた値。0 は読めなかった
    sha256: str | None


class PageInfo(NamedTuple):
    repo: str
    revision: str
    infohash: str | None
    magnet: str | None
    files: list[PageFile]


# --------------------------------------------------------------------------
# 入力 (URL / repo ID) の解釈
# --------------------------------------------------------------------------

def endpoint() -> str:
    return os.environ.get("PIRATEFACE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def parse_repo_arg(arg: str) -> tuple[str | None, str]:
    """``repo`` 引数を ``(取得元, owner/name)`` にする.

    取得元は URL のホストで決まる。URL でなければ ``None`` (= 呼び出し側が
    決める)。``ValueError`` は「URL だが読めない」。
    """
    if not re.match(r"^https?://", arg, re.IGNORECASE):
        return None, arg
    parts_url = urllib.parse.urlsplit(arg)
    host = (parts_url.hostname or "").lower()
    parts = [urllib.parse.unquote(p) for p in parts_url.path.split("/") if p]
    if host == "pirateface.co" or host.endswith(".pirateface.co"):
        kind = "pirateface"
        # /zh/owner/name のような言語プレフィックス。2文字+3区間以上のときだけ
        if len(parts) >= 3 and re.fullmatch(r"[a-z]{2}", parts[0]):
            parts = parts[1:]
    elif host in ("huggingface.co", "hf.co"):
        kind = "hf"
    else:
        raise ValueError(f"unsupported host: {host or arg}")
    if len(parts) < 2:
        raise ValueError(f"no owner/name in the URL: {arg}")
    return kind, f"{parts[0]}/{parts[1]}"


# --------------------------------------------------------------------------
# ページの取得と読み取り
# --------------------------------------------------------------------------

def page_url(repo: str) -> str:
    return f"{endpoint()}/{urllib.parse.quote(repo, safe='/')}"


def fetch_page(repo: str, timeout: float = 30.0) -> PageInfo:
    req = urllib.request.Request(page_url(repo), headers={"User-Agent": "gguf-fit"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(PAGE_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise PageError(f"HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise PageError(str(exc)) from exc
    if len(body) > PAGE_MAX_BYTES:
        raise PageError("response too large")
    return parse_page(body.decode("utf-8", errors="replace"), repo)


def parse_size(text: str) -> int:
    """``"5.6 GB"`` → バイト数。単位は 10 進 (GB = 10^9)。読めなければ 0.

    10 進と判断した根拠: 9B の Q8_0 (8.5 bpw) は約 9.6e9 バイトで、
    ページは「9.5 GB」と書く。2進 (GiB) なら 8.9 になる。
    """
    m = re.match(r"^\s*([\d.,]+)\s*([KMGT]?B)\s*$", text, re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(float(m.group(1).replace(",", "")) * _UNITS[m.group(2).upper()])
    except ValueError:
        return 0


_ROW_SPLIT = '"className":"mono ft-path"'
_NAME_RE = re.compile(r'^,"title":"([^"]+)"')
_SIZE_RE = re.compile(r'"className":"ft-size","children":"([^"]*)"')
_SHA_RE = re.compile(r'"hash":"([0-9a-fA-F]{64})"')
_MAGNET_RE = re.compile(r'magnet:\?xt=urn:btih:([0-9a-fA-F]{40}|[A-Za-z2-7]{32})[^"\\<>\s]*')
_REV_RE = re.compile(r"Source revision</dt><dd[^>]*>([0-9a-f]{40})<")
_REV_LINK_RE = re.compile(r"/resolve/([0-9a-f]{40})/")


def parse_page(text: str, repo: str) -> PageInfo:
    """ページの HTML から、リビジョン・magnet・ファイル表を取り出す.

    表は HTML ではなく RSC ペイロード (``\\"`` でエスケープされた JSON 風の
    文字列) の中にしか無い。行ごとに区切って、**行の中だけ**で名前・サイズ・
    SHA を探す (SHA の無い行が、次の行の SHA を拾わないように)。
    """
    flat = text.replace('\\"', '"')

    rev = _REV_RE.search(flat) or _REV_LINK_RE.search(flat)
    if not rev:
        raise PageError("no source revision found (page layout changed?)")
    revision = rev.group(1)

    magnet = None
    infohash = None
    m = _MAGNET_RE.search(text)
    if m:
        magnet = html.unescape(m.group(0)).replace("\\u0026", "&")
        infohash = m.group(1).lower() if len(m.group(1)) == 40 else None

    files: list[PageFile] = []
    seen: set[str] = set()
    for seg in flat.split(_ROW_SPLIT)[1:]:
        nm = _NAME_RE.match(seg)
        if not nm:
            continue
        name = nm.group(1)
        if name in seen:
            continue
        seen.add(name)
        # 次の行までに入る範囲だけを見る (split 済みなので seg が1行ぶん)
        sz = _SIZE_RE.search(seg)
        sha = _SHA_RE.search(seg)
        files.append(PageFile(
            name=name,
            size_bytes=parse_size(sz.group(1)) if sz else 0,
            sha256=sha.group(1).lower() if sha else None))
    if not files:
        raise PageError("no file table found (page layout changed?)")
    return PageInfo(repo=repo, revision=revision, infohash=infohash,
                    magnet=magnet, files=files)


def siblings_from_page(info: PageInfo) -> list[dict[str, Any]]:
    """HF の ``siblings`` と同じ形にする (``group_files`` にそのまま渡せる)."""
    return [{"rfilename": f.name, "size": f.size_bytes} for f in info.files]


# --------------------------------------------------------------------------
# .torrent (bencode) の読み取り: ファイル一覧と ``--select-file`` の番号
# --------------------------------------------------------------------------

def bdecode(data: bytes) -> Any:
    """最小の bencode デコーダ。``bytes`` / ``int`` / ``list`` / ``dict[bytes, ...]``."""
    value, pos = _bdec(data, 0)
    if pos != len(data):
        raise ValueError("trailing data after bencode value")
    return value


def _bdec(data: bytes, pos: int) -> tuple[Any, int]:
    if pos >= len(data):
        raise ValueError("unexpected end of data")
    c = data[pos:pos + 1]
    if c == b"i":
        end = data.index(b"e", pos)
        return int(data[pos + 1:end]), end + 1
    if c == b"l":
        pos += 1
        out = []
        while data[pos:pos + 1] != b"e":
            item, pos = _bdec(data, pos)
            out.append(item)
        return out, pos + 1
    if c == b"d":
        pos += 1
        out_d: dict[bytes, Any] = {}
        while data[pos:pos + 1] != b"e":
            key, pos = _bdec(data, pos)
            val, pos = _bdec(data, pos)
            out_d[key] = val
        return out_d, pos + 1
    if c.isdigit():
        colon = data.index(b":", pos)
        n = int(data[pos:colon])
        start = colon + 1
        if start + n > len(data):
            raise ValueError("string runs past the end of data")
        return data[start:start + n], start + n
    raise ValueError(f"bad bencode byte at {pos}")


class TorrentFile(NamedTuple):
    index: int      # aria2 --select-file の番号 (1 始まり)
    path: str       # torrent 内の相対パス (トップのディレクトリ名を除く)
    length: int


def torrent_files(torrent: bytes) -> tuple[str, list[TorrentFile]]:
    """``(トップ名, ファイル一覧)``。v1 (``info.files`` / 単一ファイル) のみ."""
    meta = bdecode(torrent)
    info = meta.get(b"info") if isinstance(meta, dict) else None
    if not isinstance(info, dict):
        raise ValueError("not a torrent: no info dictionary")
    name = info.get(b"name", b"").decode("utf-8", errors="replace")
    out: list[TorrentFile] = []
    if b"files" in info:
        for i, ent in enumerate(info[b"files"], start=1):
            parts = [p.decode("utf-8", errors="replace") for p in ent.get(b"path", [])]
            out.append(TorrentFile(i, "/".join(parts), int(ent.get(b"length", 0))))
    elif b"length" in info:
        out.append(TorrentFile(1, name, int(info[b"length"])))
    else:
        raise ValueError("unsupported torrent (v1 file list not found)")
    return name, out


def select_indices(files: list[TorrentFile], wanted: list[str],
                   ) -> tuple[list[int], list[str]]:
    """欲しいファイル名から番号を引く。``(番号, torrent に無かった名前)``."""
    by_path = {f.path: f for f in files}
    idx: list[int] = []
    missing: list[str] = []
    for name in wanted:
        hit = by_path.get(name)
        if hit is None:
            missing.append(name)
        else:
            idx.append(hit.index)
    return sorted(set(idx)), missing


# --------------------------------------------------------------------------
# aria2c のコマンド
# --------------------------------------------------------------------------

def metadata_command(aria2: str, magnet: str, workdir: Path,
                     timeout_s: int = METADATA_TIMEOUT_S) -> list[str]:
    """magnet から .torrent だけを取る (本体は落とさない)."""
    return [aria2, "--bt-metadata-only=true", "--bt-save-metadata=true",
            f"--bt-stop-timeout={timeout_s}", "--seed-time=0",
            "--dir", str(workdir), magnet]


def download_command(aria2: str, torrent: Path, indices: list[int] | str,
                     dest: Path) -> list[str]:
    """選んだファイルだけ swarm から落とす.

    ``indices`` は番号の列。文字列なら**そのまま**入れる (``--dry-run`` で
    まだ番号が決まっていないときの ``<n,...>`` 表示用)。

    ``--seed-time=0``: 落とし終えたら止まる (常駐しない)。続きから再開できるよう
    ``--continue``、落とした後は piece のハッシュで検証する。
    """
    select = indices if isinstance(indices, str) else ",".join(str(i) for i in indices)
    return [aria2, "--select-file=" + select,
            "--continue=true", "--check-integrity=true", "--seed-time=0",
            "--file-allocation=none", "--dir", str(dest), str(torrent)]


# --------------------------------------------------------------------------
# 落としたあとの整理と照合
# --------------------------------------------------------------------------

def flatten(dest: Path, top: str, names: list[str]) -> list[Path]:
    """``dest/<top>/<name>`` を ``dest/<name>`` へ上げる (``hf download`` と同じ配置).

    既に ``dest/<name>`` がある (前回の続き) なら、そちらを正とする。
    戻り値は ``dest`` 直下に置いたパス。
    """
    out: list[Path] = []
    for name in names:
        src = dest / top / name
        dst = dest / name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # 同名がある。上書きして消すのではなく、新しいほうを残して
                # 古いほうを .old に退避する (消さない)
                dst.replace(dst.with_name(dst.name + ".old"))
            src.replace(dst)
        out.append(dst)
    top_dir = dest / top
    if top and top_dir.is_dir():
        try:
            # 空になったディレクトリだけ片付ける。**中身が残っていれば何も消さない**。
            # 実測: aria2 は選んだファイルと piece を共有する隣のファイルの端も
            # 取るので、選ばなかったファイルの欠片が残ることがある
            for sub in sorted(top_dir.rglob("*"), reverse=True):
                if sub.is_dir():
                    sub.rmdir()
            top_dir.rmdir()
        except OSError:
            pass
    return out


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify(dest: Path, info: PageInfo, names: list[str],
           ) -> tuple[list[str], list[str], list[str]]:
    """ページの SHA-256 と照合する。``(一致, 不一致, 照合できなかった)``."""
    want = {f.name: f.sha256 for f in info.files}
    ok: list[str] = []
    bad: list[str] = []
    skipped: list[str] = []
    for name in names:
        expect = want.get(name)
        path = dest / name
        if not expect or not path.is_file():
            skipped.append(name)
            continue
        (ok if sha256_of(path) == expect else bad).append(name)
    return ok, bad, skipped
