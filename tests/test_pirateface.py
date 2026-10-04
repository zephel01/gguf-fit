"""Pirate Face (swarm) 対応のテスト。**ネットワークには一切出ない**.

Pirate Face のページは localhost の ``http.server`` で再現し、aria2c は
**偽の実行ファイル**で置き換える。偽の aria2c は本物と同じ2段の呼ばれ方
(``--bt-metadata-only`` で .torrent を取る → ``--select-file`` で本体を取る)
に従って、``<dir>/<torrent の名前>/<path>`` にファイルを置く。
つまり「番号の決め方」「配置の整え方」「SHA-256 の照合」「失敗時の切替」を、
本物の swarm 無しで通して確かめる。

固定しているのは主に4つ:

  * **ページの読み取り** — 表は RSC ペイロードの中にしかなく、SHA の無い行が
    次の行の SHA を拾うと別のファイルに別の SHA が付く
  * **--select-file の番号** — 1つずれると別の量子化を落とす
  * **HF が消えたときの挙動** — 判定はサイズだけで、そうと分かる形で出す
  * **照合の失敗は成功にしない** — 壊れたファイルを「完了」と言わない
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from gguf_fit import _hardware, fetch
from gguf_fit import _pirateface as pf

REPO = "owner/Model-GGUF"
REV = "1041d1eabbce89549c1b6e28b52e972093e832a1"
INFOHASH = "4fe4b101834ac0e5a63fbd9ee233a6e191ca6d2e"
TOP = "Model-GGUF"

CONTENT = {
    "M-Q4_K_M.gguf": b"q4" * 100,
    "M-Q8_0.gguf": b"q8" * 200,
    "mmproj-M-F16.gguf": b"pj" * 50,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """実行した場所の設定ファイル (gguf-fit.toml) や言語設定に左右されない.

    リポジトリ直下には利用者の ``gguf-fit.toml`` (``lang = "ja"`` など) があり得て、
    カレントから探されるので、**空のディレクトリに移ってから**走らせる。
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GGUF_FIT_LANG", "en")


# --- 合成ページ -----------------------------------------------------------

def _row(name: str, size: str, digest: str | None) -> str:
    sha_cell = (
        '["$","td",null,{"className":"ft-sha","children":["$","span",null,'
        '{"className":"ft-shawrap","children":[["$","span",null,'
        '{"className":"ft-match","title":"Matches","children":"✓"}],'
        f'["$","$L20",null,{{"hash":"{digest}"}}]]}}]}}]'
    ) if digest else '["$","td",null,{"className":"ft-sha","children":"-"}]'
    return (
        f'["$","tr","{name}",{{"children":['
        f'["$","td",null,{{"className":"mono ft-path","title":"{name}","children":"{name}"}}],'
        f'["$","td",null,{{"className":"ft-size","children":"{size}"}}],'
        f'{sha_cell},'
        f'["$","td",null,{{"className":"ft-dl","children":["$","a",null,'
        f'{{"href":"https://huggingface.co/{REPO}/resolve/{REV}/{name}"}}]}}]]}}]'
    )


def make_page(rows: list[tuple[str, str, str | None]], magnet: bool = True,
              revision: str | None = REV) -> str:
    """実物のページと同じ入れ子の、必要な部分だけ."""
    flat = ",".join(_row(*r) for r in rows)
    rsc = flat.replace('"', '\\"')     # 実物は RSC の文字列の中に入っている
    parts = ["<html><body>"]
    if revision:
        parts.append(f'<dl><div><dt>Source revision</dt><dd class="mono">{revision}'
                     "</dd></div></dl>")
    if magnet:
        parts.append(
            f'<a href="magnet:?xt=urn:btih:{INFOHASH}&amp;tr=udp%3A%2F%2Ftracker.example'
            '%3A6969%2Fannounce&amp;ws=https%3A%2F%2Fpf.example%2Fapi%2Fws%2F">m</a>')
    parts.append(f'<script>self.__next_f.push([1,"{rsc}"])</script></body></html>')
    return "".join(parts)


def default_rows() -> list[tuple[str, str, str | None]]:
    return [(name, f"{len(data)} B", sha(data)) for name, data in CONTENT.items()]


# --- ページの読み取り -----------------------------------------------------

def test_parse_page_reads_revision_magnet_and_rows():
    info = pf.parse_page(make_page(default_rows()), REPO)
    assert info.revision == REV
    assert info.infohash == INFOHASH
    assert info.magnet is not None and info.magnet.startswith(
        f"magnet:?xt=urn:btih:{INFOHASH}&tr=udp%3A")
    assert "&amp;" not in info.magnet
    assert [f.name for f in info.files] == list(CONTENT)
    assert info.files[0] == pf.PageFile("M-Q4_K_M.gguf", 200, sha(CONTENT["M-Q4_K_M.gguf"]))


def test_a_row_without_sha_does_not_borrow_the_next_rows_sha():
    rows = [("README.md", "1 KB", None), ("M-Q4_K_M.gguf", "5.6 GB", "ab" * 32)]
    info = pf.parse_page(make_page(rows), REPO)
    by_name = {f.name: f for f in info.files}
    assert by_name["README.md"].sha256 is None
    assert by_name["M-Q4_K_M.gguf"].sha256 == "ab" * 32
    assert by_name["M-Q4_K_M.gguf"].size_bytes == 5_600_000_000


def test_page_without_a_magnet_is_still_usable():
    info = pf.parse_page(make_page(default_rows(), magnet=False), REPO)
    assert info.magnet is None and info.infohash is None
    assert len(info.files) == 3


def test_page_with_a_changed_layout_stops_instead_of_guessing():
    with pytest.raises(pf.PageError, match="no source revision"):
        pf.parse_page("<html>nothing here</html>", REPO)
    with pytest.raises(pf.PageError, match="no file table"):
        pf.parse_page(make_page([]), REPO)


def test_revision_falls_back_to_the_resolve_links():
    info = pf.parse_page(make_page(default_rows(), revision=None), REPO)
    assert info.revision == REV


@pytest.mark.parametrize("text, expected", [
    ("5.6 GB", 5_600_000_000), ("18 GB", 18_000_000_000), ("624 MB", 624_000_000),
    ("1,000 KB", 1_000_000), ("12 B", 12), ("1.5 TB", 1_500_000_000_000),
    ("", 0), ("big", 0), ("5.6 GiB", 0),
])
def test_parse_size_is_decimal(text, expected):
    assert pf.parse_size(text) == expected


@pytest.mark.parametrize("arg, expected", [
    ("owner/name", (None, "owner/name")),
    ("https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF",
     ("pirateface", "mradermacher/Ornith-1.5-9B-uncensored-GGUF")),
    ("https://pirateface.co/zh/a/b", ("pirateface", "a/b")),
    ("https://pirateface.co/ab/cd", ("pirateface", "ab/cd")),   # 2文字の owner を削らない
    ("https://huggingface.co/org/repo/tree/main", ("hf", "org/repo")),
    ("https://hf.co/org/repo", ("hf", "org/repo")),
])
def test_parse_repo_arg(arg, expected):
    assert pf.parse_repo_arg(arg) == expected


@pytest.mark.parametrize("arg", ["https://example.com/a/b", "https://pirateface.co/onlyone"])
def test_parse_repo_arg_rejects_what_it_cannot_read(arg):
    with pytest.raises(ValueError):
        pf.parse_repo_arg(arg)


# --- .torrent -------------------------------------------------------------

def bencode(value) -> bytes:
    if isinstance(value, int):
        return b"i%de" % value
    if isinstance(value, bytes):
        return b"%d:%s" % (len(value), value)
    if isinstance(value, str):
        return bencode(value.encode())
    if isinstance(value, list):
        return b"l" + b"".join(bencode(v) for v in value) + b"e"
    if isinstance(value, dict):
        return b"d" + b"".join(bencode(k) + bencode(v)
                               for k, v in sorted(value.items())) + b"e"
    raise TypeError(value)


def make_torrent(files: dict[str, bytes], top: str = TOP) -> bytes:
    return bencode({"announce": "http://t.example/announce", "info": {
        "name": top, "piece length": 16384, "pieces": b"\0" * 20,
        "files": [{"length": len(d), "path": n.split("/")} for n, d in files.items()],
    }})


def test_bdecode_round_trip():
    value = {b"a": [1, b"xy", {b"k": -3}], b"b": b""}
    assert pf.bdecode(bencode({"a": [1, "xy", {"k": -3}], "b": ""})) == value
    with pytest.raises(ValueError):
        pf.bdecode(b"i1e-junk")
    with pytest.raises(ValueError):
        pf.bdecode(b"5:ab")


def test_torrent_files_and_select_indices_are_one_based_in_file_order():
    top, files = pf.torrent_files(make_torrent(CONTENT))
    assert top == TOP
    assert [(f.index, f.path) for f in files] == [
        (1, "M-Q4_K_M.gguf"), (2, "M-Q8_0.gguf"), (3, "mmproj-M-F16.gguf")]
    # 順序は指定順ではなく番号順に正規化する
    assert pf.select_indices(files, ["mmproj-M-F16.gguf", "M-Q4_K_M.gguf"]) == (
        [1, 3], [])
    assert pf.select_indices(files, ["M-Q4_K_M.gguf", "nope.gguf"]) == ([1], ["nope.gguf"])


def test_single_file_torrent():
    data = bencode({"info": {"name": "one.gguf", "length": 7, "piece length": 1,
                             "pieces": b""}})
    assert pf.torrent_files(data) == ("one.gguf", [pf.TorrentFile(1, "one.gguf", 7)])


def test_not_a_torrent():
    with pytest.raises(ValueError):
        pf.torrent_files(bencode({"nope": 1}))


# --- コマンド -------------------------------------------------------------

def test_aria2_commands_are_lists_not_shell_strings(tmp_path):
    magnet = f"magnet:?xt=urn:btih:{INFOHASH}&tr=udp%3A%2F%2Fx"
    meta = pf.metadata_command("aria2c", magnet, tmp_path)
    assert meta[0] == "aria2c" and meta[-1] == magnet     # & を含んでも1引数のまま
    assert "--bt-metadata-only=true" in meta and "--bt-save-metadata=true" in meta
    assert f"--bt-stop-timeout={pf.METADATA_TIMEOUT_S}" in meta
    dl = pf.download_command("aria2c", tmp_path / "x.torrent", [1, 3], tmp_path / "out")
    assert "--select-file=1,3" in dl
    assert "--seed-time=0" in dl and "--continue=true" in dl
    assert dl[-1] == str(tmp_path / "x.torrent")
    assert "--select-file=<n,...>" in pf.download_command("a", Path("t"), "<n,...>", Path("d"))


# --- 配置と照合 -----------------------------------------------------------

def test_flatten_moves_files_up_and_removes_the_empty_directory(tmp_path):
    (tmp_path / TOP).mkdir()
    (tmp_path / TOP / "a.gguf").write_bytes(b"1")
    out = pf.flatten(tmp_path, TOP, ["a.gguf"])
    assert out == [tmp_path / "a.gguf"]
    assert (tmp_path / "a.gguf").read_bytes() == b"1"
    assert not (tmp_path / TOP).exists()


def test_flatten_never_deletes_the_leftovers_of_neighbouring_pieces(tmp_path):
    """aria2 は piece を共有する隣のファイルの端も取る。それは消さずに残す."""
    (tmp_path / TOP).mkdir()
    (tmp_path / TOP / "a.gguf").write_bytes(b"1")
    (tmp_path / TOP / "neighbour.gguf").write_bytes(b"partial piece")
    pf.flatten(tmp_path, TOP, ["a.gguf"])
    assert (tmp_path / "a.gguf").exists()
    assert (tmp_path / TOP / "neighbour.gguf").read_bytes() == b"partial piece"


def test_flatten_keeps_the_previous_copy_instead_of_deleting_it(tmp_path):
    (tmp_path / TOP).mkdir()
    (tmp_path / TOP / "a.gguf").write_bytes(b"new")
    (tmp_path / "a.gguf").write_bytes(b"old")
    pf.flatten(tmp_path, TOP, ["a.gguf"])
    assert (tmp_path / "a.gguf").read_bytes() == b"new"
    assert (tmp_path / "a.gguf.old").read_bytes() == b"old"


def test_verify_separates_match_mismatch_and_unchecked(tmp_path):
    info = pf.parse_page(make_page([*default_rows(), ("extra.gguf", "1 B", None)]), REPO)
    (tmp_path / "M-Q4_K_M.gguf").write_bytes(CONTENT["M-Q4_K_M.gguf"])
    (tmp_path / "M-Q8_0.gguf").write_bytes(b"corrupt")
    (tmp_path / "extra.gguf").write_bytes(b"x")
    ok, bad, skipped = pf.verify(
        tmp_path, info, ["M-Q4_K_M.gguf", "M-Q8_0.gguf", "extra.gguf", "missing.gguf"])
    assert ok == ["M-Q4_K_M.gguf"]
    assert bad == ["M-Q8_0.gguf"]
    assert skipped == ["extra.gguf", "missing.gguf"]


# --- fetch.main を通す ----------------------------------------------------

class _Site(BaseHTTPRequestHandler):
    """Pirate Face のページと、(任意で) Hugging Face の API."""

    page: ClassVar[str] = ""
    hf_api: ClassVar[dict | None] = None
    hf_status: ClassVar[int] = 404
    seen: ClassVar[list[str]] = []

    def log_message(self, *_args):
        pass

    def _send(self, status: int, body: bytes, ctype: str = "text/html") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.seen.append(self.path)
        if self.path == f"/{REPO}":
            self._send(200, self.page.encode("utf-8"))
        elif self.path.startswith("/api/models/"):
            if self.hf_api is None:
                self._send(self.hf_status, b"{}", "application/json")
            else:
                self._send(200, json.dumps(self.hf_api).encode(), "application/json")
        else:
            self._send(404, b"")


@pytest.fixture
def site(monkeypatch):
    _Site.page = make_page(default_rows())
    _Site.hf_api = None
    _Site.hf_status = 404
    _Site.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("PIRATEFACE_ENDPOINT", url)
    monkeypatch.setenv("HF_ENDPOINT", url)       # 同じサーバ。/api/models/ だけ HF 役
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _hardware.Hardware(
        gpus=[], ram_gib=64.0, physical_cores=8, logical_cores=16, unified_memory=False))
    yield _Site
    server.shutdown()


FAKE_ARIA2 = '''#!{python}
import json, os, sys
sys.path.insert(0, {src!r})
from gguf_fit import _pirateface as pf

args = sys.argv[1:]
log = os.environ["FAKE_LOG"]
with open(log, "a") as fh:
    fh.write(json.dumps(args) + "\\n")
if os.environ.get("FAKE_FAIL") == "1":
    sys.exit(7)
dest = args[args.index("--dir") + 1]
if "--bt-metadata-only=true" in args:
    with open(os.path.join(dest, "x.torrent"), "wb") as fh:
        fh.write(open(os.environ["FAKE_TORRENT"], "rb").read())
    sys.exit(0)
torrent = open(args[-1], "rb").read()
top, files = pf.torrent_files(torrent)
select = [a for a in args if a.startswith("--select-file=")][0].split("=", 1)[1]
payload = json.load(open(os.environ["FAKE_PAYLOAD"]))
if os.environ.get("FAKE_NEIGHBOUR"):
    # 実物の aria2 は piece を共有する隣のファイルの端も取る
    path = os.path.join(dest, top, files[1].path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as out:
        out.write(b"partial piece")
for idx in (int(i) for i in select.split(",")):
    f = files[idx - 1]
    path = os.path.join(dest, top, f.path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as out:
        out.write(bytes.fromhex(payload[f.path]))
'''

FAKE_HF = '''#!{python}
import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
'''


def _script(path: Path, template: str) -> Path:
    path.write_text(template.format(python=sys.executable,
                                    src=str(Path(pf.__file__).parent.parent)))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    torrent = tmp_path / "in.torrent"
    torrent.write_bytes(make_torrent(CONTENT))
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({n: d.hex() for n, d in CONTENT.items()}))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_TORRENT", str(torrent))
    monkeypatch.setenv("FAKE_PAYLOAD", str(payload))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_NEIGHBOUR", raising=False)
    aria2 = _script(tmp_path / "fake-aria2c", FAKE_ARIA2)
    hf = _script(tmp_path / "fake-hf", FAKE_HF)

    def calls() -> list[list[str]]:
        return [json.loads(line) for line in log.read_text().splitlines()] \
            if log.exists() else []

    class Fakes:
        pass

    f = Fakes()
    f.aria2, f.hf, f.calls, f.payload = str(aria2), str(hf), calls, payload
    return f


def _run(monkeypatch, capsys, *argv) -> tuple[int, str, str]:
    monkeypatch.setattr("sys.argv", ["gguf-fetch", *argv])
    rc = fetch.main()
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_hf_gone_judges_by_size_only_and_says_so(site, monkeypatch, tmp_path, capsys, fakes):
    rc, out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                        "--fit", "--dry-run", "--dir", str(tmp_path / "m"),
                        "--aria2-bin", fakes.aria2)
    assert rc == 0
    assert f"({REV})" in out                        # main ではなく固定したリビジョン
    assert "Pirate Face" in out and "magnet listed" in out
    assert "by file size only" in out               # ヘッダが読めない = 粗い判定
    assert "Hugging Face does not return" in err
    assert "download via: swarm" in out
    assert "--bt-metadata-only=true" in out and "--select-file=<n,...>" in out
    # --dry-run は本体を落とさない。torrent の中身を見るメタデータ取得 (数十 KiB) だけ
    # は、判定を正しくするために行う
    assert all("--bt-metadata-only=true" in c for c in fakes.calls())
    assert len(fakes.calls()) == 1
    assert not any("resolve" in p for p in site.seen)  # ヘッダを取りに行かない


def test_a_bare_repo_id_falls_back_when_hf_has_dropped_it(site, monkeypatch, tmp_path,
                                                          capsys, fakes):
    site.hf_status = 401                             # 匿名での「無い」は 401
    rc, out, err = _run(monkeypatch, capsys, REPO, "--vram", "24", "--json",
                        "--dir", str(tmp_path / "m"))
    assert rc == 0
    assert "no longer available on Hugging Face" in err
    data = json.loads(out)
    assert data["source"] == "pirateface" and data["revision"] == REV
    assert data["sizes_approximate"] is True
    assert data["magnet"].startswith("magnet:?xt=urn:btih:" + INFOHASH)
    assert {c["label"] for c in data["candidates"]} == {"Q4_K_M", "Q8_0"}


def test_a_bare_repo_id_does_not_fall_back_on_a_network_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1")     # 繋がらない
    monkeypatch.setenv("PIRATEFACE_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(fetch._hardware, "detect", lambda _b: _hardware.Hardware(
        gpus=[], ram_gib=64.0, physical_cores=8, logical_cores=16, unified_memory=False))
    monkeypatch.setattr("sys.argv", ["gguf-fetch", REPO, "--vram", "24"])
    with pytest.raises(SystemExit) as exc:
        fetch.main()
    # 繋がらないのは「消えた」ではない。Pirate Face には切り替えず、HF の失敗で止まる
    assert "could not read" in str(exc.value)
    assert "Pirate Face" not in str(exc.value)


def test_a_pirateface_url_goes_straight_to_pirateface(site, monkeypatch, tmp_path, capsys):
    site.hf_status = 404
    rc, out, _err = _run(monkeypatch, capsys, f"https://pirateface.co/{REPO}",
                         "--vram", "24", "--json")
    assert rc == 0 and json.loads(out)["source"] == "pirateface"


def test_hf_alive_gives_exact_sizes_and_keeps_the_hf_path(site, monkeypatch, tmp_path, capsys):
    site.hf_api = {"siblings": [
        {"rfilename": "M-Q4_K_M.gguf", "size": 11}, {"rfilename": "M-Q8_0.gguf", "size": 22}]}
    rc, out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface",
                         "--vram", "24", "--json", "--probe", "none")
    data = json.loads(out)
    assert rc == 0 and data["source"] == "pirateface"
    assert data["sizes_approximate"] is False       # HF の正確なサイズを使った
    assert {c["size_bytes"] for c in data["candidates"]} == {11, 22}
    assert any(p.startswith(f"/api/models/{REPO}/revision/{REV}") for p in site.seen)


def test_swarm_download_selects_flattens_and_verifies(site, monkeypatch, tmp_path,
                                                      capsys, fakes):
    dest_root = tmp_path / "models"
    rc, out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--yes", "--dir", str(dest_root),
                         "--aria2-bin", fakes.aria2)
    assert rc == 0, out
    dest = dest_root / "Model-GGUF"
    # Q4_K_M と、自動で付く mmproj。Q8_0 は落とさない
    assert sorted(p.name for p in dest.iterdir()) == ["M-Q4_K_M.gguf", "mmproj-M-F16.gguf"]
    assert (dest / "M-Q4_K_M.gguf").read_bytes() == CONTENT["M-Q4_K_M.gguf"]
    meta, dl = fakes.calls()
    assert "--bt-metadata-only=true" in meta
    assert "--select-file=1,3" in dl                 # 番号は torrent のファイル順
    assert "SHA-256 matches" in out and "2 file(s)" in out
    assert "done ->" in out


def test_a_corrupted_download_is_not_reported_as_done(site, monkeypatch, tmp_path,
                                                      capsys, fakes):
    fakes.payload.write_text(json.dumps(
        {n: (b"bad" if n == "M-Q4_K_M.gguf" else d).hex() for n, d in CONTENT.items()}))
    rc, out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                        "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                        "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert rc == 1
    assert "does NOT match" in err and "M-Q4_K_M.gguf" in err
    assert "done ->" not in out


def test_auto_falls_back_to_hf_at_the_pinned_revision(site, monkeypatch, tmp_path,
                                                      capsys, fakes):
    monkeypatch.setenv("FAKE_FAIL", "1")             # aria2c が失敗する
    rc, _out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                        "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                        "--dir", str(tmp_path / "models"),
                        "--aria2-bin", fakes.aria2, "--hf-bin", fakes.hf)
    assert rc == 0
    assert "falling back to hf download" in err
    hf_call = fakes.calls()[-1]
    assert hf_call[:3] == ["download", REPO, "M-Q4_K_M.gguf"]
    assert hf_call[hf_call.index("--revision") + 1] == REV   # main ではない


def test_via_swarm_does_not_fall_back(site, monkeypatch, tmp_path, capsys, fakes):
    monkeypatch.setenv("FAKE_FAIL", "1")
    rc, _out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--mmproj", "none", "--yes", "--via", "swarm",
                         "--dir", str(tmp_path / "models"),
                         "--aria2-bin", fakes.aria2, "--hf-bin", fakes.hf)
    assert rc == 7                                   # aria2c の終了コードをそのまま返す
    assert "could not get the torrent metadata" in err
    assert all(c[0] != "download" for c in fakes.calls())


def test_via_hf_skips_the_swarm(site, monkeypatch, tmp_path, capsys, fakes):
    rc, _out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                          "--pick", "Q4_K_M", "--mmproj", "none", "--yes", "--via", "hf",
                          "--dir", str(tmp_path / "models"),
                          "--aria2-bin", fakes.aria2, "--hf-bin", fakes.hf)
    assert rc == 0
    assert [c[0] for c in fakes.calls()] == ["download"]


def test_no_magnet_means_hf_only(site, monkeypatch, tmp_path, capsys, fakes):
    site.page = make_page(default_rows(), magnet=False)
    rc, _out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--mmproj", "none", "--yes", "--via", "swarm",
                         "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert rc == 127 and "no magnet is listed" in err
    assert not fakes.calls()


def test_missing_tools_are_explained(site, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))        # aria2c も hf も見つからない
    rc, _out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                         "--dir", str(tmp_path / "models"))
    assert rc == 127 and "neither aria2c nor hf" in err


def test_a_file_missing_from_the_torrent_is_dropped_before_judging(
        site, monkeypatch, tmp_path, capsys, fakes):
    # ページには載っているが torrent には無い。実物 (Ornith-1.5-9B) では、ページに
    # 12 量子化が載っていて torrent には Q4_K_M の 1 本しか入っていなかった
    Path(os.environ["FAKE_TORRENT"]).write_bytes(
        make_torrent({"M-Q8_0.gguf": CONTENT["M-Q8_0.gguf"]}))
    rc, out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                        "--fit", "--dry-run", "--dir", str(tmp_path / "m"),
                        "--aria2-bin", fakes.aria2)
    assert rc == 0
    assert "not in the torrent, so they are left out" in err
    assert "M-Q8_0.gguf" in err                      # torrent にあるものを言う
    assert "Q4_K_M" not in out                       # 判定表にも出さない
    assert "Q8_0" in out


def test_picking_a_file_the_torrent_lacks_does_not_start_a_transfer(
        site, monkeypatch, tmp_path, capsys, fakes):
    Path(os.environ["FAKE_TORRENT"]).write_bytes(
        make_torrent({"M-Q8_0.gguf": CONTENT["M-Q8_0.gguf"]}))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
             "--pick", "Q4_K_M", "--mmproj", "none", "--yes", "--via", "swarm",
             "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert "matched nothing" in str(exc.value) and "Available: Q8_0" in str(exc.value)
    assert all("--select-file" not in " ".join(c) for c in fakes.calls())


def test_a_torrent_with_none_of_the_listed_files_stops(
        site, monkeypatch, tmp_path, capsys, fakes):
    Path(os.environ["FAKE_TORRENT"]).write_bytes(make_torrent({"other.txt": b"x"}))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
             "--fit", "--dry-run", "--dir", str(tmp_path / "m"),
             "--aria2-bin", fakes.aria2)
    assert "holds none of the GGUF files" in str(exc.value)


def test_judging_still_runs_when_the_torrent_cannot_be_read_first(
        site, monkeypatch, tmp_path, capsys, fakes):
    monkeypatch.setenv("FAKE_FAIL", "1")             # peer が居ない
    rc, out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                        "--fit", "--dry-run", "--dir", str(tmp_path / "m"),
                        "--aria2-bin", fakes.aria2)
    assert rc == 0
    assert "could not read the torrent contents before judging" in err
    assert "Q4_K_M" in out and "Q8_0" in out         # 絞らず、従来どおりの表


def test_hf_alive_does_not_wait_for_the_swarm_before_judging(
        site, monkeypatch, tmp_path, capsys, fakes):
    # HF から落とせる経路が残っているなら、どちらでも取れるので torrent は見に行かない
    site.hf_api = {"siblings": [
        {"rfilename": "M-Q4_K_M.gguf", "size": 11}, {"rfilename": "M-Q8_0.gguf", "size": 22}]}
    rc, _out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface",
                          "--vram", "24", "--json", "--probe", "none",
                          "--aria2-bin", fakes.aria2)
    assert rc == 0 and not fakes.calls()


def test_the_torrent_read_for_judging_is_reused_for_the_download(
        site, monkeypatch, tmp_path, capsys, fakes):
    rc, _out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                          "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                          "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert rc == 0
    kinds = ["meta" if "--bt-metadata-only=true" in c else "dl" for c in fakes.calls()]
    assert kinds == ["meta", "dl"]                   # メタデータは1回だけ


def test_a_page_that_cannot_be_read_stops_with_the_url(site, monkeypatch, tmp_path, capsys):
    site.page = "<html>redesigned</html>"
    monkeypatch.setattr("sys.argv", ["gguf-fetch", REPO, "--source", "pirateface",
                                     "--vram", "24"])
    with pytest.raises(SystemExit) as exc:
        fetch.main()
    assert "could not read" in str(exc.value) and "layout changed" in str(exc.value)


def test_hf_source_is_untouched_by_default(site, monkeypatch, tmp_path, capsys):
    """HF が生きていて repo ID だけなら、Pirate Face には一切触れない."""
    site.hf_api = {"siblings": [{"rfilename": "M-Q4_K_M.gguf", "size": 5}]}
    rc, out, _err = _run(monkeypatch, capsys, REPO, "--vram", "24", "--json",
                         "--probe", "none")
    assert rc == 0 and json.loads(out)["source"] == "hf"
    assert f"/{REPO}" not in site.seen              # ページを取りに行っていない


def test_a_second_run_does_not_download_again(site, monkeypatch, tmp_path, capsys, fakes):
    argv = (REPO, "--source", "pirateface", "--vram", "24", "--pick", "Q4_K_M",
            "--mmproj", "none", "--yes", "--dir", str(tmp_path / "models"),
            "--aria2-bin", fakes.aria2)
    assert _run(monkeypatch, capsys, *argv)[0] == 0
    first = len(fakes.calls())
    rc, out, _err = _run(monkeypatch, capsys, *argv)
    assert rc == 0
    assert "already downloaded" in out and "M-Q4_K_M.gguf" in out
    # 2回目は本体を落とし直さない (メタデータを読むだけ。--select-file は無い)
    new_calls = fakes.calls()[first:]
    assert all("--bt-metadata-only=true" in c for c in new_calls)


def test_a_damaged_earlier_copy_is_downloaded_again(site, monkeypatch, tmp_path, capsys, fakes):
    dest = tmp_path / "models" / "Model-GGUF"
    dest.mkdir(parents=True)
    (dest / "M-Q4_K_M.gguf").write_bytes(b"half a file")   # SHA が合わない
    rc, out, _err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                         "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert rc == 0 and "already downloaded" not in out
    assert (dest / "M-Q4_K_M.gguf").read_bytes() == CONTENT["M-Q4_K_M.gguf"]
    assert (dest / "M-Q4_K_M.gguf.old").read_bytes() == b"half a file"   # 消さずに退避


def test_neighbour_fragments_are_reported_not_deleted(site, monkeypatch, tmp_path,
                                                      capsys, fakes):
    monkeypatch.setenv("FAKE_NEIGHBOUR", "1")
    rc, _out, err = _run(monkeypatch, capsys, REPO, "--source", "pirateface", "--vram", "24",
                         "--pick", "Q4_K_M", "--mmproj", "none", "--yes",
                         "--dir", str(tmp_path / "models"), "--aria2-bin", fakes.aria2)
    assert rc == 0
    left = tmp_path / "models" / "Model-GGUF" / TOP
    assert (left / "M-Q8_0.gguf").read_bytes() == b"partial piece"
    assert str(left) in err and "fragments" in err
