"""設定の解決。**どの値がどこから来たかを保持する**.

優先順位は上から:

    1. CLI のフラグ          --lang ja
    2. 環境変数              GGUF_FIT_LANG=ja
    3. 設定ファイル          lang = "ja"
    4. このマシンの実測      vram / threads は nvidia-smi や /proc/cpuinfo から
    5. 組み込みの既定値      "en"

**実測を設定ファイルより下に置くのは意図的**。設定ファイルに書いた値が
勝手に上書きされたら、書いた意味がない。

設定ファイルの探索順 (最初に見つかった1つだけを使う。マージはしない):

    1. $GGUF_FIT_CONFIG           明示指定
    2. ./gguf-fit.toml            カレント (プロジェクトごとの設定)
    3. ~/.config/gguf-fit/config.toml     (Linux / macOS。XDG_CONFIG_HOME 尊重)
       %APPDATA%\\gguf-fit\\config.toml    (Windows)

**マージしないのは意図的**。複数ファイルを重ねると「この値はどこから来たのか」
が追えなくなる。このツールは設定の出どころを言えることが売りなので、
そこを曖昧にしない。``--show-config`` で確認できる。

設定ファイルの例:

    lang = "ja"
    vram = 24.0
    overhead = 1.0
    device = "CUDA0"
    port = 8085
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 のみ
    import tomli as tomllib

#: 設定ファイルに書ける項目と、その型。ここに無いキーは警告する
KNOWN_KEYS: dict[str, type] = {
    "lang": str,
    "vram": float,
    "overhead": float,
    "device": str,
    "port": int,
    "threads": int,
    "llama_server": str,
    "llama_servers": list,
    # gguf-calibrate が書く実測値 (B/token)。理論値より優先する
    "kv_f16_bytes": float,
    "kv_q8_bytes": float,
    "model_path": str,
    # gguf-fetch: 落とし先と、使う hf コマンド
    "models_dir": str,
    "hf_bin": str,
}

#: 環境変数名は GGUF_FIT_<大文字> で固定
ENV_PREFIX = "GGUF_FIT_"

#: このマシンを見れば決まる項目。``--refresh`` で設定ファイルの値を捨てて
#: 取り直す対象。**設定ファイルは実測より強いので、一度書くと居座る。**
#: ハードウェアを入れ替えたときに更新する手段が要る。
DETECTABLE_KEYS = ("vram", "threads", "device")


def drop_detectable(config: dict) -> dict:
    """検出で決まる項目を設定から外す (``--refresh`` 用)."""
    return {k: v for k, v in config.items() if k not in DETECTABLE_KEYS}


class Resolved(NamedTuple):
    """解決した値と、**その出どころ**."""

    value: Any
    source: str  # "cli" / "env" / "config" / "detected" / "default"


def config_search_paths(explicit: str | os.PathLike | None = None) -> list[Path]:
    """探索するパスを優先順に返す (存在チェックはしない)."""
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    env = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env:
        paths.append(Path(env))
    paths.append(Path("gguf-fit.toml"))
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        root = Path(xdg) if xdg else Path.home() / ".config"
    paths.append(root / "gguf-fit" / "config.toml")
    return paths


def load_config(explicit: str | os.PathLike | None = None) -> tuple[dict, Path | None]:
    """最初に見つかった設定ファイルを読む。戻り値は (設定, 読んだパス).

    壊れた TOML や未知のキーは **黙って無視しない**。stderr に出す。
    設定ファイルが効いていないことに気づけないのが一番困るため。
    """
    for path in config_search_paths(explicit):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as e:
            print(f"!! could not read {path}: {e}", file=sys.stderr)
            return {}, None
        cleaned: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in KNOWN_KEYS:
                print(f"!! {path}: unknown key {key!r} (ignored)", file=sys.stderr)
                continue
            want = KNOWN_KEYS[key]
            try:
                cleaned[key] = ([str(x) for x in value] if want is list
                                else want(value))
            except (TypeError, ValueError):
                print(f"!! {path}: {key!r} should be {want.__name__} (ignored)",
                      file=sys.stderr)
        return cleaned, path
    return {}, None


def resolve(key: str, cli_value: Any, config: dict, default: Any = None,
            detected: Any = None) -> Resolved:
    """1項目を優先順位に従って解決し、出どころも返す.

    ``detected`` はこのマシンを見て得た値 (VRAM 容量、物理コア数など)。
    **設定ファイルより下、組み込み既定より上**に置く。
    """
    if cli_value is not None:
        return Resolved(cli_value, "cli")

    env_raw = os.environ.get(ENV_PREFIX + key.upper())
    if env_raw:
        want = KNOWN_KEYS.get(key, str)
        try:
            # リストは環境変数ではカンマ区切りで渡す
            value = ([x.strip() for x in env_raw.split(",") if x.strip()]
                     if want is list else want(env_raw))
            return Resolved(value, "env")
        except (TypeError, ValueError):
            print(f"!! {ENV_PREFIX}{key.upper()}={env_raw!r} is not a "
                  f"{want.__name__} (ignored)", file=sys.stderr)

    if key in config:
        return Resolved(config[key], "config")

    if detected is not None:
        return Resolved(detected, "detected")

    return Resolved(default, "default")


def resolve_llama_servers(cli_values: list[str] | None, config: dict) -> Resolved:
    """``llama-server`` の実体を決める。複数のビルドを並べられる.

    **単数形 ``llama_server`` と複数形 ``llama_servers`` の両方を受ける。**
    1本しか使っていない人に配列を書かせる理由が無く、複数ビルドを使い分けて
    いる人に1本しか書かせないわけにもいかないため。

    ここを ``gguf-plan`` と ``gguf-fetch`` で**共有している**のは、片方だけが
    ROCm ビルドを見て片方が見ない、という状態を作らないため。同じマシンで
    2つのコマンドが違う GPU を前提に計算したら、予算の数字が食い違う。
    """
    r = resolve("llama_servers", cli_values, config)
    if r.value is not None:
        return r
    single = resolve("llama_server", None, config, "llama-server")
    return single._replace(value=[single.value])


def split_repeated(values: list[str] | None) -> list[str] | None:
    """``--llama-server a --llama-server b,c`` を1つの並びにする."""
    if not values:
        return None
    out = [x.strip() for item in values for x in item.split(",") if x.strip()]
    return out or None


def render_show_config(resolved: dict[str, Resolved], path: Path | None) -> str:
    """``--show-config`` の出力。どの値がどこから来たかを一覧にする."""
    lines = ["settings in effect (cli > env > config file > detected > default)", ""]
    where = str(path) if path else "(none found)"
    lines.append(f"  config file: {where}")
    if path is None:
        lines.append("  searched:")
        lines.extend(f"    {p}" for p in config_search_paths())
    lines.append("")
    width = max((len(k) for k in resolved), default=0)
    for key, r in resolved.items():
        lines.append(f"  {key:<{width}}  {_show(key, r.value):<24} <- {r.source}")
    return "\n".join(lines)


#: 見せるときに小数点を落とすキー。**単位で決める。型では決まらない**。
#: TOML 上はどちらも float だが、24.0 GiB の ".0" には意味があり、
#: 70720.0 バイトの ".0" には無い。
INTEGRAL_KEYS = ("kv_f16_bytes", "kv_q8_bytes")


def _show(key: str, value: Any) -> str:
    """``--show-config`` での値の見せ方."""
    if value is None:
        return "(unset)"
    if key in INTEGRAL_KEYS and isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_toml(resolved: dict[str, Resolved], hw_summary: str = "") -> str:
    """いま効いている値を gguf-fit.toml の中身として書き出す.

    **検出できた値は実際に書き込む**。設定ファイルに書いてしまえば、
    別のマシンに持っていっても、nvidia-smi が使えない環境でも同じ計画が出る。

    検出のままにしておきたい項目はコメントアウトして出す。書いてある値と
    検出値のどちらが効いているのか分からなくなるのを避けるため、
    **由来を各行に残す**。
    """
    lines = [
        "# gguf-fit の設定ファイル",
        "#",
        "# !! このファイルはこのマシン固有です。**コミットしないでください。**",
        "#    VRAM やコア数が書いてあるので、別のマシンに持っていくと",
        "#    「そこには無い GPU」を前提に計画が立ちます。",
        "#",
        "# gguf-plan --write-config で生成しました。",
        "# 各行の <- は、その値がどこから来たかです。",
        "#   detected = このマシンを見て決めた値",
        "#   default  = 組み込みの既定値",
        "#   cli/env/config = あなたが指定した値",
        "#",
        "# 優先順位: CLI フラグ > 環境変数 > このファイル > 実測 > 組み込み既定",
        "# いま何が効いているかは `gguf-plan --show-config` で確認できます。",
    ]
    if hw_summary:
        lines.append("#")
        lines.extend("# " + ln for ln in hw_summary.splitlines())
    lines.append("")

    for key, r in resolved.items():
        if r.value is None:
            lines.append(f"# {key} = ...   # 取得できませんでした")
            continue
        if isinstance(r.value, str):
            value = f'"{r.value}"'
        elif isinstance(r.value, list):
            value = "[" + ", ".join(f'"{x}"' for x in r.value) + "]"
        elif isinstance(r.value, float):
            # 31.8427734375 のような生値をそのまま書かない
            value = repr(round(r.value, 2))
        else:
            value = repr(r.value)
        lines.append(f"{key} = {value}   # <- {r.source}")
    return "\n".join(lines) + "\n"
