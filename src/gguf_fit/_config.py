"""設定の解決。**どの値がどこから来たかを保持する**.

優先順位は上から:

    1. CLI のフラグ          --lang ja
    2. 環境変数              GGUF_FIT_LANG=ja
    3. 設定ファイル          lang = "ja"
    4. 組み込みの既定値      "en"

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
    "model_path": str,
}

#: 環境変数名は GGUF_FIT_<大文字> で固定
ENV_PREFIX = "GGUF_FIT_"


class Resolved(NamedTuple):
    """解決した値と、**その出どころ**."""

    value: Any
    source: str  # "cli" / "env" / "config" / "default"


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
                cleaned[key] = want(value)
            except (TypeError, ValueError):
                print(f"!! {path}: {key!r} should be {want.__name__} (ignored)",
                      file=sys.stderr)
        return cleaned, path
    return {}, None


def resolve(key: str, cli_value: Any, config: dict, default: Any = None) -> Resolved:
    """1項目を優先順位に従って解決し、出どころも返す."""
    if cli_value is not None:
        return Resolved(cli_value, "cli")

    env_raw = os.environ.get(ENV_PREFIX + key.upper())
    if env_raw:
        want = KNOWN_KEYS.get(key, str)
        try:
            return Resolved(want(env_raw), "env")
        except (TypeError, ValueError):
            print(f"!! {ENV_PREFIX}{key.upper()}={env_raw!r} is not a "
                  f"{want.__name__} (ignored)", file=sys.stderr)

    if key in config:
        return Resolved(config[key], "config")

    return Resolved(default, "default")


def render_show_config(resolved: dict[str, Resolved], path: Path | None) -> str:
    """``--show-config`` の出力。どの値がどこから来たかを一覧にする."""
    lines = ["settings in effect (cli > env > config file > default)", ""]
    where = str(path) if path else "(none found)"
    lines.append(f"  config file: {where}")
    if path is None:
        lines.append("  searched:")
        lines.extend(f"    {p}" for p in config_search_paths())
    lines.append("")
    width = max((len(k) for k in resolved), default=0)
    for key, r in resolved.items():
        shown = "(unset)" if r.value is None else r.value
        lines.append(f"  {key:<{width}}  {shown!s:<24} <- {r.source}")
    return "\n".join(lines)
