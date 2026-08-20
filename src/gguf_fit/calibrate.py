"""このマシンで実際に測って、VRAM の見積り式を決める.

**なぜ測るのか。** 理論値は当てにならない。GGUF から計算した KV (f16) は
68.0 KB/token だが、実測は 69.1 KB/token。q8_0 の理論比は 34/64 = 0.531 の
はずが、実測は 0.62。しかも**この差は環境で変わる** —— llama.cpp の
バージョン、``--spec-type``、``-fa``、バックエンド (CUDA / ROCm / Metal)、
バッチサイズ。量子化の実装が改良されれば比も動く。定数を焼き込むと、
その瞬間から嘘になっていく。

**関係は完全な直線**なので、2点測れば確定する::

    使用量 = 切片 + ctx x 1トークンあたりのバイト数

**いつ測るかで答えが変わる。** ここが一度ハマった所なので書いておく。
同じ ctx 65,536 / 同じサーバで、起動前からの増分がこうなった::

    ロード直後 (推論なし)           23,922 MiB
    8トークンのリクエスト1回        24,022 MiB   (+100)
    2048トークンのリクエスト1回     24,032 MiB   (+110)
    ベンチマークを回している間      24,064 MiB   (+142、±50 で揺れる)

**どの段の差も、f16 と q8_0 でまったく同じ値**だった (+100/+110/+142)。
KV の型にも ctx にも依存しないので、KV ではなく**推論そのものが確保する分**。

プロンプトを 8 → 2048 トークン (256倍) にしても +10 MiB しか増えない。
つまり**確保はほぼ「推論を1回でも通したか」で決まり、プロンプト長では
決まらない**。

まずかったのは、条件の違う点を混ぜて直線を引いたこと。ctx 32,768 を
ロード直後に、ctx 65,536 を推論中に測って傾きを出すと、本来は切片に
乗るべき定数が傾きに化けて **73.5 KB/token** になる。条件をそろえた
今の値は **69.1 KB/token** で、理論値 68.0 の 1.016 倍。こちらが正しい。

定数であることは実測で確かめてある。ウォームアップを入れて測り直すと、
4点とも同じ量 (8トークンで +100、2048トークンで +110) だけ動き、**傾きは
1 バイトも変わらず**切片だけが
19.045 → 19.15 GiB に上がった。傾きは測るタイミングに対して頑健で、
切片だけが動く —— 見積り式の形が正しいことの裏付けでもある。

なので、このツールは**1点につき2回測る**::

    ロード直後  →  リクエストを1回通す  →  もう一度

当てはめには**後者**を使う。「入るか」を判断したいのだから、走っている
ときの値でなければ意味がない。差は結果に併記する。

**残り 32 MiB は埋めていない。** 既定はバッチ1つ分のプロンプトを流す
(``--warmup-tokens``、既定 2048 = ``--batch-size`` と同じ) が、それでも
本番の +142 には 32 MiB 届かない。プロンプトを伸ばしても頭打ちになることは
上のとおり測ってあるので、**これは「1回のリクエストでは再現しない分」**
—— 長時間の連続実行で出てくるもの (スロットの回転、KV の defrag、
投機デコードのグラフ違いなど) と考えている。追いかけずに、既定の
``overhead`` 1.0 GiB が実測 0.68 に対して持っている余裕で見る。

**差分で測る。** 起動前後の差を取るので、デスクトップや他プロセスの使用量が
混ざらない。``--device CUDA0`` と nvidia-smi の並び順が一致しない問題も、
「一番増えた GPU」を見ることで回避できる。

**再現性は確認済み。** 同じ条件で2回通したところ、4点とも 1 MiB の差もなく
一致した (21,822 / 24,032 / 20,948 / 22,326)。ラン間のばらつきは無いので、
1回測れば足りる。値が動いたときは環境が変わったと考えてよい。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from . import _hardware

GIB = 1024 ** 3

#: 起動を待つ上限 (秒)。大きいモデルのロードは分単位になりうる
LOAD_TIMEOUT_S = 600
#: /health が応答してから、確保が落ち着くまで待つ (秒)
SETTLE_S = 4
#: 何秒おきに見るか
POLL_S = 2
#: ウォームアップで生成するトークン数。デコード側を通せばよいので少しでよい
WARMUP_TOKENS = 8
#: ウォームアップで流すプロンプトのトークン数。**既定の --batch-size と同じ**。
#: 8 -> 2048 で +10 MiB。256倍にして 10 MiB なので、ここから伸ばしても意味がない。
WARMUP_PROMPT_TOKENS = 2048
#: プロンプトに使うトークンID。どの語彙にも実在する低い ID なら中身は何でもよい
WARMUP_TOKEN_ID = 100
#: ウォームアップの応答を待つ上限 (秒)
WARMUP_TIMEOUT_S = 300


class Point(NamedTuple):
    """1回の測定。**起動前からの増分** (MiB)."""

    kv_mode: str      # "f16" / "q8_0"
    ctx: int
    used_mib: int     # 推論を1回通したあと。**当てはめに使うのはこちら**
    loaded_mib: int | None = None   # ロード直後 (推論なし)。差を見せるため

    @property
    def warmup_mib(self) -> int | None:
        """推論を1回通して増えた分。測っていなければ None."""
        if self.loaded_mib is None:
            return None
        return self.used_mib - self.loaded_mib


class Fit(NamedTuple):
    """1つの kv_mode についての直線。"""

    kv_mode: str
    bytes_per_token: float
    intercept_gib: float
    n_points: int
    max_error_mib: float

    def predict_gib(self, ctx: int) -> float:
        return self.intercept_gib + self.bytes_per_token * ctx / GIB


def fit_points(points: list[Point]) -> Fit:
    """同じ kv_mode の測定から直線を出す.

    2点なら厳密解。3点以上なら最小二乗。**1点では出せない** —— 傾きと切片を
    分離できないため。実際に ctx 64k の1点だけで判断しようとして、
    「KV 比 0.61」と「オーバーヘッド +0.33」のどちらとも決められなかった。

    **点は同じ条件で測ること。** ロード直後の点と推論中の点を混ぜると、
    本来は切片に乗る差が傾きに化ける (73.5 と 69.1 の食い違いはこれだった)。
    """
    kv_modes = {p.kv_mode for p in points}
    if len(kv_modes) != 1:
        raise ValueError(f"points must share one kv_mode, got {sorted(kv_modes)}")
    if len({p.ctx for p in points}) < 2:
        raise ValueError(
            "need at least two different --ctx values; "
            "one point cannot separate the slope from the intercept")

    n = len(points)
    xs = [p.ctx for p in points]
    ys = [p.used_mib * 1024 * 1024 for p in points]   # bytes
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = (my - slope * mx) / GIB
    err = max(abs(intercept + slope * p.ctx / GIB - p.used_mib / 1024) * 1024
              for p in points)
    return Fit(points[0].kv_mode, slope, intercept, n, err)


def _gpu_used_mib() -> dict[int, int]:
    """GPU ごとの使用量 (MiB)。取れなければ空。"""
    out = _hardware._run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"])
    if not out:
        return {}
    used = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                used[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    return used


def _biggest_delta(before: dict[int, int], after: dict[int, int]) -> int:
    """一番増えた GPU の増分 (MiB).

    **どの GPU かを当てにいかない。**nvidia-smi の並び順と CUDA のデバイス
    番号は一致しないので、「増えたほう」を見るのが確実。
    """
    if not before or not after:
        return 0
    return max((after.get(i, 0) - v for i, v in before.items()), default=0)


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    """GET して 2xx が返るか。落ちていれば False."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(port: int, proc: subprocess.Popen, *,
                     timeout_s: int = LOAD_TIMEOUT_S) -> float:
    """``/health`` が応答するまで待ち、かかった秒数を返す.

    **VRAM が動かなくなったことをロード完了と見なさない。** 大きい ctx ほど
    確保に間があき、「静かになった」が早く成立して途中の値を拾いうる。
    サーバ自身が「用意できた」と言うまで待つほうが確実。
    """
    url = f"http://127.0.0.1:{port}/health"
    waited = 0.0
    while waited < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}). "
                f"Run the command by hand to see why.")
        if _http_ok(url):
            return waited
        time.sleep(POLL_S)
        waited += POLL_S
    raise RuntimeError(f"llama-server did not answer /health within {timeout_s}s")


def warmup_payloads(prompt_tokens: int = WARMUP_PROMPT_TOKENS,
                    predict: int = WARMUP_TOKENS) -> list[dict]:
    """ウォームアップに投げる本文の候補。**上から順に試す**.

    llama.cpp の ``/completion`` は prompt にトークンIDの配列を取れる。
    文字列だと「何トークンになるか」が語彙次第で分からないので、**バッチを
    きっちり1つ埋めたいときは配列のほうが確実**。受け付けないサーバのために
    文字列版も残しておく (こちらは長さが概算になる)。
    """
    return [
        {"prompt": [WARMUP_TOKEN_ID] * prompt_tokens,
         "n_predict": predict, "temperature": 0},
        {"prompt": "lorem ipsum dolor sit amet " * max(1, prompt_tokens // 5),
         "n_predict": predict, "temperature": 0},
    ]


def warm_up(port: int, *, prompt_tokens: int = WARMUP_PROMPT_TOKENS,
            tokens: int = WARMUP_TOKENS,
            timeout_s: int = WARMUP_TIMEOUT_S) -> bool:
    """リクエストを1回だけ通す。成功したら True.

    **推論そのものが確保する分**を出させるため。ロード直後との差は、実測で
    ctx 65,536 のとき 8トークンで +100、2048トークンで +110、本番のベンチ
    マーク中で +142 MiB (いずれも f16 / q8_0 で同じ値)。プロンプト長を
    256倍にしても +10 しか動かないので、**バッチ1つ分で頭打ち**とみて
    既定を 2048 にしてある。
    """
    for payload in warmup_payloads(prompt_tokens, tokens):
        body = json.dumps(payload).encode()
        for path in ("/completion", "/v1/completions"):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    if 200 <= r.status < 300:
                        r.read()
                        return True
            except (urllib.error.URLError, OSError, ValueError):
                continue
    return False


def measure_point(launch_cmd: list[str], kv_mode: str, ctx: int, *,
                  port: int, warmup: bool = True,
                  warmup_prompt_tokens: int = WARMUP_PROMPT_TOKENS,
                  verbose: bool = True) -> Point:
    """サーバを起動して VRAM の増分を測り、落とす.

    ``warmup=True`` なら、ロード直後と**リクエストを1回通したあと**の両方を
    測る。当てはめに使うのは後者 —— 「入るか」を見たいのだから、走っている
    ときの値でなければ意味がない。
    """
    before = _gpu_used_mib()
    if not before:
        raise RuntimeError("nvidia-smi did not report anything; cannot measure")

    if verbose:
        print(f"  launching: kv={kv_mode} ctx={ctx:,}", file=sys.stderr)
    proc = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        took = wait_until_ready(port, proc)
        time.sleep(SETTLE_S)
        loaded = _biggest_delta(before, _gpu_used_mib())
        used = loaded
        if warmup:
            if warm_up(port, prompt_tokens=warmup_prompt_tokens):
                time.sleep(SETTLE_S)
                used = max(loaded, _biggest_delta(before, _gpu_used_mib()))
            elif verbose:
                print("    (warm-up request failed; using the load-time figure)",
                      file=sys.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    if verbose:
        extra = f"  (+{used - loaded} after one request)" if used != loaded else ""
        print(f"    -> {used:,} MiB{extra}   ready in {took:.0f}s", file=sys.stderr)
    return Point(kv_mode, ctx, used, loaded)


def render_fits(fits: list[Fit], points: list[Point] | None = None) -> str:
    """較正結果を人が読める形に."""
    lines = ["calibration result", ""]
    for f in fits:
        lines.append(f"  {f.kv_mode:<5} {f.bytes_per_token / 1024:6.1f} KB/token"
                     f"   intercept {f.intercept_gib:5.2f} GiB"
                     f"   ({f.n_points} points, max error {f.max_error_mib:.0f} MiB)")
    by_mode = {f.kv_mode: f for f in fits}
    if "f16" in by_mode and "q8_0" in by_mode:
        ratio = by_mode["q8_0"].bytes_per_token / by_mode["f16"].bytes_per_token
        lines.append("")
        lines.append(f"  q8_0 / f16 = {ratio:.3f}"
                     f"   (the naive 34/64-byte figure would say 0.531)")
    warm = [p.warmup_mib for p in (points or []) if p.warmup_mib is not None]
    if warm:
        lo, hi = min(warm), max(warm)
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        lines.append("")
        lines.append(f"  one request added {span} MiB on top of the load-time "
                     f"figure; that is what these numbers include")
    return "\n".join(lines)


#: 書き込んだブロックの目印。**再実行時にここから下を差し替える**ので、
#: この行より下に手で何かを足さないこと。
BLOCK_MARKER = "# --- calibrated on this machine (gguf-calibrate) ---"

#: 較正ブロックが持つキー。**再実行で差し替える対象**。ここに足し忘れると、
#: 2回目の --write-config で同じキーが2つ並んで TOML が壊れる
_CALIBRATED_KEYS = ("kv_f16_bytes", "kv_q8_bytes",
                    "kv_measured_on", "kv_derived_f16_bytes")


def render_toml_fragment(fits: list[Fit], measured_on: str | None = None,
                         derived_f16: float | None = None) -> str:
    """``gguf-fit.toml`` に貼れる形。**測った事実として書く**.

    ``measured_on`` / ``derived_f16`` は「**どのモデルで測ったか**」の記録。
    ``kv_f16_bytes`` は KB/token の値で、これは**層構造で決まるモデル固有の
    数字**なのに、設定ファイルに書くと以降すべてのモデルに当たる。実際に
    起きた: Qwen3.8-27B で測った 69.1 KB/token が、KV の単価が 22.0 KB/token
    しかない Ornith-1.5-35B の計画にも使われ、最大 ctx を3倍近く低く出した。

    そこで、測ったモデルの**GGUF からの計算値**も一緒に書いておく。
    ``gguf-plan`` / ``gguf-fetch`` は、いま見ているモデルの計算値がこれと
    大きく違えば「その較正値はこのモデルのものではない」と言える。
    """
    lines = [BLOCK_MARKER,
             "# Measured after one request, not derived. Re-run after changing",
             "# llama.cpp, the backend, or the launch flags."]
    for f in fits:
        key = "kv_f16_bytes" if f.kv_mode == "f16" else "kv_q8_bytes"
        lines.append(f"{key} = {f.bytes_per_token:.0f}"
                     f"   # {f.bytes_per_token / 1024:.1f} KB/token, "
                     f"{f.n_points} points")
    if measured_on:
        lines.append(f'kv_measured_on = "{measured_on}"'
                     "   # これらの値はこのモデルのものです")
    if derived_f16:
        lines.append(f"kv_derived_f16_bytes = {derived_f16:.0f}"
                     f"   # そのモデルの GGUF からの計算値 "
                     f"({derived_f16 / 1024:.1f} KB/token)。"
                     "別モデルに当たっていないかの照合用")
    if fits:
        oh = sum(f.intercept_gib for f in fits) / len(fits)
        lines.append(f"# intercept was {oh:.2f} GiB "
                     f"(model file + fixed buffers; gguf-plan subtracts the file)")
    return "\n".join(lines) + "\n"


def strip_calibrated_block(text: str) -> str:
    """既に書いてある較正ブロックと ``kv_*_bytes`` の行を取り除く.

    **同じキーを2回書くと TOML は壊れる。**追記していくだけだと、2回目の
    ``--write-config`` で設定ファイルが読めなくなる。行単位で消すのは、
    手で書いたコメントや他のキーをそのまま残したいため。
    """
    out, skipping = [], False
    for line in text.splitlines():
        bare = line.strip()
        if bare == BLOCK_MARKER:
            skipping = True
            continue
        if skipping:
            # ブロックの中身はコメントと kv_*_bytes だけ。他が来たら抜ける
            if (not bare or bare.startswith("#")
                    or bare.startswith(_CALIBRATED_KEYS)):
                continue
            skipping = False
        if bare.startswith(_CALIBRATED_KEYS):
            continue
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def derived_f16_bytes(model_path: str) -> tuple[str, float] | None:
    """測ったモデルの「GGUF からの計算値」を読む。取れなければ ``None``.

    較正値が**どのモデルのものか**を後から照合するために書き残す。
    ここで失敗しても較正そのものは成立するので、黙って諦めてよい
    （書けるものだけ書く）。
    """
    try:
        from .probe import probe  # noqa: PLC0415 - 書き出すときだけ要る

        rec = probe(Path(model_path))
        per_token = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
        if not per_token:
            return None
        return Path(model_path).name, float(per_token)
    except Exception:  # noqa: BLE001 - 記録が取れなくても較正は成立する
        return None


def write_config(fits: list[Fit], path: Path, model_path: str | None = None) -> bool:
    """較正結果を設定ファイルに書く。新規作成したら True.

    **書く前に TOML として成立することを確かめる。**壊れたファイルを置いて
    「設定が効かない」に気づけないのが一番困る。成立しなければ例外を投げて
    元のファイルには触らない。
    """
    existed = path.is_file()
    head = strip_calibrated_block(path.read_text(encoding="utf-8")) if existed else ""
    origin = derived_f16_bytes(model_path) if model_path else None
    body = render_toml_fragment(fits, *(origin or (None, None)))
    text = f"{head}\n\n{body}" if head else body

    tomllib.loads(text)          # 壊れていればここで止まる
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return not existed


def build_launch_cmd(binary: str, model_path: str, ctx: int, kv_mode: str,
                     device: str | None, threads: int | None,
                     port: int, extra: list[str] | None = None) -> list[str]:
    """測定用の起動コマンド。**gguf-plan が出すものと同じ形にそろえる**.

    ここが本番と違うと、測った値が本番に当てはまらない。
    """
    cmd = [binary, "-m", model_path, "--port", str(port),
           "-ngl", "99", "-fa", "on",
           "--ctx-size", str(ctx), "--parallel", "1",
           "--batch-size", "2048", "--ubatch-size", "512"]
    if device:
        cmd += ["--device", device]
    if kv_mode == "q8_0":
        cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
    if threads:
        cmd += ["--threads", str(threads)]
    return cmd + list(extra or [])


def main() -> int:
    """``gguf-calibrate``: 実測して見積り式を決める."""
    # main() でしか使わないものはここで import する。ライブラリとして
    # 使うときに argparse まで引きずらないため。
    import argparse  # noqa: PLC0415

    from ._config import load_config, resolve  # noqa: PLC0415
    from ._hardware import detect  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description="Measure this machine's VRAM behaviour and derive the "
                    "estimate used by gguf-plan")
    ap.add_argument("--model", required=True, help="path to a .gguf to measure with")
    ap.add_argument("--llama-server", default=None, dest="llama_server",
                    help="llama-server binary to launch")
    ap.add_argument("--ctx", default="32768,65536",
                    help="comma-separated context sizes; at least two "
                         "(default: 32768,65536)")
    ap.add_argument("--kv", default="f16,q8_0",
                    help="comma-separated KV types (default: f16,q8_0)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--port", type=int, default=18085,
                    help="port for the throwaway servers (default: 18085)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="read VRAM straight after loading, without sending a "
                         "request. Faster, but ~110 MiB low: inference itself "
                         "allocates, and that has to fit too.")
    ap.add_argument("--warmup-tokens", type=int, default=WARMUP_PROMPT_TOKENS,
                    dest="warmup_tokens",
                    help=f"prompt length for the warm-up request, in tokens "
                         f"(default: {WARMUP_PROMPT_TOKENS}, one --batch-size). "
                         f"Raise it if you run with a bigger batch.")
    ap.add_argument("--extra", default=None,
                    help="extra flags passed through verbatim, e.g. "
                         "'--spec-type draft-mtp'. Use the same ones you run with.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--write-config", nargs="?", const="", default=None,
                    dest="write_config", metavar="PATH",
                    help="write kv_f16_bytes / kv_q8_bytes into the config file "
                         "instead of leaving you to copy them. Defaults to the "
                         "config file in effect, or ./gguf-fit.toml. Re-running "
                         "replaces the block; everything else is kept.")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    hw = detect(args.llama_server or cfg.get("llama_server", "llama-server"))
    binary = resolve("llama_server", args.llama_server, cfg, "llama-server").value
    device = resolve("device", args.device, cfg,
                     detected=hw.suggested_device()).value
    threads = resolve("threads", args.threads, cfg,
                      detected=hw.suggested_threads()).value

    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    kvs = [x.strip() for x in args.kv.split(",") if x.strip()]
    if len(ctxs) < 2:
        ap.error("need at least two --ctx values to separate slope from intercept")
    extra = args.extra.split() if args.extra else []
    warmup = not args.no_warmup

    tail = (f"one {args.warmup_tokens}-token request each" if warmup
            else "no inference at all")
    print(f"measuring {len(ctxs) * len(kvs)} points "
          f"(each server is started, measured and killed; {tail})",
          file=sys.stderr)
    fits, measured = [], []
    for kv_mode in kvs:
        points = []
        for ctx in ctxs:
            cmd = build_launch_cmd(str(binary), args.model, ctx, kv_mode,
                                   device, threads, args.port, extra)
            try:
                points.append(measure_point(
                    cmd, kv_mode, ctx, port=args.port, warmup=warmup,
                    warmup_prompt_tokens=args.warmup_tokens))
            except (RuntimeError, OSError) as e:
                print(f"!! {kv_mode} ctx {ctx}: {e}", file=sys.stderr)
        measured += points
        if len(points) >= 2:
            fits.append(fit_points(points))
        else:
            print(f"!! {kv_mode}: not enough points, skipped", file=sys.stderr)

    if not fits:
        sys.exit("nothing measured")
    print()
    print(render_fits(fits, measured))
    print()
    if args.write_config is None:
        print("--- paste into gguf-fit.toml ---")
        print(render_toml_fragment(fits), end="")
        return 0

    target = Path(args.write_config) if args.write_config else (
        cfg_path or Path("gguf-fit.toml"))
    try:
        created = write_config(fits, target, args.model)
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"could not write {target}: {e}")
    print(f"[{'created' if created else 'updated'}] {target}")
    return 0
