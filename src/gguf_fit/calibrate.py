"""このマシンで実際に測って、VRAM の見積り式を決める.

**なぜ測るのか。** 理論値は当てにならないことが実測で分かっている::

    GGUF から計算した KV (f16)   68.0 KB/token
    実測                          73.5 KB/token   (1.08 倍)

    q8_0 の理論比 (34/64 バイト)   0.531
    実測                           0.646

理論より 1〜3割ずれる。しかも**この差は環境で変わる**。llama.cpp の
バージョン、``--spec-type``、``-fa``、バックエンド (CUDA / ROCm / Metal)、
バッチサイズ。量子化の実装が改良されれば比も動く。定数を焼き込むと、
その瞬間から嘘になっていく。

**関係は完全な直線**なので、2点測れば確定する::

    使用量 = 切片 + ctx x 1トークンあたりのバイト数

実測4点 (Qwen3.8-27B-Q5_K_M / RTX 5090 / CUDA) で、切片が小数3桁まで
一致することを確認した。誤差 0 MiB で再現する。

**差分で測る。** 起動前後の差を取るので、デスクトップや他プロセスの使用量が
混ざらない。``--device CUDA0`` と nvidia-smi の並び順が一致しない問題も、
「一番増えた GPU」を見ることで回避できる。
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import NamedTuple

from . import _hardware

GIB = 1024 ** 3

#: 起動を待つ上限 (秒)。大きいモデルのロードは分単位になりうる
LOAD_TIMEOUT_S = 600
#: VRAM が何秒変化しなければ「ロード完了」と見なすか
SETTLE_S = 6
#: 何秒おきに見るか
POLL_S = 2
#: 増加がこの値未満なら「変化なし」とする (MiB)。計測ノイズを無視する
QUIET_MIB = 8


class Point(NamedTuple):
    """1回の測定。"""

    kv_mode: str      # "f16" / "q8_0"
    ctx: int
    used_mib: int     # 起動前後の差


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


def measure_point(launch_cmd: list[str], kv_mode: str, ctx: int, *,
                  verbose: bool = True) -> Point:
    """サーバを起動して VRAM の増分を測り、落とす。**推論はしない**."""
    before = _gpu_used_mib()
    if not before:
        raise RuntimeError("nvidia-smi did not report anything; cannot measure")

    if verbose:
        print(f"  launching: kv={kv_mode} ctx={ctx:,}", file=sys.stderr)
    proc = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        best, quiet_for, waited = 0, 0.0, 0.0
        while waited < LOAD_TIMEOUT_S:
            time.sleep(POLL_S)
            waited += POLL_S
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {proc.returncode}). "
                    f"Run the command by hand to see why.")
            delta = _biggest_delta(before, _gpu_used_mib())
            if delta - best >= QUIET_MIB:
                best, quiet_for = delta, 0.0
            else:
                best = max(best, delta)
                quiet_for += POLL_S
            if best > 0 and quiet_for >= SETTLE_S:
                break
        else:
            raise RuntimeError(f"gave up after {LOAD_TIMEOUT_S}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    if verbose:
        print(f"    -> {best:,} MiB", file=sys.stderr)
    return Point(kv_mode, ctx, best)


def render_fits(fits: list[Fit]) -> str:
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
    return "\n".join(lines)


def render_toml_fragment(fits: list[Fit]) -> str:
    """``gguf-fit.toml`` に貼れる形。**測った事実として書く**."""
    lines = ["# --- calibrated on this machine (gguf-calibrate) ---",
             "# Measured, not derived. Re-run after changing llama.cpp,",
             "# the backend, or the launch flags."]
    for f in fits:
        key = "kv_f16_bytes" if f.kv_mode == "f16" else "kv_q8_bytes"
        lines.append(f"{key} = {f.bytes_per_token:.0f}"
                     f"   # {f.bytes_per_token / 1024:.1f} KB/token, "
                     f"{f.n_points} points")
    if fits:
        oh = sum(f.intercept_gib for f in fits) / len(fits)
        lines.append(f"# intercept was {oh:.2f} GiB "
                     f"(model file + fixed buffers; gguf-plan subtracts the file)")
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--extra", default=None,
                    help="extra flags passed through verbatim, e.g. "
                         "'--spec-type draft-mtp'. Use the same ones you run with.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg, _ = load_config(args.config)
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

    print(f"measuring {len(ctxs) * len(kvs)} points "
          f"(no inference is run; each server is started and killed)",
          file=sys.stderr)
    fits = []
    for kv_mode in kvs:
        points = []
        for ctx in ctxs:
            cmd = build_launch_cmd(str(binary), args.model, ctx, kv_mode,
                                   device, threads, args.port, extra)
            try:
                points.append(measure_point(cmd, kv_mode, ctx))
            except (RuntimeError, OSError) as e:
                print(f"!! {kv_mode} ctx {ctx}: {e}", file=sys.stderr)
        if len(points) >= 2:
            fits.append(fit_points(points))
        else:
            print(f"!! {kv_mode}: not enough points, skipped", file=sys.stderr)

    if not fits:
        sys.exit("nothing measured")
    print()
    print(render_fits(fits))
    print()
    print("--- paste into gguf-fit.toml ---")
    print(render_toml_fragment(fits), end="")
    return 0
