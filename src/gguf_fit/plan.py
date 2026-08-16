"""gguf_probe.py の出力 (gguf.json) から、実際の起動設定に落とし込む.

    gguf-probe --json --out gguf.json /path/to/*.gguf   # 先にこれ
    gguf-plan gguf.json --vram 24                       # 何が載るか一覧
    gguf-plan gguf.json --vram 24 --pick Q5_K_M         # 起動コマンドを出す
    gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072

決めてくれるもの:

  * `--ctx-size`      … VRAM 予算から逆算した上限 (native ctx も超えない)
  * `-ctk / -ctv`     … f16 で入らないとき q8_0 を提案する
  * `--spec-type`     … MTP テンソルがあるときだけ draft-mtp を付ける
  * サンプリング一式  … chat_template が thinking なら Qwen3.8 公式推奨

--- 単位について (ここを間違えると全部ずれる) ---

GGUF の `size_gb` は **バイト ÷ 10^9 (GB)**。一方 GPU の「24GB」は
**GiB (÷ 2^30)**。24GB のカードは 10^9 換算だと 25.77 GB ある。
このスクリプトは**すべて GiB に揃えて**計算する。

--- VRAM の見積り式 ---

    使用量 = モデルファイル + KVキャッシュ + オーバーヘッド

オーバーヘッド (計算バッファ・CUDAコンテキスト・投機デコードのバッファ) は
実測1点から較正した:

    Qwen3.8-27B Q5_K_M / ctx 65536 / KV f16 / -fa on / --spec-type draft-mtp
      ファイル 18.47 GiB + KV 4.25 GiB = 22.72 GiB
      実測 (llama-server) 23.5 GiB
      → オーバーヘッド 0.78 GiB

較正点が1つしかないので、既定は安全側に **1.0 GiB** を置く。
実測が増えたら `--overhead` で上書きすること。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _hardware
from ._config import load_config, render_show_config, render_toml, resolve
from ._messages import DEFAULT_LANG, t

GIB = 1024 ** 3
#: 量子化KVキャッシュのサイズ比 (8bit + スケール)。gguf_probe と同じ値
Q8_FACTOR = 0.53
#: 実測1点から較正したオーバーヘッド (GiB)。安全側に丸めてある
DEFAULT_OVERHEAD_GIB = 1.0
#: --ctx-size はこの単位に切り下げる
CTX_STEP = 4096
#: `--ctx` 未指定のときに選ぶ「きりのいい」ctx。予算に収まる最大のものを採る。
#: 予算いっぱいまで詰めた半端な値 (例 69,632) を勧めてはいけない:
#:   ・オーバーヘッドの見積り誤差 (較正点は1つしかない) を吸収できない
#:   ・65,536 に対して context 6% 増しか得ていない
STANDARD_CTX = (4096, 8192, 16384, 32768, 49152, 65536, 98304,
                131072, 196608, 262144, 393216, 524288, 1048576)
#: 余りがこれを下回るときは警告する (GiB)
MIN_HEADROOM_GIB = 0.5
#: max_tokens の上限。thinking モデルでもこれ以上は要らない (実測での運用値)
MAX_TOKENS_CAP = 49152

THINKING_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
}
NON_THINKING_SAMPLING = {
    "temperature": 0.7, "top_p": 0.80, "top_k": 20,
}


def file_gib(rec: dict) -> float:
    return rec["size_gb"] * 1e9 / GIB


def kv_gib(rec: dict, ctx: int, kv_mode: str) -> float:
    per_tok = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not per_tok:
        return 0.0
    factor = Q8_FACTOR if kv_mode == "q8_0" else 1.0
    return per_tok * ctx * factor / GIB


def max_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float) -> int:
    """VRAM 予算に収まる最大の ctx。native ctx を超えない。CTX_STEP に切り下げ。"""
    per_tok = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not per_tok:
        return 0
    factor = Q8_FACTOR if kv_mode == "q8_0" else 1.0
    budget = (vram_gib - overhead - file_gib(rec)) * GIB
    if budget <= 0:
        return 0
    n = int(budget / (per_tok * factor))
    n -= n % CTX_STEP
    native = rec.get("context_length") or n
    return max(0, min(n, native))


def recommended_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float) -> int:
    """`--ctx` 未指定のときに勧める ctx。

    ``max_ctx`` の「予算に入る最大値」をそのまま使うと 69,632 のような半端な
    値になり、余りが 0.02 GiB しか残らない。オーバーヘッドの較正点は1つしか
    ないので、その誤差で起動に失敗する。**きりのいい値に丸めて余裕を残す。**
    """
    cap = max_ctx(rec, vram_gib, kv_mode, overhead)
    if not cap:
        return 0
    fits = [c for c in STANDARD_CTX if c <= cap]
    return fits[-1] if fits else cap


def headroom_gib(rec: dict, ctx: int, kv_mode: str, vram_gib: float,
                 overhead: float) -> float:
    """この設定で VRAM がどれだけ余るか (GiB)。負なら入らない。"""
    return vram_gib - (file_gib(rec) + kv_gib(rec, ctx, kv_mode) + overhead)


def model_path_of(rec: dict, override: str | None) -> str:
    """起動コマンドの -m に書くパス.

    gguf_probe が絶対パスを記録していればそれを使う。古い JSON にはこのキーが
    無いので、その場合だけプレースホルダに落とす。
    """
    if override:
        return override
    return rec.get("path") or f"/path/to/{rec['file']}"


def max_tokens_for(ctx: int) -> tuple[int, str, dict]:
    """``max_tokens`` と、**その値になった理由**をメッセージキーとして返す.

    2つの制約の小さいほうを採る。

      * ctx の 3/4 … 残りをプロンプトに使う。max_tokens >= n_ctx にすると
        実効上限が n_ctx - プロンプト長になり、max_tokens が効かなくなる
      * MAX_TOKENS_CAP … thinking モデルでもこれ以上は要らない実測値

    **どちらが効いたかを言い分けること。**ctx 131,072 で 49,152 を出しながら
    「ctx の 3/4」と書くと嘘になる (3/4 は 98,304)。

    翻訳できるよう、文字列そのものではなく (キー, 差し込む値) を返す。
    """
    three_quarters = (ctx * 3 // 4) // 1024 * 1024
    if three_quarters <= MAX_TOKENS_CAP:
        return three_quarters, "cfg_max_tokens_frac", {"ctx": ctx}
    return MAX_TOKENS_CAP, "cfg_max_tokens_cap", {"cap": MAX_TOKENS_CAP,
                                                  "frac": three_quarters}


def short(rec: dict) -> str:
    return rec["file"].replace(".gguf", "")


def cmd_table(recs, vram, overhead, ctx_req, lang=DEFAULT_LANG):
    rows = []
    for r in sorted(recs, key=lambda d: d["size_gb"]):
        f16 = max_ctx(r, vram, "f16", overhead)
        q8 = max_ctx(r, vram, "q8_0", overhead)
        fit = ""
        if ctx_req:
            need_f16 = file_gib(r) + kv_gib(r, ctx_req, "f16") + overhead
            need_q8 = file_gib(r) + kv_gib(r, ctx_req, "q8_0") + overhead
            if need_f16 <= vram:
                fit = t("fits_f16", lang, used=need_f16)
            elif need_q8 <= vram:
                fit = t("fits_q8", lang, used=need_q8)
            else:
                fit = t("fits_no", lang, used=need_q8)
        rows.append((short(r), file_gib(r), f16, q8, fit))

    w = max(len(x[0]) for x in rows) + 2
    head = (f"{t('col_quant', lang):<{w}}{t('col_filesize', lang):>9}"
            f"{t('col_maxctx_f16', lang):>16}{t('col_maxctx_q8', lang):>17}")
    if ctx_req:
        head += "   " + t("col_fits", lang, ctx=ctx_req)
    print(head)
    print("-" * len(head))
    none = t("no_fit_short", lang)
    for name, fg, f16, q8, fit in rows:
        f16s = f"{f16:,}" if f16 else none
        q8s = f"{q8:,}" if q8 else none
        line = f"{name:<{w}}{fg:>8.2f}G{f16s:>16}{q8s:>17}"
        if ctx_req:
            line += f"   {fit}"
        print(line)


def emit_config(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                model_path: str, port: int, device: str | None,
                lang: str = DEFAULT_LANG, threads: int | None = None,
                device_ambiguous: bool = False, n_gpus: int = 0) -> str:
    think = rec.get("chat_template_has_think")
    samp = THINKING_SAMPLING if think else NON_THINKING_SAMPLING
    mtp = bool(rec.get("mtp_tensor_count"))
    used = file_gib(rec) + kv_gib(rec, ctx, kv_mode) + overhead
    max_tokens, mt_key, mt_kw = max_tokens_for(ctx)

    L = []
    L.append("# " + t("hdr_title", lang, name=short(rec), ctx=ctx, kv=kv_mode))
    L.append("# " + t("hdr_estimate", lang, model=file_gib(rec),
                      kv=kv_gib(rec, ctx, kv_mode), overhead=overhead,
                      used=used, vram=vram))
    hr = vram - used
    if hr < 0:
        L.append("# " + t("hdr_over_budget", lang, over=-hr))
    elif hr < MIN_HEADROOM_GIB:
        L.append("# " + t("hdr_thin", lang, hr=hr))
        # 既に q8_0 なら「q8_0 にしろ」とは言わない
        L.append("#" + (t("hdr_thin_try_q8", lang) if kv_mode == "f16"
                        else t("hdr_thin_try_ctx", lang)))
    else:
        L.append("# " + t("hdr_headroom", lang, hr=hr))
    st = rec.get("rope_scaling_type")
    L.append("# " + t("hdr_native_ctx", lang, ctx=rec.get("context_length", 0))
             + (t("rope_some", lang, type=st, factor=rec.get("rope_scaling_factor"))
                if st else t("rope_none", lang)))
    if rec.get("is_hybrid_attention"):
        L.append("# " + t("hdr_hybrid", lang, n_kv=len(rec["kv_layers"]),
                          n_all=rec["kv_cache"]["total_layers"],
                          kb=rec["kv_cache"]["bytes_per_token_f16"] / 1024))
    L.append("")
    L.append("# " + t("sec_server", lang))
    # 注記は**コマンドの外**に出す。継続行 (\) の途中に # を書くと
    # そこから行末までがコメントになり、\ ごと消えてコマンドが壊れる。
    if kv_mode == "q8_0":
        L.append("# " + t("note_fa_required", lang))
    if not device:
        L.append("# " + t("note_no_device", lang))
    elif device_ambiguous:
        L.append("# " + t("note_device_ambiguous", lang, n=n_gpus))
    L.append("# " + (t("note_mtp_yes", lang, n=rec["mtp_tensor_count"]) if mtp
                     else t("note_mtp_no", lang)))
    args = [
        f"llama-server -m {model_path}",
        f"--port {port} --device {device}" if device else f"--port {port}",
        "-ngl 99 -fa on",
        f"--ctx-size {ctx} --parallel 1",
        "--batch-size 2048 --ubatch-size 512",
    ]
    if threads:
        # llama.cpp の --threads は論理コアではなく**物理コア**に合わせる
        args.append(f"--threads {threads}")
    if kv_mode == "q8_0":
        args.append("-ctk q8_0 -ctv q8_0")
    if mtp:
        args.append("--spec-type draft-mtp")
    L.append(" \\\n  ".join(args))
    L.append("")
    L.append("# " + t("sec_config", lang))
    L.append(f"  {short(rec).lower().replace('.', '-')}:")
    L.append("    type: openai")
    L.append(f'    base_url: "http://localhost:{port}/v1"')
    L.append('    model: "auto"')
    L.append('    api_key: "sk-local"')
    for k, v in samp.items():
        L.append(f"    {k}: {v}")
    L.append(f"    max_tokens: {max_tokens}   # " + t(mt_key, lang, **mt_kw))
    L.append("    seed: 42          # " + t("cfg_seed", lang))
    tag = "thinking" if think else "non-thinking"
    L.append("    # " + t("cfg_sampling_from", lang, tag=tag))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=t("desc_plan", DEFAULT_LANG))
    ap.add_argument("json_path", nargs="?",
                    help="the JSON produced by gguf-probe --json --out")
    ap.add_argument("--vram", type=float, default=None,
                    help=t("help_vram", DEFAULT_LANG))
    ap.add_argument("--overhead", type=float, default=None,
                    help=t("help_overhead", DEFAULT_LANG)
                    + f" (default {DEFAULT_OVERHEAD_GIB})")
    ap.add_argument("--ctx", type=int, default=None, help=t("help_ctx", DEFAULT_LANG))
    ap.add_argument("--pick", default=None, help=t("help_pick", DEFAULT_LANG))
    ap.add_argument("--kv", choices=["f16", "q8_0", "auto"], default="auto",
                    help=t("help_kv", DEFAULT_LANG))
    ap.add_argument("--model-path", default=None, dest="model_path",
                    help=t("help_model_path", DEFAULT_LANG))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help=t("help_threads", DEFAULT_LANG))
    ap.add_argument("--llama-server", action="append", default=None,
                    dest="llama_server", help=t("help_llama_server", DEFAULT_LANG))
    ap.add_argument("--lang", default=None, choices=["en", "ja"],
                    help=t("help_lang", DEFAULT_LANG))
    ap.add_argument("--config", default=None,
                    help="path to a gguf-fit.toml (overrides the search)")
    ap.add_argument("--show-config", action="store_true", dest="show_config",
                    help="print the settings in effect and where each came from")
    ap.add_argument("--write-config", nargs="?", const="gguf-fit.toml",
                    default=None, dest="write_config", metavar="PATH",
                    help="write the detected settings to a gguf-fit.toml "
                         "(default: ./gguf-fit.toml)")
    ap.add_argument("--force", action="store_true",
                    help="allow --write-config to overwrite an existing file")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    # --llama-server は複数回渡せる。カンマ区切りも受ける。
    cli_bins = None
    if args.llama_server:
        cli_bins = [x.strip() for a in args.llama_server
                    for x in a.split(",") if x.strip()]
    r_llama = resolve("llama_servers", cli_bins, cfg)
    if r_llama.value is None:
        # 単数形のキー / 環境変数にも対応する
        single = resolve("llama_server", None, cfg, "llama-server")
        r_llama = single._replace(value=[single.value])
    hw = _hardware.detect(r_llama.value)
    r_lang = resolve("lang", args.lang, cfg, DEFAULT_LANG)
    r_vram = resolve("vram", args.vram, cfg, detected=hw.suggested_vram_gib())
    r_overhead = resolve("overhead", args.overhead, cfg, DEFAULT_OVERHEAD_GIB)
    r_port = resolve("port", args.port, cfg, 8085)
    r_device = resolve("device", args.device, cfg,
                       detected=hw.suggested_device())
    r_threads = resolve("threads", args.threads, cfg,
                        detected=hw.suggested_threads())
    r_model_path = resolve("model_path", args.model_path, cfg)

    settings = {"lang": r_lang, "vram": r_vram, "overhead": r_overhead,
                "port": r_port, "device": r_device, "threads": r_threads,
                "llama_servers": r_llama, "model_path": r_model_path}

    if args.show_config:
        print(render_show_config(settings, cfg_path))
        print()
        print(_hardware.render(hw))
        if not hw.gpus:
            print()
            print(t("hint_no_devices", r_lang.value))
        return 0

    if args.write_config:
        dest = Path(args.write_config)
        if dest.exists() and not args.force:
            sys.exit(f"{dest} already exists. Pass --force to overwrite it.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(render_toml(settings, _hardware.render(hw)),
                        encoding="utf-8")
        print(f"[written] {dest}", file=sys.stderr)
        print(dest.read_text(encoding="utf-8"), end="")
        return 0

    lang = r_lang.value
    if not args.json_path:
        ap.error("json_path is required")
    if r_vram.value is None:
        print("# " + t("hint_no_devices", lang), file=sys.stderr)
        ap.error("could not detect any GPU, so --vram is required "
                 "(or set vram in the config file)")
    vram, overhead = float(r_vram.value), float(r_overhead.value)

    # 設定ファイルを別のマシンに持っていったときに気づけるようにする。
    # 実測が取れているのに指定値と 食い違うなら、そう言う。
    # 総量は足りていても、いま空いていなければ載らない。
    # 統合GPUで実際に見た: 総量 96 GiB / 空き 16.3 GiB。
    disagree = hw.free_figures_disagree()
    if disagree is not None:
        dev, driver_free = disagree
        print("# " + t("warn_free_disagrees", lang, name=dev.name,
                       runtime=dev.free_gib, driver=driver_free,
                       total=dev.total_gib), file=sys.stderr)

    tight = hw.tight_on_free_memory()
    if tight is not None:
        print("# " + t("warn_low_free", lang, name=tight.name,
                       total=tight.total_gib, free=tight.free_gib),
              file=sys.stderr)

    if r_vram.source != "detected" and _hardware.vram_disagrees(vram, hw):
        print("# " + t("warn_vram_mismatch", lang, given=vram,
                       source=r_vram.source, detected=hw.suggested_vram_gib()),
              file=sys.stderr)

    recs = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    if isinstance(recs, dict):
        recs = [recs]
    lm = [r for r in recs if r.get("is_language_model") and r.get("kv_cache")]
    if not lm:
        sys.exit(t("err_no_lm", lang))

    if not args.pick:
        print(t("budget_line", lang, vram=vram, overhead=overhead) + "\n")
        cmd_table(lm, vram, overhead, args.ctx, lang)
        print("\n" + t("note_gib", lang))
        print(t("note_pick", lang))
        return 0

    hits = [r for r in lm if args.pick.lower() in r["file"].lower()]
    if not hits:
        sys.exit(t("err_pick_none", lang, pick=args.pick,
                   names=", ".join(short(r) for r in lm)))
    if len(hits) > 1:
        sys.exit(t("err_pick_many", lang, pick=args.pick,
                   names=", ".join(short(r) for r in hits)))
    rec = hits[0]

    kv_mode = args.kv
    ctx = args.ctx
    if kv_mode == "auto":
        kv_mode = "f16"
        if ctx and file_gib(rec) + kv_gib(rec, ctx, "f16") + overhead > vram:
            kv_mode = "q8_0"
            print("# " + t("auto_q8", lang, ctx=ctx), file=sys.stderr)
    if ctx is None:
        cap = max_ctx(rec, vram, kv_mode, overhead)
        ctx = recommended_ctx(rec, vram, kv_mode, overhead)
        if not ctx:
            sys.exit(t("err_no_fit", lang, name=short(rec), vram=vram,
                       model=file_gib(rec)))
        note = "# " + t("auto_ctx", lang, ctx=ctx)
        if cap > ctx:
            note += t("auto_ctx_rounded", lang, cap=cap)
        print(note, file=sys.stderr)

    hr = headroom_gib(rec, ctx, kv_mode, vram, overhead)
    if hr < 0:
        print("# " + t("over_budget_stderr", lang, vram=vram, over=-hr), file=sys.stderr)

    mp = model_path_of(rec, r_model_path.value)
    if not rec.get("path") and not r_model_path.value:
        print("# " + t("no_path_in_json", lang), file=sys.stderr)
    device = str(r_device.value) if r_device.value else None
    print(emit_config(rec, ctx, kv_mode, vram, overhead,
                      mp, int(r_port.value), device, lang,
                      threads=r_threads.value,
                      device_ambiguous=hw.device_index_is_ambiguous(),
                      n_gpus=len(hw.gpus)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
