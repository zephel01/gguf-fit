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

`--target {llama-server,ollama,lmstudio}` (既定 llama-server) で出力形式を選べる。
見積り部分 (モデル+KV+オーバーヘッド、余りの警告、実測/計算の由来) は3形式共通。
ただし Ollama の Modelfile も LM Studio の公開インターフェースも llama-server の
CLI 全部を書けるわけではない (GPU 層数・KV量子化の型・MTP)。**書けないものは
推測で埋めず、書けない理由をコメントに残す。**詳細は emit_ollama / emit_lmstudio
の docstring。

--- 単位について (ここを間違えると全部ずれる) ---

GGUF の `size_gb` は **バイト ÷ 10^9 (GB)**。一方 GPU の「24GB」は
**GiB (÷ 2^30)**。24GB のカードは 10^9 換算だと 25.77 GB ある。
このスクリプトは**すべて GiB に揃えて**計算する。

--- VRAM の見積り式 ---

    使用量 = モデルファイル + KVキャッシュ + オーバーヘッド

Qwen3.8-27B Q5_K_M / -fa on / --spec-type draft-mtp / RTX 5090 で、ctx を
2点測って直線を出した (`gguf-calibrate`):

    KV f16   69.1 KB/token     GGUF からの計算 68.0 の 1.016 倍
    KV q8_0  43.1 KB/token     理論 (0.531 倍) の 36.1 に対して 1.19 倍
    切片     19.15 GiB         = ファイル 18.47 + オーバーヘッド 0.68
                               (リクエストを1回通したあとの値)

本番のベンチマークを回している間はさらに +32 MiB あった。既定の
オーバーヘッド **1.0 GiB** はそこまで込みで安全側に取ってある。

ただし**この数字は環境で変わる**ので、自分のマシンでは `gguf-calibrate` を
回して `gguf-fit.toml` に `kv_f16_bytes` / `kv_q8_bytes` を書くこと。
書いてあればそちらを使い、無ければ GGUF からの計算値に落ちる。
どちらを使ったかは出力に書く。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from . import _hardware
from ._config import (
    drop_detectable,
    load_config,
    render_show_config,
    render_toml,
    resolve,
    resolve_llama_servers,
    split_repeated,
)
from ._messages import DEFAULT_LANG, pad, t, width

GIB = 1024 ** 3
#: 量子化KVキャッシュのサイズ比 (8bit + スケール)。gguf_probe と同じ値
Q8_FACTOR = 0.53
#: オーバーヘッドの既定 (GiB)。実測 0.68 + 本番で見えた +0.03 を安全側に丸めた
DEFAULT_OVERHEAD_GIB = 1.0
#: --ctx-size はこの単位に切り下げる
CTX_STEP = 4096
#: `--ctx` 未指定のときに選ぶ「きりのいい」ctx。予算に収まる最大のものを採る。
#: 予算いっぱいまで詰めた半端な値 (例 69,632) を勧めてはいけない:
#:   ・オーバーヘッドの見積り誤差を吸収できない (推論そのものが後から
#:     0.11〜0.14 GiB 確保する)
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


class KvRates(NamedTuple):
    """このマシンで実測した1トークンあたりのバイト数.

    ``None`` は**測っていない**という意味で、0 ではない。測っていなければ
    GGUF からの計算値に落ちる。``gguf-calibrate`` が両方を埋める。
    """

    f16: float | None = None
    q8_0: float | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> KvRates:
        def num(key):
            v = cfg.get(key)
            return float(v) if isinstance(v, (int, float)) and v > 0 else None
        return cls(num("kv_f16_bytes"), num("kv_q8_bytes"))

    def get(self, kv_mode: str) -> float | None:
        return self.q8_0 if kv_mode == "q8_0" else self.f16

    @property
    def has_any(self) -> bool:
        return self.f16 is not None or self.q8_0 is not None


def file_gib(rec: dict) -> float:
    return rec["size_gb"] * 1e9 / GIB


def per_token_bytes(rec: dict, kv_mode: str,
                    calibrated: KvRates | None = None) -> float:
    """1トークンあたりの KV バイト数。**測った値があればそちらを使う**.

    GGUF からの計算は下振れする。同じモデル・同じサーバで f16 は 68.0 に対して
    実測 69.1 KB/token (1.6% 増)、q8_0 は理論 36.1 に対して 43.1 (19% 増)。
    q8_0 のずれが大きいのは、比 0.531 が格納形式だけの話で、逆量子化の
    作業領域が入っていないため。
    """
    if calibrated is not None:
        measured = calibrated.get(kv_mode)
        if measured:
            return measured
    f16 = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not f16:
        return 0.0
    return f16 * (Q8_FACTOR if kv_mode == "q8_0" else 1.0)


def kv_gib(rec: dict, ctx: int, kv_mode: str,
           calibrated: KvRates | None = None) -> float:
    return per_token_bytes(rec, kv_mode, calibrated) * ctx / GIB


def max_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float,
            calibrated: KvRates | None = None) -> int:
    """VRAM 予算に収まる最大の ctx。native ctx を超えない。CTX_STEP に切り下げ。"""
    per_tok = per_token_bytes(rec, kv_mode, calibrated)
    if not per_tok:
        return 0
    budget = (vram_gib - overhead - file_gib(rec)) * GIB
    if budget <= 0:
        return 0
    n = int(budget / per_tok)
    n -= n % CTX_STEP
    native = rec.get("context_length") or n
    return max(0, min(n, native))


def recommended_ctx(rec: dict, vram_gib: float, kv_mode: str, overhead: float,
                    calibrated: KvRates | None = None) -> int:
    """`--ctx` 未指定のときに勧める ctx。

    ``max_ctx`` の「予算に入る最大値」をそのまま使うと 69,632 のような半端な
    値になり、余りが 0.02 GiB しか残らない。推論を始めた時点でさらに
    0.11〜0.14 GiB 確保されるので、それで起動に失敗する。
    **きりのいい値に丸めて余裕を残す。**
    """
    cap = max_ctx(rec, vram_gib, kv_mode, overhead, calibrated)
    if not cap:
        return 0
    fits = [c for c in STANDARD_CTX if c <= cap]
    return fits[-1] if fits else cap


def headroom_gib(rec: dict, ctx: int, kv_mode: str, vram_gib: float,
                 overhead: float, calibrated: KvRates | None = None) -> float:
    """この設定で VRAM がどれだけ余るか (GiB)。負なら入らない。"""
    return vram_gib - (file_gib(rec) + kv_gib(rec, ctx, kv_mode, calibrated)
                       + overhead)


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


def cmd_table(recs, vram, overhead, ctx_req, lang=DEFAULT_LANG, calibrated=None):
    rows = []
    for r in sorted(recs, key=lambda d: d["size_gb"]):
        f16 = max_ctx(r, vram, "f16", overhead, calibrated)
        q8 = max_ctx(r, vram, "q8_0", overhead, calibrated)
        fit = ""
        if ctx_req:
            need_f16 = file_gib(r) + kv_gib(r, ctx_req, "f16", calibrated) + overhead
            need_q8 = file_gib(r) + kv_gib(r, ctx_req, "q8_0", calibrated) + overhead
            if need_f16 <= vram:
                fit = t("fits_f16", lang, used=need_f16)
            elif need_q8 <= vram:
                fit = t("fits_q8", lang, used=need_q8)
            else:
                fit = t("fits_no", lang, used=need_q8)
        rows.append((short(r), file_gib(r), f16, q8, fit))

    # 日本語の見出しと「入らない」は1文字で2桁を占める。**文字数で詰めると
    # 必ず崩れる**ので、表示幅で揃える (_messages.pad)
    w = max([*(width(x[0]) for x in rows), width(t("col_quant", lang))]) + 2
    head = (pad(t("col_quant", lang), w) + pad(t("col_filesize", lang), 9, True)
            + pad(t("col_maxctx_f16", lang), 16, True)
            + pad(t("col_maxctx_q8", lang), 17, True))
    if ctx_req:
        head += "   " + t("col_fits", lang, ctx=ctx_req)
    print(head)
    print("-" * width(head))
    none = t("no_fit_short", lang)
    for name, fg, f16, q8, fit in rows:
        f16s = f"{f16:,}" if f16 else none
        q8s = f"{q8:,}" if q8 else none
        line = (pad(name, w) + f"{fg:>8.2f}G"
                + pad(f16s, 16, True) + pad(q8s, 17, True))
        if ctx_req:
            line += f"   {fit}"
        print(line)


#: 較正値が「別のモデルで測ったもの」と判断する比。GGUF からの計算値どうしを
#: 比べるので、同じモデルなら一致するはず。1.15 は丸めと版差のための余裕
CALIBRATION_MISMATCH_RATIO = 1.15


def calibration_mismatch(rec: dict, cfg: dict,
                         lang: str = DEFAULT_LANG) -> str | None:
    """設定の較正値が、**いま見ているモデルのものでない**なら注意書きを返す.

    ``kv_f16_bytes`` は層構造で決まる**モデル固有の値**なのに、設定ファイルに
    書くと以降すべてのモデルに当たる。実際に起きた: Qwen3.8-27B で測った
    69.1 KB/token が、KV が 22.0 KB/token しかない Ornith-1.5-35B にも使われ、
    最大 ctx を3倍近く低く出した。**数字は静かに出るので気づけない。**

    判定は名前ではなく数字で行う。``gguf-calibrate`` が書き残した「測った
    モデルの GGUF からの計算値」と、いま見ているモデルの計算値を比べる。
    記録が無い設定ファイル (古いもの) では何も言わない —— 比べる相手が
    無いのに警告するのは当て推量になる。
    """
    measured = cfg.get("kv_f16_bytes")
    origin = cfg.get("kv_derived_f16_bytes")
    here = (rec.get("kv_cache") or {}).get("bytes_per_token_f16")
    if not measured or not origin or not here:
        return None
    ratio = max(origin, here) / min(origin, here)
    if ratio < CALIBRATION_MISMATCH_RATIO:
        return None
    return "# " + t("warn_calibration_model", lang,
                    file=cfg.get("kv_measured_on") or "?",
                    there=origin / 1024, here=here / 1024, ratio=ratio,
                    kb=measured / 1024)


def budget_warnings(hw, picked_device: str | None, vram: float | None,
                    vram_source: str, device_source: str,
                    lang: str = DEFAULT_LANG) -> list[str]:
    """予算の取り方が怪しいときの注意書き。**gguf-plan と gguf-fetch で共有**.

    片方だけが「一番大きい = 一番速い、ではない」と言い、もう片方が黙って
    同じ device を選ぶ、という状態を作らないためにここに集めてある。
    実機 (5090 / 3090 / 8060S が CUDA・ROCm・Vulkan で同居) では、容量だけで
    自動選択すると **96 GiB の APU を勧めて、生成速度で 5090 に負ける**。
    """
    out: list[str] = []
    disagree = hw.free_figures_disagree(picked_device)
    if disagree is not None:
        dev, driver_free = disagree
        out.append("# " + t("warn_free_disagrees", lang, name=dev.name,
                            runtime=dev.free_gib, driver=driver_free,
                            total=dev.total_gib))

    tight = hw.tight_on_free_memory(picked_device)
    if tight is not None:
        out.append("# " + t("warn_low_free", lang, name=tight.name,
                            total=tight.total_gib, free=tight.free_gib))

    if _hardware.has_mixed_backends(hw) and device_source == "detected":
        big = hw.largest_gpu
        kinds = ", ".join(sorted({g.device_id.rstrip("0123456789")
                                  for g in hw.gpus if g.device_id}))
        out.append("# " + t("warn_mixed_backends", lang, kinds=kinds,
                            name=big.name, total=big.total_gib))

    if vram is not None and vram_source != "detected" and \
            _hardware.vram_disagrees(vram, hw, picked_device):
        out.append("# " + t("warn_vram_mismatch", lang, given=vram,
                            source=vram_source,
                            detected=hw.suggested_vram_gib()))
    return out


def _header_lines(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                  lang: str, calibrated: KvRates | None) -> list[str]:
    """出力3形式 (llama-server / Ollama / LM Studio) で共通の見積り部分.

    見積り・余りの判定・KVの由来 (測った値か計算値か) はサーバの種類に関係ない
    事実なので、ここで1回だけ書く。書き方をサーバごとに変えると、同じ
    モデル・同じ ctx なのに数字が食い違って見える事故になる。
    """
    used = file_gib(rec) + kv_gib(rec, ctx, kv_mode, calibrated) + overhead
    L = []
    L.append("# " + t("hdr_title", lang, name=short(rec), ctx=ctx, kv=kv_mode))
    L.append("# " + t("hdr_estimate", lang, model=file_gib(rec),
                      kv=kv_gib(rec, ctx, kv_mode, calibrated), overhead=overhead,
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
    # **その数字がどこから来たかを言う。**測った値と計算値では 19% 違いうる。
    per_tok = per_token_bytes(rec, kv_mode, calibrated)
    if per_tok:
        measured = calibrated is not None and calibrated.get(kv_mode)
        L.append("# " + t("hdr_kv_measured" if measured else "hdr_kv_derived",
                          lang, kv=kv_mode, kb=per_tok / 1024))
    return L


def emit_config(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                model_path: str, port: int, device: str | None,
                lang: str = DEFAULT_LANG, threads: int | None = None,
                device_ambiguous: bool = False, n_gpus: int = 0,
                calibrated: KvRates | None = None) -> str:
    think = rec.get("chat_template_has_think")
    samp = THINKING_SAMPLING if think else NON_THINKING_SAMPLING
    mtp = bool(rec.get("mtp_tensor_count"))
    max_tokens, mt_key, mt_kw = max_tokens_for(ctx)

    L = _header_lines(rec, ctx, kv_mode, vram, overhead, lang, calibrated)
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


def emit_ollama(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                model_path: str, lang: str = DEFAULT_LANG,
                calibrated: KvRates | None = None) -> str:
    """Modelfile を出す.

    **Ollama の Modelfile には、llama-server の全部が書けるわけではない。**
    公式ドキュメント (docs.ollama.com/modelfile) の PARAMETER 一覧から
    num_gpu / num_thread は消えている。GPU に何層乗せるかは既定では Ollama
    自身が決める。

    ただし ``num_gpu`` は**ドキュメントから消えただけで、いまも動く**
    (github.com/ollama/ollama#13986 で ollama 側が認めている: 「e54a3c7 で
    ドキュメントから消したが機能は残した」)。gguf-fit が計画するのは常に
    全層オフロード (llama-server 側の `-ngl 99` と同じ) なので、**目安として**
    大きめの値を書いておく。非公式なので保証はしない、とコメントに残す。
    num_thread のほうは同種の裏付けが取れていないので書かない。

    KV キャッシュの型 (f16/q8_0) も Modelfile には書けない。決めるのは
    ``OLLAMA_KV_CACHE_TYPE`` という**サーバ全体の環境変数**で、モデルごとには
    切り替えられない。この関数が q8_0 を選んだときは、そのことを書くだけで、
    無いパラメータを捏造はしない。

    MTP (--spec-type draft-mtp) に対応する Modelfile のキーは無い。
    ``draft_num_predict`` という投機的デコード用のパラメータはあるが、
    llama.cpp の MTP ドラフトと同じ仕組みで動くかどうかは確認していないので、
    ここでは出さない。
    """
    think = rec.get("chat_template_has_think")
    samp = THINKING_SAMPLING if think else NON_THINKING_SAMPLING
    mtp = bool(rec.get("mtp_tensor_count"))

    L = _header_lines(rec, ctx, kv_mode, vram, overhead, lang, calibrated)
    L.append("")
    L.append("# " + t("sec_ollama", lang))
    L.append("# " + t("note_ollama_num_gpu_undocumented", lang))
    if kv_mode == "q8_0":
        L.append("# " + t("note_ollama_kv_is_global", lang))
    if mtp:
        L.append("# " + t("note_ollama_mtp_unverified", lang, n=rec["mtp_tensor_count"]))
    L.append("")
    L.append(f"FROM {model_path}")
    L.append(f"PARAMETER num_ctx {ctx}")
    L.append("PARAMETER num_gpu 99   # " + t("cfg_num_gpu_approx", lang))
    for k, v in samp.items():
        L.append(f"PARAMETER {k} {v}")
    L.append("")
    name = short(rec).lower().replace(".", "-").replace("_", "-")
    L.append("# " + t("ollama_create_hint", lang, name=name))
    L.append(f"#   ollama create {name} -f ./Modelfile")
    if kv_mode == "q8_0":
        L.append("#   OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve")
    return "\n".join(L)


def emit_lmstudio(rec: dict, ctx: int, kv_mode: str, vram: float, overhead: float,
                  model_path: str, lang: str = DEFAULT_LANG,
                  calibrated: KvRates | None = None) -> str:
    """LM Studio 向けの読み込み設定 (JSON) を出す.

    ``POST /api/v1/models/load`` (lmstudio.ai/docs/developer/rest/load) と
    ``lms load`` CLI が公開しているキーだけを書く: context_length /
    eval_batch_size / flash_attention / offload_kv_cache_to_gpu。

    **GPU に何層乗せるか、KV キャッシュを q8_0 にできるかは、この2つの
    公開インターフェースのどちらにも見当たらない。**GUI 側には項目がある
    かもしれないが確認していないので、無いものとして扱う。kv_mode が
    q8_0 のときは、その前提が LM Studio では満たせないかもしれないと書く。

    層数そのものは JSON のキーとしては書かない (でっち上げになる) が、
    ``--gpu max`` が結局何層ぶんの話をしているかは**目安として**コメントに
    残す。GGUF のテンソル一覧から数えた総層数がその値。
    """
    import json as _json  # noqa: PLC0415 - この関数でしか使わない

    L = _header_lines(rec, ctx, kv_mode, vram, overhead, lang, calibrated)
    L.append("")
    L.append("# " + t("sec_lmstudio", lang))
    L.append("# " + t("note_lmstudio_no_gpu_layers", lang))
    n_layers = (rec.get("kv_cache") or {}).get("total_layers")
    if n_layers:
        L.append("# " + t("note_lmstudio_layer_count", lang, n=n_layers))
    if kv_mode == "q8_0":
        L.append("# " + t("note_lmstudio_kv_unsupported", lang))
    body = {
        "model": model_path,
        "context_length": ctx,
        "eval_batch_size": 512,
        "flash_attention": True,
        "offload_kv_cache_to_gpu": True,
    }
    L.append(_json.dumps(body, indent=2, ensure_ascii=False))
    L.append("")
    name = short(rec)
    L.append("# " + t("lmstudio_cli_hint", lang))
    L.append(f"#   lms load \"{name}\" --gpu max --context-length {ctx}")
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
    ap.add_argument("--target", choices=["llama-server", "ollama", "lmstudio"],
                    default="llama-server", help=t("help_target", DEFAULT_LANG))
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
    ap.add_argument("--refresh", action="store_true",
                    help="ignore vram/threads/device in the config file and take "
                         "them from the hardware again")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    if args.refresh:
        # 設定ファイルは実測より強い。一度書いた値は居座るので、
        # ハードウェアを見て取り直す手段を用意しておく。
        cfg = drop_detectable(cfg)
    # --llama-server は複数回渡せる。カンマ区切りも受ける。
    # 解決そのものは gguf-fetch と共有する (_config)。片方だけが ROCm ビルドを
    # 見る状態になると、同じマシンで2つのコマンドが違う予算を出す
    r_llama = resolve_llama_servers(split_repeated(args.llama_server), cfg)
    hw = _hardware.detect(r_llama.value)
    r_lang = resolve("lang", args.lang, cfg, DEFAULT_LANG)
    # **device を先に決める。**予算はそのデバイスの容量から取る。
    # 別々に決めると「96 GiB を前提に 31.8 GiB のカードで起動」が起こる。
    r_device = resolve("device", args.device, cfg,
                       detected=hw.suggested_device())
    picked_device = str(r_device.value) if r_device.value else None
    r_vram = resolve("vram", args.vram, cfg,
                     detected=hw.suggested_vram_gib(picked_device))
    r_overhead = resolve("overhead", args.overhead, cfg, DEFAULT_OVERHEAD_GIB)
    r_port = resolve("port", args.port, cfg, 8085)
    r_threads = resolve("threads", args.threads, cfg,
                        detected=hw.suggested_threads())
    r_model_path = resolve("model_path", args.model_path, cfg)
    # gguf-calibrate が書いた実測値も **--show-config に出す**。
    # 「いま何が効いているか」を答えるのがこのツールの役目なのに、KV の単価
    # だけ見えないのでは、19% 違う数字が黙って効いていることになる。
    r_kv_f16 = resolve("kv_f16_bytes", None, cfg)
    r_kv_q8 = resolve("kv_q8_bytes", None, cfg)

    settings = {"lang": r_lang, "vram": r_vram, "overhead": r_overhead,
                "port": r_port, "device": r_device, "threads": r_threads,
                "llama_servers": r_llama, "model_path": r_model_path,
                "kv_f16_bytes": r_kv_f16, "kv_q8_bytes": r_kv_q8}

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
    # gguf-calibrate が書いた実測値。無ければ None のままで、GGUF からの
    # 計算値に落ちる。**測っていないものを測ったことにはしない。**
    kv = KvRates.from_config(cfg)

    # 設定ファイルを別のマシンに持っていったときに気づけるようにする。
    # 実測が取れているのに指定値と 食い違うなら、そう言う。
    # 総量は足りていても、いま空いていなければ載らない。
    # 統合GPUで実際に見た: 総量 96 GiB / 空き 16.3 GiB。
    for line in budget_warnings(hw, picked_device, vram, r_vram.source,
                                r_device.source, lang):
        print(line, file=sys.stderr)

    recs = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    if isinstance(recs, dict):
        recs = [recs]
    lm = [r for r in recs if r.get("is_language_model") and r.get("kv_cache")]
    if not lm:
        sys.exit(t("err_no_lm", lang))

    # 較正値が別のモデルで測ったものなら、表を出す前に言う
    mismatch = calibration_mismatch(lm[0], cfg, lang)
    if mismatch:
        print(mismatch, file=sys.stderr)

    if not args.pick:
        print(t("budget_line", lang, vram=vram, overhead=overhead) + "\n")
        cmd_table(lm, vram, overhead, args.ctx, lang, kv)
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
        if ctx and file_gib(rec) + kv_gib(rec, ctx, "f16", kv) + overhead > vram:
            kv_mode = "q8_0"
            print("# " + t("auto_q8", lang, ctx=ctx), file=sys.stderr)
    if ctx is None:
        cap = max_ctx(rec, vram, kv_mode, overhead, kv)
        ctx = recommended_ctx(rec, vram, kv_mode, overhead, kv)
        if not ctx:
            sys.exit(t("err_no_fit", lang, name=short(rec), vram=vram,
                       model=file_gib(rec)))
        note = "# " + t("auto_ctx", lang, ctx=ctx)
        if cap > ctx:
            note += t("auto_ctx_rounded", lang, cap=cap)
        print(note, file=sys.stderr)

    hr = headroom_gib(rec, ctx, kv_mode, vram, overhead, kv)
    if hr < 0:
        print("# " + t("over_budget_stderr", lang, vram=vram, over=-hr), file=sys.stderr)

    mp = model_path_of(rec, r_model_path.value)
    if not rec.get("path") and not r_model_path.value:
        print("# " + t("no_path_in_json", lang), file=sys.stderr)
    device = str(r_device.value) if r_device.value else None

    if args.target == "ollama":
        print(emit_ollama(rec, ctx, kv_mode, vram, overhead, mp, lang,
                          calibrated=kv))
    elif args.target == "lmstudio":
        print(emit_lmstudio(rec, ctx, kv_mode, vram, overhead, mp, lang,
                            calibrated=kv))
    else:
        print(emit_config(rec, ctx, kv_mode, vram, overhead,
                          mp, int(r_port.value), device, lang,
                          threads=r_threads.value,
                          device_ambiguous=hw.device_index_is_ambiguous(),
                          n_gpus=len(hw.gpus), calibrated=kv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
