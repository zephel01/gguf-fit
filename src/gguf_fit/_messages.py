"""出力メッセージのカタログ.

既定は英語。``--lang ja`` または ``GGUF_FIT_LANG=ja`` で日本語になる。

**CLI のフラグ名・JSON のキー・ライブラリの API 名は翻訳しない。**
それらは機械が読むものなので、言語で変わると使う側が壊れる。翻訳するのは
「人間に向けた説明文」だけ。

新しいメッセージを足すときは en と ja の両方を書くこと。片方だけだと
``test_messages_cover_both_languages`` が落ちる。
"""

from __future__ import annotations

import os

LANGS = ("en", "ja")
DEFAULT_LANG = "en"

#: メッセージ本体。key -> {"en": ..., "ja": ...}
#: 位置引数ではなく **名前つきプレースホルダ** を使う。語順が言語で変わるため。
MESSAGES: dict[str, dict[str, str]] = {
    # ---- probe: render() ----
    "not_a_language_model": {
        "en": "(not a language model; excluded from comparison)",
        "ja": "（言語モデルではありません。比較対象から除外します）",
    },
    "native_ctx": {
        "en": "native ctx = {ctx:,}",
        "ja": "native ctx = {ctx:,}",
    },
    "rope_none": {
        "en": "  / no rope scaling",
        "ja": "  / rope scaling なし",
    },
    "rope_some": {
        "en": "  / rope scaling: {type} x{factor}",
        "ja": "  / rope scaling: {type} x{factor}",
    },
    "weight_quant": {
        "en": "weight quantization: {mix}",
        "ja": "重み量子化: {mix}",
    },
    "dominant_type": {
        "en": "-> dominant type is {type}",
        "ja": "→ 最多の型は {type}",
    },
    "alloc_varies": {
        "en": "type allocation: {n_types} by role / **also varies by layer** "
              "({n_mixed}/{n_roles} roles)",
        "ja": "型の割り当て: 役割ごと {n_types}種 ／ **層ごとにも変える** "
              "({n_mixed}/{n_roles} 役割)",
    },
    "alloc_uniform": {
        "en": "type allocation: {n_types} by role / uniform across layers "
              "(0/{n_roles} roles){extra}",
        "ja": "型の割り当て: 役割ごと {n_types}種 ／ 層ごとには一律 "
              "(0/{n_roles} 役割){extra}",
    },
    "alloc_mtp_only": {
        "en": " ({n} roles differ only in the MTP block; uniform within the body)",
        "ja": "（MTPブロックだけ型が違う役割が {n} あるが、本体の層の中では一律）",
    },
    "hybrid_attention": {
        "en": "structure: hybrid attention. {n_kv} layers full attention (hold KV) "
              "/ {n_lin} layers linear attention (attn_qkv, no KV)",
        "ja": "構造: ハイブリッド注意。{n_kv}層がフルAttention（KV保持）"
              "/ {n_lin}層が線形注意（attn_qkv・KVなし）",
    },
    "mtp_count": {
        "en": "MTP/nextn tensors: {n}",
        "ja": "MTP/nextn テンソル: {n} 本",
    },
    "mtp_yes": {
        "en": "  -> --spec-type draft-mtp works with this file",
        "ja": "  → --spec-type draft-mtp が効くファイル",
    },
    "mtp_no": {
        "en": "  -> no MTP weights (draft-mtp should not work)",
        "ja": "  → MTP重みなし（draft-mtp は効かないはず）",
    },
    "chat_template": {
        "en": "chat_template: {n:,} chars",
        "ja": "chat_template: {n:,} 文字",
    },
    "kv_cache": {
        "en": "KV cache: {n_kv}/{n_all} layers hold it = {kb:.0f} KB/token",
        "ja": "KVキャッシュ: {n_kv}/{n_all} 層が保持 = {kb:.0f} KB/token",
    },
    "kv_sizes": {
        "en": "   f16: 32k={g32}GB  64k={g64}GB   / q8_0: 64k~{q64}GB",
        "ja": "   f16: 32k={g32}GB  64k={g64}GB   / q8_0: 64k≈{q64}GB",
    },
    "roles_header": {
        "en": "--- every role whose type varies by layer ---",
        "ja": "--- 層で型が変わる役割の全リスト ---",
    },
    # ---- probe: main() ----
    "compare_header": {
        "en": "--- comparison across quantizations ---",
        "ja": "--- 量子化間の比較 ---",
    },
    "col_file": {"en": "file", "ja": "file"},
    "col_size": {"en": "size", "ja": "size"},
    "col_dominant": {"en": "dominant", "ja": "最多の型"},
    "col_varies": {"en": "by layer", "ja": "層で可変"},
    "col_kv": {"en": "KV KB/tok", "ja": "KV KB/tok"},
    "not_found": {
        "en": "!! not found: {path}",
        "ja": "!! not found: {path}",
    },
    "need_gguf_package": {
        "en": "Install the gguf package first:  pip install gguf",
        "ja": "pip install gguf を先に実行してください",
    },
    "saved": {
        "en": "[saved] {path}",
        "ja": "[saved] {path}",
    },
    # ---- plan: table ----
    "budget_line": {
        "en": "VRAM budget {vram:.1f} GiB / overhead {overhead:.2f} GiB",
        "ja": "VRAM 予算 {vram:.1f} GiB / オーバーヘッド {overhead:.2f} GiB",
    },
    "col_quant": {"en": "quantization", "ja": "量子化"},
    "col_filesize": {"en": "file", "ja": "ファイル"},
    "col_maxctx_f16": {"en": "max ctx (f16)", "ja": "最大ctx(f16)"},
    "col_maxctx_q8": {"en": "max ctx (q8_0)", "ja": "最大ctx(q8_0)"},
    "col_fits": {
        "en": "ctx {ctx:,}?",
        "ja": "ctx {ctx:,} は?",
    },
    "fits_f16": {
        "en": "f16 OK ({used:.1f})",
        "ja": "f16でOK ({used:.1f})",
    },
    "fits_q8": {
        "en": "OK with q8_0 ({used:.1f})",
        "ja": "q8_0ならOK ({used:.1f})",
    },
    "fits_no": {
        "en": "no fit ({used:.1f})",
        "ja": "入らない ({used:.1f})",
    },
    "no_fit_short": {
        "en": "no fit",
        "ja": "入らない",
    },
    "note_gib": {
        "en": "* All figures are GiB. GGUF's size_gb (bytes / 10^9) is a different unit.",
        "ja": "※ 数値はすべて GiB。GGUF の size_gb (バイト÷10^9) とは違うので注意",
    },
    "note_pick": {
        "en": "* Use --pick <part of a name> to emit a launch command and a config.",
        "ja": "※ --pick <名前の一部> で起動コマンドと config.yaml を出します",
    },
    # ---- plan: emit_config() ----
    "hdr_title": {
        "en": "===== {name} / ctx {ctx:,} / KV {kv} =====",
        "ja": "===== {name} / ctx {ctx:,} / KV {kv} =====",
    },
    "hdr_estimate": {
        "en": "estimate: model {model:.2f} + KV {kv:.2f} + overhead {overhead:.2f} "
              "= {used:.2f} GiB / budget {vram:.1f} GiB",
        "ja": "見積り: モデル {model:.2f} + KV {kv:.2f} + オーバーヘッド {overhead:.2f} "
              "= {used:.2f} GiB / 予算 {vram:.1f} GiB",
    },
    "hdr_over_budget": {
        "en": "X over budget by {over:.2f} GiB. It will not start.",
        "ja": "❌ 予算を {over:.2f} GiB 超えています。起動しません",
    },
    "hdr_thin": {
        "en": "!! only {hr:.2f} GiB of headroom. The overhead has a single calibration "
              "point, so a measurement above it means a failed start",
        "ja": "⚠️ 余りが {hr:.2f} GiB しかありません。オーバーヘッドの較正点は1つ"
              "しかないので、実測がこれを超えると起動に失敗します",
    },
    "hdr_thin_try_q8": {
        "en": "   Use q8_0 for the KV cache (-ctk q8_0 -ctv q8_0), or drop ctx one step",
        "ja": "   KV を q8_0 にする (-ctk q8_0 -ctv q8_0) か、ctx を1段下げてください",
    },
    "hdr_thin_try_ctx": {
        "en": "   Drop ctx one step, or move to a lighter quantization",
        "ja": "   ctx を1段下げるか、1つ軽い量子化にしてください",
    },
    "hdr_headroom": {
        "en": "headroom {hr:.2f} GiB",
        "ja": "余り {hr:.2f} GiB",
    },
    "hdr_native_ctx": {
        "en": "native ctx = {ctx:,}",
        "ja": "native ctx = {ctx:,}",
    },
    "hdr_hybrid": {
        "en": "hybrid attention: only {n_kv}/{n_all} layers hold KV = {kb:.0f} KB/token",
        "ja": "ハイブリッド注意: {n_kv}/{n_all} 層のみ KV 保持 = {kb:.0f} KB/token",
    },
    "sec_server": {
        "en": "--- llama-server ---",
        "ja": "--- llama-server ---",
    },
    "note_fa_required": {
        "en": "quantized -ctk/-ctv requires -fa on (included below)",
        "ja": "-ctk/-ctv の量子化には -fa on が必須 (下に含めてある)",
    },
    "warn_low_free": {
        "en": "!! {name} reports {total:.1f} GiB total but only {free:.1f} GiB free "
              "right now. The plan below uses the total. Free it up, or pass "
              "--vram {free:.0f} to plan against what is actually available.",
        "ja": "⚠️ {name} は総量 {total:.1f} GiB ですが、いま空いているのは "
              "{free:.1f} GiB です。以下の計画は総量で立てています。空けるか、"
              "--vram {free:.0f} で実際に使える分に合わせてください",
    },
    "warn_free_disagrees": {
        "en": "note: llama.cpp reports {runtime:.1f} GiB free on {name}, but the "
              "driver reports {driver:.1f} GiB. On integrated GPUs the runtime "
              "figure often tracks GTT, not the VRAM carve-out. Planning against "
              "the {total:.1f} GiB total -- confirm by actually loading.",
        "ja": "参考: llama.cpp は {name} の空きを {runtime:.1f} GiB と言って"
              "いますが、ドライバは {driver:.1f} GiB と言っています。統合GPU では"
              "ランタイム側が VRAM ではなく GTT を見ていることがあります。"
              "総量 {total:.1f} GiB で計画します -- 実際に載せて確かめてください",
    },
    "help_llama_server": {
        "en": "path(s) to llama-server for --list-devices; repeat or comma-separate "
              "to cover several builds (default: llama-server on PATH)",
        "ja": "llama-server のパス。--list-devices に使います。カンマ区切りで"
              "複数のビルドを指定できます (既定: PATH 上の llama-server)",
    },
    "hint_no_devices": {
        "en": "no GPU detected. llama.cpp shows only the backends its build was "
              "compiled with, so a CUDA build never lists ROCm or Vulkan devices. "
              "Point --llama-server (or llama_servers in the config) at the builds "
              "you actually use.",
        "ja": "GPU が検出できませんでした。llama.cpp はビルドに含まれる"
              "バックエンドしか出さないので、CUDA ビルドからは ROCm や Vulkan の"
              "デバイスは見えません。--llama-server (または設定の llama_servers) で"
              "実際に使うビルドを指してください",
    },
    "warn_vram_mismatch": {
        "en": "!! vram {given:.1f} GiB came from {source}, but this machine reports "
              "{detected:.1f} GiB. Planning for another machine? If not, drop the "
              "setting (or run --write-config) so it matches this one.",
        "ja": "⚠️ vram {given:.1f} GiB は {source} から来ていますが、このマシンの"
              "実測は {detected:.1f} GiB です。別のマシン向けの計画ならそれで"
              "構いません。違うなら設定を消すか --write-config で取り直してください",
    },
    "note_no_device": {
        "en": "no NVIDIA GPU detected, so --device is omitted "
              "(llama.cpp picks Metal / CPU on its own)",
        "ja": "NVIDIA GPU が見つからないので --device は付けません "
              "(llama.cpp が Metal / CPU を自分で選びます)",
    },
    "note_device_ambiguous": {
        "en": "{n} GPUs detected. nvidia-smi order is PCI order and does NOT match "
              "CUDA device numbering -- check with --list-devices",
        "ja": "GPU が {n} 枚あります。nvidia-smi の並びは PCI 順で、CUDA の"
              "デバイス番号とは一致しません -- --list-devices で確認してください",
    },
    "note_mtp_yes": {
        "en": "found {n} MTP tensors -> adding --spec-type draft-mtp",
        "ja": "MTP テンソルを {n} 本確認 → --spec-type draft-mtp を付ける",
    },
    "note_mtp_no": {
        "en": "no MTP tensors -> not adding --spec-type draft-mtp",
        "ja": "MTP テンソルなし → --spec-type draft-mtp は付けない",
    },
    "sec_config": {
        "en": "--- config.yaml (under models:) ---",
        "ja": "--- config.yaml (models: の下) ---",
    },
    "cfg_max_tokens_frac": {
        "en": "3/4 of ctx {ctx:,}; the rest is for the prompt",
        "ja": "ctx {ctx:,} の 3/4。残りはプロンプト用",
    },
    "cfg_max_tokens_cap": {
        "en": "capped at {cap:,} (3/4 of ctx would be {frac:,})",
        "ja": "上限 {cap:,} で頭打ち (ctx の 3/4 なら {frac:,})",
    },
    "cfg_seed": {
        "en": "only set this when runs: 1 (use a random seed when runs > 1)",
        "ja": "runs: 1 のときだけ書く (runs>1 は毎回ランダムにする)",
    },
    "cfg_sampling_from": {
        "en": "sampling defaults for a {tag} chat template",
        "ja": "サンプリングは chat_template の判定 ({tag}) に基づく既定値",
    },
    # ---- plan: main() ----
    "auto_q8": {
        "en": "* ctx {ctx:,} does not fit with f16, so the KV cache is set to q8_0",
        "ja": "※ ctx {ctx:,} は f16 では入らないので KV を q8_0 にしました",
    },
    "auto_ctx": {
        "en": "* ctx not given, using {ctx:,}",
        "ja": "※ ctx 未指定なので {ctx:,} にしました",
    },
    "auto_ctx_rounded": {
        "en": " ({cap:,} would fit, but an odd value leaves no room for the "
              "overhead estimate to be wrong, so it is rounded to a standard size)",
        "ja": " (予算上は {cap:,} まで入りますが、半端な値はオーバーヘッドの"
              "誤差を吸収できないのできりのいい値に丸めます)",
    },
    "no_path_in_json": {
        "en": "* This gguf.json has no absolute path, so -m is a placeholder. "
              "Re-run gguf-probe, or pass --model-path.",
        "ja": "※ この gguf.json には絶対パスが入っていないので -m はプレースホルダです。"
              "gguf-probe を取り直すか --model-path を渡してください",
    },
    "over_budget_stderr": {
        "en": "* The estimate exceeds the {vram:.1f} GiB budget by {over:.2f} GiB",
        "ja": "※ 見積りが予算 {vram:.1f} GiB を {over:.2f} GiB 超えています",
    },
    "err_no_lm": {
        "en": "No language model records found (pass the output of gguf-probe --json).",
        "ja": "言語モデルの記録が見つかりません（gguf-probe --json の出力を渡してください）",
    },
    "err_pick_none": {
        "en": "--pick {pick!r} matched nothing. Available: {names}",
        "ja": "--pick {pick!r} に一致するファイルがありません: {names}",
    },
    "err_pick_many": {
        "en": "--pick {pick!r} matched more than one: {names}",
        "ja": "--pick {pick!r} が複数に一致します: {names}",
    },
    "err_no_fit": {
        "en": "{name} does not fit in {vram} GiB (the weights alone are {model:.2f} GiB)",
        "ja": "{name} は VRAM {vram} GiB に入りません (モデルだけで {model:.2f} GiB)",
    },
    # ---- argparse ----
    "help_lang": {
        "en": "output language (default: en, or $GGUF_FIT_LANG)",
        "ja": "出力言語 (既定: en / $GGUF_FIT_LANG)",
    },
    "desc_probe": {
        "en": "Read GGUF metadata and report what decides the launch settings",
        "ja": "GGUF のメタデータを読んで適正値の材料を出す",
    },
    "desc_plan": {
        "en": "Turn gguf-probe's JSON into llama-server settings and a config",
        "ja": "gguf-probe の JSON から llama-server / config.yaml の設定を出す",
    },
    "help_json": {"en": "emit JSON", "ja": "JSON で出力"},
    "help_roles": {
        "en": "list every role whose type varies by layer",
        "ja": "層で型が変わる役割を全部出す",
    },
    "help_out": {
        "en": "also write the result to this file",
        "ja": "結果をファイルに保存する（画面にも出す）",
    },
    "help_vram": {
        "en": "usable VRAM in GiB (24 for a 24GB card)",
        "ja": "使える VRAM (GiB)。24GBのカードなら 24",
    },
    "help_overhead": {
        "en": "expected compute-buffer overhead in GiB",
        "ja": "計算バッファ等の見込み (GiB)",
    },
    "help_ctx": {
        "en": "context size to check; omit to get the largest that fits",
        "ja": "使いたい ctx。指定すると各量子化が入るか判定する",
    },
    "help_pick": {
        "en": "part of a file name; emits a launch command and a config",
        "ja": "ファイル名の一部。指定すると起動コマンドと config を出す",
    },
    "help_kv": {
        "en": "KV cache type (default auto = q8_0 when f16 does not fit)",
        "ja": "KVキャッシュの型 (既定 auto = f16 で入らなければ q8_0)",
    },
    "help_threads": {
        "en": "--threads for llama-server (default: detected physical cores)",
        "ja": "llama-server の --threads (既定: 検出した物理コア数)",
    },
    "help_model_path": {
        "en": "path to write in the launch command",
        "ja": "起動コマンドに書くパス",
    },
}


def resolve_lang(explicit: str | None = None) -> str:
    """使う言語を決める。``--lang`` > ``$GGUF_FIT_LANG`` > 既定 (en)."""
    for candidate in (explicit, os.environ.get("GGUF_FIT_LANG")):
        if candidate and candidate.lower() in LANGS:
            return candidate.lower()
    return DEFAULT_LANG


def t(key: str, lang: str = DEFAULT_LANG, **kw) -> str:
    """メッセージを引く。未知のキーは黙って落とさず、そうと分かる形で返す."""
    entry = MESSAGES.get(key)
    if entry is None:
        return f"<missing message: {key}>"
    return entry.get(lang, entry[DEFAULT_LANG]).format(**kw)
