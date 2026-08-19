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
import unicodedata

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
        "en": "!! only {hr:.2f} GiB of headroom. Inference itself allocated another "
              "0.11-0.14 GiB after load when measured, so this can still fail to start",
        "ja": "⚠️ 余りが {hr:.2f} GiB しかありません。実測では推論を通すと"
              "さらに 0.11〜0.14 GiB 増えたので、これでも起動に失敗しえます",
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
    "hdr_kv_measured": {
        "en": "KV {kv} = {kb:.1f} KB/token, measured here (gguf-calibrate), "
              "not derived from the GGUF",
        "ja": "KV {kv} = {kb:.1f} KB/token。GGUF からの計算ではなく"
              "このマシンでの実測 (gguf-calibrate)",
    },
    "hdr_kv_derived": {
        "en": "KV {kv} = {kb:.1f} KB/token, derived from the GGUF. Real usage ran "
              "2-19% higher when measured; gguf-calibrate settles it",
        "ja": "KV {kv} = {kb:.1f} KB/token。GGUF からの計算値。実測では "
              "2〜19% 多かったので、gguf-calibrate で確かめられます",
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
    "warn_mixed_backends": {
        "en": "note: several backends are present ({kinds}). The budget defaults to "
              "the largest device, {name} ({total:.1f} GiB) -- largest is not "
              "fastest. Pin --device / --vram if you meant another one.",
        "ja": "参考: 種類の違うバックエンドが同居しています ({kinds})。予算は"
              "一番大きい {name} ({total:.1f} GiB) を既定にしました -- "
              "**一番大きい = 一番速い、ではありません。**別のデバイスで動かす"
              "なら --device / --vram を明示してください",
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
    # ---- plan: emit_ollama() ----
    "help_target": {
        "en": "output format: a llama-server command (default), an Ollama "
              "Modelfile, or an LM Studio load config",
        "ja": "出力形式。既定は llama-server の起動コマンド。ほかに Ollama の "
              "Modelfile、LM Studio の読み込み設定も選べる",
    },
    "sec_ollama": {
        "en": "--- Ollama Modelfile ---",
        "ja": "--- Ollama Modelfile ---",
    },
    "note_ollama_num_gpu_undocumented": {
        "en": "Ollama decides GPU offload itself by default. num_gpu was "
              "dropped from docs.ollama.com/modelfile but Ollama's own "
              "maintainers confirm it still works (github.com/ollama/ollama"
              "#13986). gguf-fit only ever plans full offload, so num_gpu is "
              "set high as an approximate hint -- treat it as best-effort, "
              "not guaranteed. num_thread has no such confirmation, so it is "
              "left out",
        "ja": "GPU に何層乗せるかは既定では Ollama 自身が決める。num_gpu は "
              "docs.ollama.com/modelfile から消えたが、Ollama 側が「機能は "
              "残っている」と認めている (github.com/ollama/ollama#13986)。"
              "gguf-fit は全層オフロードしか計画しないので、num_gpu は目安として "
              "大きめに設定してある -- 保証はしない。num_thread は同様の裏付けが "
              "取れていないので出さない",
    },
    "cfg_num_gpu_approx": {
        "en": "approximate; undocumented but still works as of ollama/ollama#13986",
        "ja": "目安。非公式だが ollama/ollama#13986 の時点では動作する",
    },
    "note_ollama_kv_is_global": {
        "en": "this plan assumes q8_0 KV, but Ollama has no per-model KV cache "
              "type in the Modelfile -- it is the server-wide OLLAMA_KV_CACHE_TYPE "
              "env var, which affects every loaded model, not just this one",
        "ja": "この計画は KV を q8_0 前提にしているが、Ollama には Modelfile 単位の "
              "KV 型指定が無い。決めるのはサーバ全体の環境変数 "
              "OLLAMA_KV_CACHE_TYPE で、いま動いている全モデルに効く",
    },
    "note_ollama_mtp_unverified": {
        "en": "found {n} MTP tensors, but whether Ollama's draft_num_predict "
              "uses them the way llama.cpp's --spec-type draft-mtp does has not "
              "been checked, so nothing is set for it here",
        "ja": "MTP テンソルを {n} 本確認したが、Ollama の draft_num_predict が "
              "llama.cpp の --spec-type draft-mtp と同じ仕組みで使うかは "
              "確認していないので、ここでは何も設定しない",
    },
    "ollama_create_hint": {
        "en": "save as ./Modelfile, then:",
        "ja": "./Modelfile として保存してから:",
    },
    # ---- plan: emit_lmstudio() ----
    "sec_lmstudio": {
        "en": "--- LM Studio load config ---",
        "ja": "--- LM Studio 読み込み設定 ---",
    },
    "note_lmstudio_no_gpu_layers": {
        "en": "checked against the documented load API (lmstudio.ai/docs/"
              "developer/rest/load) and the lms CLI: neither exposes a GPU "
              "layer count, only lms load --gpu as a 0-1 fraction, so it is "
              "left out of the JSON below",
        "ja": "公開されているロード API (lmstudio.ai/docs/developer/rest/load) と "
              "lms CLI を確認したが、どちらも GPU に何層乗せるかは指定できず、 "
              "lms load --gpu は 0〜1 の割合を渡すだけなので、下の JSON には "
              "書いていない",
    },
    "note_lmstudio_layer_count": {
        "en": "{n} layers total (counted from the tensor list) -- --gpu max / "
              "offload_kv_cache_to_gpu below already assume all of them go to "
              "the GPU",
        "ja": "全 {n} 層 (テンソル一覧から数えた数) -- 下の --gpu max / "
              "offload_kv_cache_to_gpu はこの全部を GPU に乗せる前提",
    },
    "note_lmstudio_kv_unsupported": {
        "en": "this plan assumes q8_0 KV to fit the budget, but neither the "
              "load API nor the lms CLI documents a KV cache quantization "
              "option -- if LM Studio has no such setting in the GUI either, "
              "this budget cannot actually be reached there",
        "ja": "この計画は予算に収めるため KV を q8_0 前提にしているが、ロード API "
              "にも lms CLI にも KV キャッシュの量子化を指定する項目は見当たらない。 "
              "GUI にも同等の設定が無ければ、この予算は LM Studio では達成できない",
    },
    "lmstudio_cli_hint": {
        "en": "roughly equivalent with the lms CLI (context length only; "
              "no KV-type or GPU-layer control there either):",
        "ja": "lms CLI でのおおよその対応 (context length のみ。こちらにも "
              "KV 型・GPU 層数を指定する項目は無い):",
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

    # ---- fetch: CLI のヘルプ ----
    "desc_fetch": {
        "en": "download GGUF files from Hugging Face, after deciding which ones fit",
        "ja": "Hugging Face から GGUF を落とす。落とす前に、どれが載るかを決める",
    },
    "help_repo": {
        "en": "Hugging Face repo id, for example ornith-ai/Ornith-1.5-35B-A3B-GGUF",
        "ja": "Hugging Face のリポジトリ id。例 ornith-ai/Ornith-1.5-35B-A3B-GGUF",
    },
    "help_fit": {
        "en": "download the largest quantizations that fit the VRAM budget",
        "ja": "VRAM 予算に収まるもののうち、大きいほうから落とす",
    },
    "help_fetch_pick": {
        "en": "download this quantization only (for example Q5_K_M)",
        "ja": "この量子化だけを落とす (例 Q5_K_M)",
    },
    "help_all": {
        "en": "download every GGUF in the repo",
        "ja": "リポジトリの GGUF を全部落とす",
    },
    "help_top": {
        "en": "how many quantizations --fit downloads",
        "ja": "--fit が落とす本数",
    },
    "help_min_ctx": {
        "en": "a quantization counts as fitting only if it reaches this ctx",
        "ja": "この ctx に届いて初めて「載る」と数える",
    },
    "help_dir": {
        "en": "where to put the files (a subdirectory named after the repo is created)",
        "ja": "置き場所 (リポジトリ名のサブディレクトリを作る)",
    },
    "help_revision": {
        "en": "branch, tag or commit (default main)",
        "ja": "ブランチ・タグ・コミット (既定 main)",
    },
    "help_mmproj": {
        "en": "vision projector: auto = the smallest one, all, or none",
        "ja": "ビジョン投影: auto = 最小の1本 / all = 全部 / none = 付けない",
    },
    "help_probe_mode": {
        "en": "how many GGUF headers to read over HTTP: one (default), all, or none",
        "ja": "GGUF ヘッダを何本 HTTP で読むか: one (既定) / all / none",
    },
    "help_yes": {
        "en": "do not ask before downloading",
        "ja": "確認せずに落とす",
    },
    "help_dry_run": {
        "en": "print the hf download command and stop",
        "ja": "hf download のコマンドを出すだけで落とさない",
    },
    "help_fetch_json": {
        "en": "print the candidates and the verdict as JSON",
        "ja": "候補と判定を JSON で出す",
    },
    "help_hf_bin": {
        "en": "the hf executable to use (default: hf, then huggingface-cli)",
        "ja": "使う hf コマンド (既定: hf、無ければ huggingface-cli)",
    },

    # ---- fetch: 表と判定 ----
    "fetch_col_verdict": {
        "en": "verdict",
        "ja": "判定",
    },
    "fetch_mark_yes": {
        "en": "fits",
        "ja": "載る",
    },
    "fetch_mark_no": {
        "en": "no",
        "ja": "入らない",
    },
    "fetch_mark_maybe": {
        "en": "maybe (size only)",
        "ja": "たぶん (サイズだけ)",
    },
    "fetch_mark_take": {
        "en": "-> DOWNLOAD",
        "ja": "→ 落とす",
    },
    "fetch_unknown": {
        "en": "?",
        "ja": "?",
    },
    "fetch_header_line": {
        "en": "{repo}  ({rev})",
        "ja": "{repo}  ({rev})",
    },
    "fetch_budget": {
        "en": "budget {vram:.1f} GiB / overhead {overhead:.1f} GiB / KV {kv} "
              "/ counts as fitting from ctx {min_ctx:,}",
        "ja": "予算 {vram:.1f} GiB / オーバーヘッド {overhead:.1f} GiB / KV {kv} "
              "/ ctx {min_ctx:,} 以上で「載る」",
    },
    "fetch_kv_source": {
        "en": "# the KV figures come from the header of {files} ({mb:.1f} MB "
              "transferred). Quantizations of the same model share the layer "
              "structure, so the same KV/token applies to every row.",
        "ja": "# KV の数字は {files} のヘッダから読みました ({mb:.1f} MB 転送)。"
              "同じモデルの量子化違いは層構造が同じなので、KV/token は全行に"
              "そのまま当てはまります。",
    },
    "fetch_kv_source_all": {
        "en": "# read the header of every candidate: {files} ({mb:.1f} MB transferred)",
        "ja": "# 候補すべてのヘッダを読みました: {files} ({mb:.1f} MB 転送)",
    },
    "fetch_size_only": {
        "en": "# no header was read (--probe none), so this is file size against the "
              "budget only. The KV cache is not counted.",
        "ja": "# ヘッダを読んでいません (--probe none)。ファイルサイズと予算を"
              "比べただけで、**KVキャッシュのぶんは入っていません**。",
    },
    "fetch_mmproj_found": {
        "en": "# {n} vision projector(s) found; taking {name} ({gib:.2f} GiB). "
              "--mmproj all / none changes this.",
        "ja": "# ビジョン投影が {n} 本あります。{name} ({gib:.2f} GiB) を付けます。"
              "--mmproj all / none で変えられます。",
    },
    "fetch_next": {
        "en": "nothing downloaded. Pick a mode:\n"
              "  gguf-fetch {repo} --fit          # the ones that fit\n"
              "  gguf-fetch {repo} --pick Q5_K_M  # just this one\n"
              "  gguf-fetch {repo} --all          # everything",
        "ja": "まだ何も落としていません。モードを選んでください:\n"
              "  gguf-fetch {repo} --fit          # 載るものを上から\n"
              "  gguf-fetch {repo} --pick Q5_K_M  # 指定した1本だけ\n"
              "  gguf-fetch {repo} --all          # 全部",
    },
    "fetch_nothing_fits": {
        "en": "nothing fits in {vram:.1f} GiB. Lower --min-ctx, pass --kv q8_0, "
              "or name one with --pick and run it on the CPU.",
        "ja": "{vram:.1f} GiB に載るものがありません。--min-ctx を下げるか、"
              "--kv q8_0 を指定するか、--pick で名指しして CPU で回してください。",
    },
    "fetch_plan": {
        "en": "downloading {n} file(s), {gib:.2f} GiB total:",
        "ja": "落とすもの: {n} ファイル / 合計 {gib:.2f} GiB",
    },
    "fetch_disk_ok": {
        "en": "# disk free: {free:.1f} GiB",
        "ja": "# ディスクの空き: {free:.1f} GiB",
    },
    "fetch_disk_short": {
        "en": "!! disk free is {free:.1f} GiB but {need:.2f} GiB is needed. Stopping "
              "here rather than filling the disk.",
        "ja": "!! ディスクの空きが {free:.1f} GiB しかありません ({need:.2f} GiB 必要)。"
              "埋め切る前に止めます。",
    },
    "fetch_confirm": {
        "en": "download? [y/N] ",
        "ja": "落としますか? [y/N] ",
    },
    "fetch_cancelled": {
        "en": "cancelled.",
        "ja": "やめました。",
    },
    "fetch_done": {
        "en": "done -> {dir}\nThese figures are still estimates. Measure them:",
        "ja": "完了 -> {dir}\nここまでの数字はまだ見積りです。測ってください:",
    },
    "fetch_no_hf": {
        "en": "hf was not found. Install it with `pip install -U huggingface_hub`, "
              "or run the command above yourself.",
        "ja": "hf が見つかりません。`pip install -U huggingface_hub` で入れるか、"
              "上のコマンドを自分で実行してください。",
    },
    "fetch_repo_failed": {
        "en": "could not read {repo}: {err}",
        "ja": "{repo} を読めませんでした: {err}",
    },
    "fetch_no_gguf": {
        "en": "{repo} has no .gguf files",
        "ja": "{repo} に .gguf がありません",
    },
    "fetch_pick_none": {
        "en": "--pick {pick} matched nothing. Available: {names}",
        "ja": "--pick {pick} に当たるものがありません。あるのは: {names}",
    },
    "fetch_header_failed": {
        "en": "could not read the header of {file}: {err}",
        "ja": "{file} のヘッダを読めませんでした: {err}",
    },
}


def width(text: str) -> int:
    """端末で何桁を占めるか。**日本語は1文字で2桁**.

    ``str.ljust`` も f-string の ``{:<10}`` も**文字数**で詰める。「入らない」は
    4文字だが端末では8桁あるので、日本語の列がある表は必ず崩れる。
    Unicode の East Asian Width が W (Wide) か F (Fullwidth) のものを2桁と数える。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)


def pad(text: str, columns: int, right: bool = False) -> str:
    """表示幅で桁を揃える。``right=True`` で右寄せ."""
    fill = " " * max(0, columns - width(text))
    return fill + text if right else text + fill


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
