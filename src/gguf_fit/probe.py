"""GGUF の「適正値」を、推論を1トークンも行わずに読み取る.

    gguf-probe /mnt/data/models/Qwen3.8-27B-GGUF/*.gguf
    gguf-probe --out gguf_report.txt  /path/*.gguf     # テキスト保存
    gguf-probe --json --out gguf.json /path/*.gguf     # JSON保存
    gguf-probe --roles /path/one.gguf                  # 役割ごとの量子化型

読み取れること:

  1. native context_length と RoPE スケーリング → `--ctx-size` の上限が決まる
  2. **KVキャッシュの実サイズ** → ctx を増やしたときの VRAM 増分が計算できる
  3. 量子化の中身（役割ごと・層ごとの型） → 「UD-Q4_K_XL」の実体が何なのかが分かる
  4. MTP / nextn テンソルの有無 → `--spec-type draft-mtp` が効くかが確定する
  5. チャットテンプレート → thinking タグを含むか

mmap で開くだけなので 27B の GGUF でも1秒。

--- 実ファイルで検証して直した点 -------------------------------------------

  * **KV層数を block_count と決め打ちしていた（4倍の過大評価）。**
    ハイブリッド注意のモデルでは一部の層しかKVキャッシュを持たない。
    実測 Qwen3.8-27B（arch=qwen35）は 65層のうち
      - 17層 … attn_q/attn_k/attn_v/attn_output を持つ = フルAttention = KV保持
      - 48層 … attn_qkv（融合）だけ = 線形注意 = **KVキャッシュを持たない**
    17層で計算した 68 KB/token が実測（ctx 32k→64k の VRAM 増分 ≈66 KB/token）
    と一致する。65層で計算すると 260 KB/token になり4倍ズレる。
    → **attn_k / attn_v を持つ層だけを数える。**attn_qkv は数えない。

  * **「層シグネチャの種類 > 1 なら Dynamic」が緩すぎた。**標準 K-quant でも
    層ごとにテンソル構成が違えばシグネチャは増え、Q8_0 まで「Dynamic 的」と
    出てしまった。
    → **同じ役割（ffn_down.weight 等）が層によって違う型になっているか**で判定。
    実測ではこの判定で、標準 K-quant が「層ごとに変える」側、
    Unsloth Dynamic が「層ごとには一律（役割ごとには変える）」側に分かれた。

  * `output.weight` の部分一致で `attn_output.weight` を拾って出力が汚れていた。
    → 完全一致に変更。

  注意: 量子化タイプが同じでも imatrix（キャリブレーション）が層ごとに違う
  可能性はあり、それは GGUF からは読めない。ここで分かるのは
  **型の割り当てだけ**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ._config import load_config, render_show_config, resolve
from ._messages import DEFAULT_LANG, t


def _reader_class():
    """``gguf`` の import は **実際にファイルを開くときまで遅らせる**。

    テンソル分類・KV計算は純粋なロジックで、gguf パッケージが無くても
    テストできる。モジュール先頭で import して sys.exit すると、
    そのテストごと落ちてしまう。
    """
    try:
        from gguf import GGUFReader  # noqa: PLC0415 - 遅延 import が本関数の目的
    except ImportError:  # pragma: no cover
        sys.exit(t("need_gguf_package", DEFAULT_LANG))
    return GGUFReader


_MTP_PAT = re.compile(r"(nextn|mtp|multi_token|draft)", re.IGNORECASE)
_BLK_PAT = re.compile(r"^blk\.(\d+)\.(.+)$")
#: **KVキャッシュを持つ**層の判定。K/V を別々に持つ層だけが対象。
#: ハイブリッド注意のモデルでは、線形注意（Gated DeltaNet 等）の層が
#: ``attn_qkv`` という融合テンソルを持つが、**これはKVキャッシュを持たない**
#: （固定サイズの再帰状態しか持たない）。実測 Qwen3.8-27B は
#: 65層中 attn_k/attn_v を持つのが17層、attn_qkv だけなのが48層で、
#: 17層で計算した 68 KB/token が実測（約66 KB/token）と一致する。
_KV_PAT = re.compile(r"\.(attn_k|attn_v)\.")
#: 線形注意（KVキャッシュを持たない）層の目印
_LINEAR_ATTN_PAT = re.compile(r"\.attn_qkv\.")
#: 正規化・スケール類。量子化の議論では除外する
_NORM_TYPES = {"F32"}
_NOTABLE = {"token_embd.weight", "output.weight", "output_norm.weight"}


def _val(field):
    if field is None:
        return None
    try:
        parts, data = field.parts, field.data
        if not parts or data is None or len(data) == 0:
            return None
        if len(data) == 1:
            v = parts[data[0]]
            if hasattr(v, "tolist"):
                v = v.tolist()
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", "replace")
            if isinstance(v, list) and v and all(isinstance(i, int) for i in v):
                try:
                    return bytes(v).decode("utf-8")
                except Exception:  # noqa: BLE001 - 未知のGGUFメタデータは何でも来る
                    return v
            return v
        out = []
        for i in data:
            v = parts[i]
            out.append(v.tolist() if hasattr(v, "tolist") else v)
        return out
    except Exception:  # noqa: BLE001 - 1フィールドの解釈失敗で全体を落とさない
        return None


def _kv(reader, key):
    for f in reader.fields.values():
        if f.name == key:
            return _val(f)
    return None


def _find_kv(reader, suffix):
    for f in reader.fields.values():
        if f.name.endswith(suffix):
            return _val(f)
    return None


def summarize_tensors(tensors, meta: dict | None = None) -> dict:
    """テンソル一覧から、量子化の配分・ハイブリッド注意・KVサイズを出す.

    **この関数は純粋**。``tensors`` は ``(テンソル名, 量子化型名)`` の並びで、
    ``("blk.3.attn_k.weight", "Q5_K")`` のようなタプルを渡す。GGUF ファイルも
    ``gguf`` パッケージも要らないので、そのままテストできる。

    ``meta`` は ``block_count`` / ``head_count_kv`` / ``key_length`` /
    ``value_length`` を含む辞書（GGUF のメタデータから読んだもの）。
    KVサイズの計算にだけ使う。無ければ ``kv_cache`` は返らない。
    """
    meta = meta or {}
    out: dict = {}

    by_type = Counter()
    role_types: dict[str, Counter] = defaultdict(Counter)      # 役割 -> 型の分布
    role_layers: dict[str, dict[int, str]] = defaultdict(dict)  # 役割 -> {層: 型}
    kv_layers_set: set[int] = set()
    linear_layers: set[int] = set()
    mtp_layers: set[int] = set()
    all_layers: set[int] = set()
    notable = []
    mtp = []

    for name, tname in tensors:
        by_type[tname] += 1
        if name in _NOTABLE:
            notable.append({"name": name, "type": tname})
        if _MTP_PAT.search(name):
            mtp.append(name)
            mm = _BLK_PAT.match(name)
            if mm:
                mtp_layers.add(int(mm.group(1)))
        m = _BLK_PAT.match(name)
        if m:
            layer, role = int(m.group(1)), m.group(2)
            all_layers.add(layer)
            role_types[role][tname] += 1
            role_layers[role][layer] = tname
            if _KV_PAT.search("." + role):
                kv_layers_set.add(layer)
            elif _LINEAR_ATTN_PAT.search("." + role):
                linear_layers.add(layer)

    out["quant_mix"] = dict(by_type.most_common())
    weight_mix = {k: v for k, v in by_type.items() if k not in _NORM_TYPES}
    total_w = sum(weight_mix.values()) or 1
    out["weight_mix"] = {
        k: {"n": v, "share": round(v / total_w * 100, 1)}
        for k, v in sorted(weight_mix.items(), key=lambda x: -x[1])
    }
    out["dominant_weight_type"] = next(iter(out["weight_mix"]), None)
    out["notable_tensors"] = sorted(notable, key=lambda d: d["name"])
    out["mtp_tensor_count"] = len(mtp)
    out["mtp_tensors"] = sorted(mtp)[:8]

    # ---- 「層ごとにビット配分を変えている」の判定
    # 同じ役割のテンソルが層によって違う型になっているか。
    # ただし **MTPブロック(blk.N.nextn.* を持つ層) だけが違う**ケースは
    # 「層ごとの配分」ではない（実測: IQ4_XS は 64層 IQ4_XS + blk.64 だけ Q4_K）。
    # MTP層を除いてもなお混ざっているものだけを per_layer とする。
    mixed_roles: dict[str, dict] = {}
    mtp_only_roles: dict[str, dict] = {}
    for role, types in role_types.items():
        real = {k: v for k, v in types.items() if k not in _NORM_TYPES}
        if len(real) <= 1:
            continue
        without_mtp = Counter(
            t for layer, t in role_layers[role].items()
            if layer not in mtp_layers and t not in _NORM_TYPES
        )
        entry = dict(sorted(real.items(), key=lambda x: -x[1]))
        if len(without_mtp) > 1:
            mixed_roles[role] = entry
        else:
            mtp_only_roles[role] = entry
    out["mixed_roles"] = mixed_roles
    out["mtp_only_roles"] = mtp_only_roles
    out["n_mixed_roles"] = len(mixed_roles)
    out["n_mtp_only_roles"] = len(mtp_only_roles)
    out["n_roles"] = len(role_types)
    out["per_layer_varying"] = len(mixed_roles) > 0
    out["mtp_block_layers"] = sorted(mtp_layers)

    # ---- KVキャッシュ（実際に attn テンソルを持つ層だけで計算する）
    n_block = meta.get("block_count")
    n_kv = meta.get("head_count_kv")
    if isinstance(n_kv, list):
        n_kv = n_kv[0] if n_kv else None
    k_len = meta.get("key_length")
    v_len = meta.get("value_length") or k_len
    kv_layers = len(kv_layers_set) if kv_layers_set else n_block
    out["kv_layers"] = sorted(kv_layers_set)
    out["linear_attn_layers"] = sorted(linear_layers - kv_layers_set)
    out["is_hybrid_attention"] = bool(kv_layers_set and (linear_layers - kv_layers_set))
    if kv_layers and n_kv and k_len:
        per_tok = kv_layers * n_kv * (k_len + v_len) * 2  # f16
        out["kv_cache"] = {
            "kv_bearing_layers": kv_layers,
            "total_layers": n_block or len(all_layers),
            "counted_from": "attn_k/attn_v tensors" if kv_layers_set else "block_count (fallback)",
            "bytes_per_token_f16": per_tok,
            "gb_32k_f16": round(per_tok * 32768 / 1e9, 2),
            "gb_64k_f16": round(per_tok * 65536 / 1e9, 2),
            "gb_64k_q8_0": round(per_tok * 65536 / 1e9 * 0.53, 2),  # 8bit+スケール
        }
    return out


def probe(path: Path) -> dict:
    r = _reader_class()(str(path))
    out: dict = {
        "file": path.name,
        # 絶対パスも残す。gguf_plan が -m にそのまま書けるようにするため
        # (ファイル名だけだと起動コマンドがプレースホルダになる)
        "path": str(path.resolve()),
        "size_gb": round(path.stat().st_size / 1e9, 2),
        "architecture": _kv(r, "general.architecture"),
        "file_type": _kv(r, "general.file_type"),
        "n_tensors": len(r.tensors),
    }
    out["is_language_model"] = out["architecture"] not in ("clip", "mmproj", None)

    for suffix, key in (
        (".context_length", "context_length"),
        (".block_count", "block_count"),
        (".attention.head_count", "head_count"),
        (".attention.head_count_kv", "head_count_kv"),
        (".attention.key_length", "key_length"),
        (".attention.value_length", "value_length"),
        (".embedding_length", "embedding_length"),
        (".rope.freq_base", "rope_freq_base"),
        (".rope.scaling.type", "rope_scaling_type"),
        (".rope.scaling.factor", "rope_scaling_factor"),
        (".attention.sliding_window", "sliding_window"),
    ):
        v = _find_kv(r, suffix)
        if v is not None:
            out[key] = v

    # ---- テンソル走査・量子化配分・KVサイズ（純粋部分に委譲）
    out.update(summarize_tensors(
        ((t.name, t.tensor_type.name) for t in r.tensors), out))

    # ---- チャットテンプレート
    tmpl = _kv(r, "tokenizer.chat_template")
    if isinstance(tmpl, str):
        out["chat_template_len"] = len(tmpl)
        out["chat_template_has_think"] = "think" in tmpl.lower()

    return out


def render(p: dict, roles: bool = False, lang: str = DEFAULT_LANG) -> str:
    L = [f"== {p['file']}  ({p['size_gb']} GB / {p['n_tensors']} tensors)",
         f"   arch={p.get('architecture')}  file_type={p.get('file_type')}"]
    if not p["is_language_model"]:
        L.append("   " + t("not_a_language_model", lang))
        return "\n".join(L)

    ctx = p.get("context_length")
    if ctx:
        line = "   " + t("native_ctx", lang, ctx=ctx)
        st = p.get("rope_scaling_type")
        line += (t("rope_some", lang, type=st, factor=p.get("rope_scaling_factor"))
                 if st else t("rope_none", lang))
        L.append(line)

    wm = p.get("weight_mix", {})
    mix = ", ".join(f"{k} {v['share']}%({v['n']})" for k, v in list(wm.items())[:5])
    L.append("   " + t("weight_quant", lang, mix=mix))
    dom = p.get("dominant_weight_type")
    if dom:
        L.append("   " + t("dominant_type", lang, type=dom))

    n_wt = len(p.get("weight_mix", {}))
    if p["per_layer_varying"]:
        L.append("   " + t("alloc_varies", lang, n_types=n_wt,
                           n_mixed=p["n_mixed_roles"], n_roles=p["n_roles"]))
        for role, types in list(p["mixed_roles"].items())[:3]:
            L.append("      " + role + ": "
                     + ", ".join(f"{k}x{v}" for k, v in types.items()))
    else:
        extra = ""
        if p.get("n_mtp_only_roles"):
            extra = t("alloc_mtp_only", lang, n=p["n_mtp_only_roles"])
        L.append("   " + t("alloc_uniform", lang, n_types=n_wt,
                           n_roles=p["n_roles"], extra=extra))

    if p.get("is_hybrid_attention"):
        L.append("   " + t("hybrid_attention", lang, n_kv=len(p["kv_layers"]),
                           n_lin=len(p["linear_attn_layers"])))

    if p.get("notable_tensors"):
        L.append("   " + ", ".join(f"{x['name']}={x['type']}" for x in p["notable_tensors"]))

    n = p.get("mtp_tensor_count", 0)
    L.append("   " + t("mtp_count", lang, n=n)
             + (t("mtp_yes", lang) if n else t("mtp_no", lang)))

    if p.get("chat_template_len"):
        L.append("   " + t("chat_template", lang, n=p["chat_template_len"]))

    kv = p.get("kv_cache")
    if kv:
        L.append("   " + t("kv_cache", lang, n_kv=kv["kv_bearing_layers"],
                           n_all=kv["total_layers"],
                           kb=kv["bytes_per_token_f16"] / 1024))
        L.append("   " + t("kv_sizes", lang, g32=kv["gb_32k_f16"],
                           g64=kv["gb_64k_f16"], q64=kv["gb_64k_q8_0"]))
    if roles and p.get("mixed_roles"):
        L.append("   " + t("roles_header", lang))
        for role, types in p["mixed_roles"].items():
            L.append("      " + role + ": " + ", ".join(f"{k}x{v}" for k, v in types.items()))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=t("desc_probe", DEFAULT_LANG))
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--json", action="store_true", help=t("help_json", DEFAULT_LANG))
    ap.add_argument("--roles", action="store_true", help=t("help_roles", DEFAULT_LANG))
    ap.add_argument("--out", default=None, help=t("help_out", DEFAULT_LANG))
    ap.add_argument("--lang", default=None, choices=["en", "ja"],
                    help=t("help_lang", DEFAULT_LANG))
    ap.add_argument("--config", default=None,
                    help="path to a gguf-fit.toml (overrides the search)")
    ap.add_argument("--show-config", action="store_true", dest="show_config",
                    help="print the settings in effect and where each came from")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)
    lang = resolve("lang", args.lang, cfg, DEFAULT_LANG)

    if args.show_config:
        print(render_show_config({"lang": lang}, cfg_path))
        return 0
    if not args.paths:
        ap.error("at least one path is required")

    results = []
    for pat in args.paths:
        targets = sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [Path(pat)]
        for path in targets:
            if not path.exists():
                print(t("not_found", lang.value, path=path), file=sys.stderr)
                continue
            try:
                results.append(probe(path))
            except Exception as e:  # noqa: BLE001 - 1本壊れても残りは読む
                print(f"!! {path.name}: {e}", file=sys.stderr)

    if args.json:
        text = json.dumps(results, ensure_ascii=False, indent=2, default=str)
    else:
        chunks = [render(p, roles=args.roles, lang=lang.value) for p in results]
        lm = [p for p in results if p["is_language_model"]]
        if len(lm) > 1:
            rows = [t("compare_header", lang.value),
                    (f"  {t('col_file', lang.value):<34}"
                     f"{t('col_size', lang.value):>8}  "
                     f"{t('col_dominant', lang.value):<9}"
                     f"{t('col_varies', lang.value):>9}  "
                     f"{t('col_kv', lang.value):>10}")]
            for p in sorted(lm, key=lambda d: d["size_gb"]):
                kv = p.get("kv_cache") or {}
                rows.append(
                    f"  {p['file'][:33]:<34}{p['size_gb']:>7.2f}G  "
                    f"{p.get('dominant_weight_type')!s:<12}"
                    f"{p['n_mixed_roles']:>3}/{p['n_roles']:<4}  "
                    f"{kv.get('bytes_per_token_f16', 0) / 1024:>9.0f}")
            chunks.append("\n".join(rows))
        text = "\n\n".join(chunks)

    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("\n" + t("saved", lang.value, path=args.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
