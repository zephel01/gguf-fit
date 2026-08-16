<div align="center">

# gguf-fit

**GGUF を読むだけで `llama-server` の起動設定を決める。推論は1トークンも行わない。**

[![CI](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://github.com/zephel01/gguf-fit)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[導入](#導入) · [なぜ](#なぜ) · [分かること](#分かること) · [VRAM の見積り](#vram-の見積り) · [使い方](#使い方) · [English](README.md)

</div>

---

`--ctx-size` をいくつにするか。この量子化は自分の GPU に載るのか。`--spec-type draft-mtp`
は効くのか。UD-Q4_K_XL は Q4_K_M と何が違うのか。

こうしたことは普通、GPU を1時間ずつ溶かしながら総当たりで確かめます。

**その多くは、確かめなくても分かります。** GGUF のヘッダとテンソル一覧に答えが書いてあるからです。

`gguf-fit` はそれを読みます。mmap で開くだけなので、27B のファイルでも1秒です。

## 導入

```bash
uv tool install git+https://github.com/zephel01/gguf-fit
```

インストールせず1回だけ試すなら:

```bash
uvx --from git+https://github.com/zephel01/gguf-fit gguf-probe /models/*.gguf
```

<details>
<summary>pip を使う場合</summary>

```bash
pip install git+https://github.com/zephel01/gguf-fit
```

</details>

## まず動かす

```bash
gguf-probe --json --out gguf.json /models/Qwen3.8-27B-GGUF/*.gguf
gguf-plan gguf.json --vram 24 --pick Q5_K_M --lang ja
```

**出力は既定で英語**です。`--lang ja` で日本語になります（設定ファイルに `lang = "ja"`
と書けば毎回付ける必要はありません）。

```console
# ===== Qwen3.8-27B-Q5_K_M / ctx 65,536 / KV f16 =====
# 見積り: モデル 18.47 + KV 4.25 + オーバーヘッド 1.00 = 23.72 GiB / 予算 24.0 GiB
# ⚠️ 余りが 0.28 GiB しかありません …
# native ctx = 262,144  / rope scaling なし
# ハイブリッド注意: 17/65 層のみ KV 保持 = 68 KB/token

# --- llama-server ---
# MTP テンソルを 4 本確認 → --spec-type draft-mtp を付ける
llama-server -m /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q5_K_M.gguf \
  --port 8085 --device CUDA0 \
  -ngl 99 -fa on \
  --ctx-size 65536 --parallel 1 \
  --batch-size 2048 --ubatch-size 512 \
  --spec-type draft-mtp
```

## なぜ

コマンドは2本。**「読む」と「決める」**で切ってあります。

| | |
| :-- | :-- |
| **`gguf-probe`** | GGUF を**読む**。何が書いてあるかを出す |
| **`gguf-plan`** | 読んだ結果と VRAM 予算から、**起動コマンドと config を出す** |

GGUF のどの項目が、どの設定を決めるか。

| 項目 | 決まること |
| :-- | :-- |
| `context_length` + `rope_scaling_type` | `--ctx-size` の**絶対上限**（scaling なしで超えたら壊れる） |
| `kv_cache.bytes_per_token_f16` | VRAM 予算からの `--ctx-size` **逆算**／`-ctk -ctv` の要否 |
| `size_gb` | モデル本体の占有 |
| `mtp_tensor_count` | `--spec-type draft-mtp` が**そもそも効くか** |
| `chat_template_has_think` | thinking か否か → 適用すべきサンプリング |

## 分かること

12量子化の 27B を読んで出てきたものです。

### ファイル名はビット幅について嘘をつく

`UD-Q6_K_XL` は名前に Q6 と入っていますが、重みの **55.1% が Q8_0**。標準の `Q6_K` より
**3.0 GB 重い**。この2つをベンチで比べても、それは**同じビット帯の勝負ではありません**。
UD が勝ったとしても「Dynamic が優秀」ではなく「単に重い」可能性があります。

### 「Dynamic」が変えているのは*層*ではなく*役割*

| | 層ごとに型を変える？ |
| :-- | :-- |
| 標準 K-quant | **変える** — Q4_K_M 3/24 役割、Q5_K_M 3/24、Q3_K_M 2/24 |
| Unsloth Dynamic | **変えない** — UD-Q4/Q5/Q6/Q8 すべて 0/24 |

層ごとに型を上げているのは llama.cpp 自身の `use_more_bits` ヒューリスティックのほうです。
UD が変えているのは**どのテンソル役割を厚くするか**であって、層ごとではありません。

### ハイブリッド注意で KV は `block_count` の 4分の1

Qwen3.8-27B は 65層あるうち、**KV キャッシュを持つのは 17層だけ**。残り 48層は `attn_qkv`
という融合テンソルだけを持つ線形注意で、固定サイズの再帰状態しか持ちません。

> [!IMPORTANT]
> 65層で計算すると **260 KB/token**。実際は **68 KB/token**。**4倍違います。**
> `gguf-probe` は `attn_k`/`attn_v` を実際に持つ層だけを数えます。

## VRAM の見積り

```
使用量 = モデルファイル + KVキャッシュ + オーバーヘッド
```

オーバーヘッド（計算バッファ・CUDAコンテキスト・投機デコード）は実測から較正しました。

| | GiB |
| :-- | --: |
| Qwen3.8-27B-Q5_K_M ファイル | 18.47 |
| KV（ctx 65,536 / f16） | 4.25 |
| 小計 | 22.72 |
| **実測**（llama-server / RTX 5090） | **23.50** |
| **→ オーバーヘッド** | **0.78** |

既定は安全側に **1.0 GiB**。`--overhead` で上書きできます。実測が取れたら issue で教えてください。

> [!WARNING]
> **単位を間違えると結論が反転します。** GGUF の `size_gb` はバイト ÷ 10⁹（**GB**）。
> GPU の「24GB」は **GiB**（÷ 2³⁰）で、10⁹ 換算だと **25.77 GB** あります。
> このツールはすべて GiB に揃えて計算します。

### 24GiB カードに何が載るか（単位はすべて GiB）

| 量子化 | ファイル | +32k | +64k | +128k | 最大ctx (f16) | 同 (q8_0) |
| :-- | --: | --: | --: | --: | --: | --: |
| Q3_K_S | 11.71 | 14.8 | 17.0 | 21.2 | 172,032 | 262,144 |
| Q3_K_M | 12.87 | 16.0 | 18.1 | 22.4 | 155,648 | 262,144 |
| IQ4_XS | 14.63 | 17.8 | 19.9 | 24.1 | 126,976 | 241,664 |
| IQ4_NL | 15.22 | 18.3 | 20.5 | 24.7 | 118,784 | 225,280 |
| Q4_K_M | 15.93 | 19.1 | 21.2 | 25.4 | 106,496 | 204,800 |
| UD-Q4_K_XL | 16.69 | 19.8 | 21.9 | 26.2 | 94,208 | 180,224 |
| **Q5_K_M** | 18.47 | 21.6 | **23.7** | 28.0 | **69,632** | **131,072** |
| UD-Q5_K_XL | 18.83 | 22.0 | 24.1 | 28.3 | 61,440 | 118,784 |
| Q6_K | 21.31 | 24.4 | 26.6 | 30.8 | 24,576 | 49,152 |
| UD-Q6_K_XL | 24.14 | 27.3 | 29.4 | 33.6 | 入らない | 入らない |
| Q8_0 | 27.05 | 30.2 | 32.3 | 36.6 | 入らない | 入らない |
| UD-Q8_K_XL | 29.30 | 32.4 | 34.5 | 38.8 | 入らない | 入らない |

- **Q5_K_M は 24GiB カードに ctx 64k / f16 で載ります**（23.7、余裕 0.3）
- **q8_0 の値打ちは「載せる」ことより「コンテキストを倍にする」こと。** f16 で 64k、
  q8_0 なら **128k**
- KV を持つ層が 65層中17層しかないモデルなので、こういう伸び方をします

**この表を作るのに GPU は一度も回していません。**（較正に使った実測を除く）

<details>
<summary><b>量子化 KV は、計算どおりには縮まない</b></summary>

`q8_0` は 32値ごとに int8 32バイト + fp16 スケール1個 = 34バイト。f16 の 64バイトに対して
**0.531倍**です。ctx 65,536 での実測:

| | 実測 |
| :-- | --: |
| KV f16 | 24,068 MiB |
| KV q8_0 | 22,362 MiB |
| 節約 | **1,706 MiB（1.67 GiB）** |
| 0.531 からの予測 | 1.99 GiB |

差の 0.33 GiB は**逆量子化の作業領域**が一番ありそうです。ただし ctx 1点だけでは
「KV 比が実効 0.61」と「比は 0.531 のままオーバーヘッドが増える」を区別できません。
別の `--ctx-size` でもう1点測れば決着します。

現状は理論値 0.531 を使っているので、**`gguf-plan` は q8_0 の節約をやや過大に見積もります。**

</details>

## 使い方

<details open>
<summary><b>gguf-probe</b> — 読む</summary>

```bash
gguf-probe /models/Qwen3.8-27B-Q5_K_M.gguf          # 1本
gguf-probe /models/Qwen3.8-27B-GGUF/*.gguf          # ディレクトリごと（比較表つき）
gguf-probe --out report.txt /models/*.gguf          # テキスト保存（画面にも出る）
gguf-probe --json --out gguf.json /models/*.gguf    # JSON 保存
gguf-probe --roles /models/one.gguf                 # 層で型が変わる役割を全部
```

JSON には GGUF の**絶対パス**も入るので、`gguf-plan` が `-m` に実物を書けます。

</details>

<details open>
<summary><b>gguf-plan</b> — 決める</summary>

```bash
gguf-plan gguf.json --vram 24 --ctx 65536              # 24GiB に何が載るか
gguf-plan gguf.json --vram 24 --pick Q5_K_M            # 取れる最大の ctx
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072
```

`--kv auto`（既定）なので、f16 で入らなければ**理由を添えて** q8_0 に切り替えます。

`--ctx` を省くと**きりのいい値に切り下げます**。予算いっぱいは勧めません。69,632 は余りが
0.02 GiB しか残らず、65,536 に対して context 6% 増しか得ていないためです。

</details>

<details>
<summary><b>ライブラリとして</b> — GGUF ファイルも <code>gguf</code> パッケージも不要</summary>

```python
from gguf_fit import summarize_tensors, recommended_ctx

s = summarize_tensors(
    [("blk.3.attn_k.weight", "Q5_K"), ("blk.0.attn_qkv.weight", "Q5_K")],
    {"block_count": 65, "head_count_kv": 4, "key_length": 256, "value_length": 256},
)
s["kv_cache"]["bytes_per_token_f16"]

recommended_ctx(rec, vram_gib=24.0, kv_mode="q8_0", overhead=1.0)
```

分類も VRAM 計算も純粋関数です。テストがモデルファイル1本なしで走るのはそのためです。

</details>

## 場面別

<details>
<summary><b>量子化を12本落としてきた。どれを使うか</b></summary>

まず全部読んで、載るものだけに絞ります。**まだ1本も起動していない段階**でできます。
`入らない` と出たものは候補から消えるので、**回す本数がそのまま減ります**。
12本を1本30分で回すと6時間、5本に絞れば2.5時間です。

</details>

<details>
<summary><b>ctx が足りない。何を削るか</b></summary>

| 手段 | 効果 | 代償 |
| :-- | :-- | :-- |
| **KV を q8_0 にする** | 実測 1.67 GiB 減。Q5_K_M で 64k → 128k | 品質影響は未測定 |
| 量子化を1段下げる | Q5_K_M → Q4_K_M で 2.5 GiB | 品質が落ちる |
| ctx を諦める | — | 長い入力が扱えない |

**KV を持つ層が少ないモデルほど、q8_0 の効きは相対的に小さくなります。**
`構造:` の行を見れば、どちらのタイプか分かります。

</details>

### やってはいけないこと

| | |
| :-- | :-- |
| **native ctx を超えて `--ctx-size` を指定する** | rope scaling が無いモデルでは品質が壊れます |
| **`-fa on` なしで `-ctk/-ctv` を量子化する** | 効かないか起動しません。`gguf-plan` は必ずセットで出します |
| **MTP テンソルが無いのに `--spec-type draft-mtp`** | `MTP/nextn テンソル: 0 本` を確認してから |
| **見積りだけで結論を書く** | `nvidia-smi` の実測を取ってください。較正点はまだ少数です |

## 設定

すべての設定はこの順で解決され、**どれが効いたかは `--show-config` で分かります**。

```
CLI フラグ  >  環境変数  >  設定ファイル  >  組み込みの既定値
```

```toml
# ./gguf-fit.toml  （または ~/.config/gguf-fit/config.toml、$GGUF_FIT_CONFIG）
lang     = "ja"
vram     = 24.0
overhead = 1.0
device   = "CUDA0"
port     = 8085
```

```console
$ gguf-plan --show-config
settings in effect (cli > env > config file > default)

  config file: gguf-fit.toml

  lang        ja                       <- config
  vram        24.0                     <- config
  overhead    1.0                      <- default
  port        8085                     <- default
  device      CUDA0                    <- config
  model_path  (unset)                  <- default
```

環境変数は同じ名前に `GGUF_FIT_` を付けたものです（`GGUF_FIT_LANG`、`GGUF_FIT_VRAM` など）。

> [!TIP]
> **設定ファイルは最初に見つかった1つだけを使い、マージしません。** 重ねると
> 「この値はどこから来たのか」が追えなくなるからです。それを言えることがこのツールの
> 売りなので、そこは曖昧にしません。未知のキー・型違い・壊れた TOML は
> **黙って無視せず stderr に出します**（設定が効いていないことに気づけないのが一番困る）。

## 分からないこと（正直に）

- **imatrix の中身。** 量子化タイプが同じでも層ごとに違う imatrix が当たっている可能性は
  あり、GGUF からは読めません。分かるのは**型の割り当てだけ**です
- **実際の品質。** 「Q5_K が 71.7%」は事実ですが、それが何点になるかは回さないと分かりません。
  **探索の前に候補を削る**道具であって、測定の代わりではありません
- **複数GPUへの分割。** `--vram` は1デバイスぶんの予算です。`nvidia-smi` で確かめてください

## 開発

```bash
uv sync            # .venv と開発ツール
uv run pytest -q   # 47件。GGUF ファイルは不要
uv run ruff check .
```

`uv.lock` はコミットしてあり、CI は `uv sync --locked` で回します。**依存を足してロックを
更新し忘れたまま push すると CI が落ちます**（黙ってズレない）。

<details>
<summary>pip を使う場合</summary>

```bash
pip install -e . --group dev   # pip 25.1 以降
pytest -q
ruff check .
```

</details>

### 意図的に固定しているもの

- **ruff のルールセット**（`[tool.ruff.lint] select`）。ruff はマイナー更新で既定ルールが
  増えます。実際 0.15 で通ったコードが 0.16 で10件出ました。選んでおかないと
  「コードを触っていないのに CI が赤くなる」が起きます
- **ruff のバージョン**（`>=0.16.3,<0.17`）。同じ理由です

### 簡略化する前に

このコードには**6つのバグが住んでいました**。どれも「出力が一見それらしい」ものです。
テストが全部固定しているので、リファクタで消さないでください。

<details>
<summary>一覧</summary>

1. **KV層は `block_count` ではない。** `attn_k`/`attn_v` を持つ層だけを数えます。`attn_qkv`
   だけの層は線形注意で KV を持ちません。間違えると4倍の過大評価になります
2. **「層シグネチャが複数種 = Dynamic」は緩すぎる。** Q8_0 まで Dynamic 判定しました。
   **同じ役割**が層によって違う型になっているかで判定します
3. **MTPブロックだけの差は「層ごとの配分」ではない。** IQ4_XS は 64層が IQ4_XS で
   `blk.64` だけ Q4_K。数えると偽陽性になります
4. **`output.weight` は完全一致で。** 部分一致だと `attn_output.weight` を65本拾います
5. **GB と GiB を混ぜた。** 「24GB カードに載るか」の判定が反転します
6. **継続行（`\`）の途中の `#` コメント。** `\` ごとコメントになって起動コマンドが切れます。
   生成コマンドを `bash -n` に通すテストがあります

</details>

## ライセンス

MIT
