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
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --fit   # 載るものだけ落とす
gguf-probe --json --out gguf.json /models/Qwen3.8-27B-GGUF/*.gguf
gguf-plan gguf.json --pick Q5_K_M --lang ja
```

VRAM・物理コア数・そもそも `--device` が要るかどうかは**このマシンを見て決めます**。
フラグは要りません。`--vram 24` を渡すのは、**別のマシン向けに計画を立てたいとき**です。

**出力は既定で英語**です。`--lang ja` で日本語になります（設定ファイルに `lang = "ja"`
と書けば毎回付ける必要はありません）。

```console
# ===== Qwen3.8-27B-Q5_K_M / ctx 65,536 / KV f16 =====
# 見積り: モデル 18.47 + KV 4.32 + オーバーヘッド 1.00 = 23.78 GiB / 予算 24.0 GiB
# ⚠️ 余りが 0.22 GiB しかありません。実測では推論を通すとさらに 0.11〜0.14 GiB 増えたので …
# native ctx = 262,144  / rope scaling なし
# ハイブリッド注意: 17/65 層のみ KV 保持 = 68 KB/token
# KV f16 = 69.1 KB/token。GGUF からの計算ではなくこのマシンでの実測 (gguf-calibrate)

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

コマンドは4本。**「落とす」「読む」「決める」「測る」**で切ってあります。

| | |
| :-- | :-- |
| **`gguf-fetch`** | Hugging Face から**落とす**。ただし**落とす前に載るかを決める** |
| **`gguf-probe`** | GGUF を**読む**。何が書いてあるかを出す |
| **`gguf-plan`** | 読んだ結果と VRAM 予算から、**起動コマンドと config を出す** |
| **`gguf-calibrate`** | このマシンを1回**測る**。予算計算を推測でなくす |

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

KV の単価もオーバーヘッドも、**仮定ではなく実測**です。`gguf-calibrate` が ctx を2つ変えて
`llama-server` を起動し、直線を引きます（関係は完全な直線なので2点で確定します）。
Qwen3.8-27B-Q5_K_M / RTX 5090 / `-fa on` / `--spec-type draft-mtp` で:

| | 実測 | GGUF からの計算 | 倍率 |
| :-- | --: | --: | --: |
| KV f16 | **69.1 KB/token** | 68.0 | 1.02 |
| KV q8_0 | **43.1 KB/token** | 36.1 | **1.19** |
| 切片 | **19.15 GiB** | ファイルは 18.47 | → オーバーヘッド **0.68** |

4点の最大誤差は **0 MiB**。既定のオーバーヘッドは **1.0 GiB**。連続実行で見えた
さらに 0.03 GiB 分も、ここに含めて安全側に取ってあります。

> [!IMPORTANT]
> **この数字は環境で動きます。** llama.cpp のバージョン、バックエンド、起動フラグ、
> そして量子化の実装が変われば比も変わります。自分のマシンで `gguf-calibrate` を回して
> `gguf-fit.toml` に書いてください。`gguf-plan` は**実測値を使ったか計算値を使ったか**を
> 出力に書きます。

> [!WARNING]
> **単位を間違えると結論が反転します。** GGUF の `size_gb` はバイト ÷ 10⁹（**GB**）。
> GPU の「24GB」は **GiB**（÷ 2³⁰）で、10⁹ 換算だと **25.77 GB** あります。
> このツールはすべて GiB に揃えて計算します。

### 24GiB カードに何が載るか（単位はすべて GiB / 実測値で計算）

| 量子化 | ファイル | +32k | +64k | +128k | 最大ctx (f16) | 同 (q8_0) |
| :-- | --: | --: | --: | --: | --: | --: |
| Q3_K_S | 11.71 | 14.9 | 17.0 | 21.3 | 167,936 | 262,144 |
| Q3_K_M | 12.87 | 16.0 | 18.2 | 22.5 | 151,552 | 245,760 |
| IQ4_XS | 14.63 | 17.8 | 19.9 | 24.3 | 126,976 | 200,704 |
| IQ4_NL | 15.22 | 18.4 | 20.5 | 24.9 | 114,688 | 188,416 |
| Q4_K_M | 15.93 | 19.1 | 21.3 | 25.6 | 106,496 | 172,032 |
| UD-Q4_K_XL | 16.69 | 19.8 | 22.0 | 26.3 | 94,208 | 151,552 |
| **Q5_K_M** | 18.47 | 21.6 | **23.8** | 28.1 | **65,536** | **106,496** |
| UD-Q5_K_XL | 18.83 | 22.0 | 24.1 | 28.5 | 61,440 | 98,304 |
| Q6_K | 21.31 | 24.5 | 26.6 | 30.9 | 24,576 | 40,960 |
| UD-Q6_K_XL | 24.14 | 27.3 | 29.5 | 33.8 | 入らない | 入らない |
| Q8_0 | 27.05 | 30.2 | 32.4 | 36.7 | 入らない | 入らない |
| UD-Q8_K_XL | 29.30 | 32.5 | 34.6 | 38.9 | 入らない | 入らない |

- **Q5_K_M は 24GiB カードに ctx 64k / f16 で載ります**（23.8、余裕 0.2。かなりぎりぎり）
- **q8_0 の値打ちは「載せる」ことより「コンテキストを伸ばす」こと。** f16 で 64k、
  q8_0 なら **96k〜104k**
- KV を持つ層が 65層中17層しかないモデルなので、こういう伸び方をします

**この表を作るのに推論は一度も回していません。**（`gguf-calibrate` の4点を除く）

> [!NOTE]
> **以前ここに「q8_0 なら 128k」と書いていました。**理論値 0.531 で計算するとそうなります。
> 実測 (0.624) だと ctx 131,072 は **24.5 GiB** 必要で、24GiB カードには入りません。
> 計算値と実測値で答えが変わる、実際の例です。

<details>
<summary><b>量子化 KV は、計算どおりには縮まない</b></summary>

`q8_0` は 32値ごとに int8 32バイト + fp16 スケール1個 = 34バイト。f16 の 64バイトに対して
**0.531倍**。実測の比は **0.624** でした:

| ctx 65,536 で | 実測 | 0.531 からの予測 |
| :-- | --: | --: |
| KV f16 | 4.32 GiB | 4.25 |
| KV q8_0 | **2.69 GiB** | 2.26 |
| 節約 | **1.62 GiB** | 2.02 |

差の 0.4 GiB は**逆量子化の作業領域**が一番ありそうです。0.531 は「どう格納するか」の比で
あって、カーネルが動いている間に要る分は入っていません。

</details>

<details>
<summary><b>「いつ測るか」でも答えが変わる</b></summary>

同じサーバ・同じ ctx 65,536 で、4通りの読み方をすると（待機時からの増分）:

| | MiB | |
| :-- | --: | --: |
| ロード直後 | 23,922 | |
| 8トークンのリクエスト1回 | 24,022 | **+100** |
| 2,048トークンのリクエスト1回 | 24,032 | **+110** |
| 連続してベンチマークを回している最中 | 24,064 | **+142**（±50 で揺れる） |

**どの段の差も、f16 と q8_0 でまったく同じ値**でした。どれも KV の量に比例しないので、
これは**推論そのものが確保する分**です。しかもプロンプトを 256倍にして +10 MiB。
確保は「推論を1回でも通したか」でほぼ決まり、**プロンプト長では決まりません**。

まずいのは条件の違う点を混ぜること。ctx 32,768 をロード直後に、ctx 65,536 を推論中に測って
傾きを出すと、定数が傾きに化けて **73.5 KB/token** になります。条件をそろえると **69.1**。
このプロジェクトは実際にこれで間違えました。

**見積り式の形が正しいことは、測り直しで確かめられました。** ウォームアップを入れて測ると
4点とも同じ量だけ動き、**傾きは1バイトも変わらず**、切片だけが 19.04 → 19.15 GiB に
上がりました。定数は切片に乗るべきもので、実際にそこに乗りました。

いまの `gguf-calibrate` はサーバ自身の `/health` を待ち、バッチ1つ分のプロンプトを持つ
リクエストを1回通してから測り、**その1回で何 MiB 増えたか**を出力に書きます。到達するのは
+142 のうち +110 まで。残る約 32 MiB は追いかけていません —— プロンプトを伸ばしても出ない
ので、連続運転で出てくるもの（スロットの回転、KV の defrag、投機デコードのグラフ違いなど）
と見ています。既定の `overhead` が実測値ぴったりでないのはこのためです。

</details>

## 使い方

<details open>
<summary><b>gguf-fetch</b> — 落とす。ただし決めてから</summary>

```bash
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF                  # 判定だけ
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --fit            # 載るものを上から
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --pick Q5_K_M    # 指定した1本だけ
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --all            # リポジトリの GGUF 全部
```

`hf download` は落とすこと自体には何の不足もありません。足りないのは
**「このリポジトリの5本のうち、自分の 24 GiB に載るのはどれか」**という判断で、
それは普通、**全部落としてから** `gguf-probe` で調べることになります。
21 GB を5本落として4本消す、がいちばんありがちな失敗です。

順番を入れ替えます。ファイル一覧は数 KB。GGUF はヘッダが**ファイルの先頭に
固まっている**ので、HTTP の `Range` で数 MB 取れば層構造も native ctx も
MTP の有無も読めます。あとは `gguf-plan` と同じ式に通すだけです。

```console
$ gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --vram 24 --fit --lang ja
ornith-ai/Ornith-1.5-35B-A3B-GGUF  (main)
予算 24.0 GiB / オーバーヘッド 1.0 GiB / KV auto / ctx 16,384 以上で「載る」

量子化   ファイル    最大ctx(f16)    最大ctx(q8_0)   判定
---------------------------------------------------------
Q4_K_M     20.22G         131,072          249,856   → 落とす
Q5_K_M     23.61G        入らない         入らない   入らない
Q6_K       27.20G        入らない         入らない   入らない
Q8_0       35.21G        入らない         入らない   入らない
BF16       66.19G        入らない         入らない   入らない

# KV の数字は Q4_K_M のヘッダから読みました (12.0 MB 転送)。同じモデルの量子化違いは
# 層構造が同じなので、KV/token は全行にそのまま当てはまります。
# ビジョン投影が 1 本あります。mmproj-Ornith-1.5-35B-BF16.gguf (0.84 GiB) を付けます。
```

**候補 172 GB ぶんの判定に、転送は 12 MB。**

* **なぜ代表1本で足りるのか。** 量子化が変えるのはテンソルの**型**であって
  **名前**ではありません。だから同じリポジトリの量子化違いは層構造が同じで、
  KV/token も native ctx も MTP の有無も一致し、**動くのはファイルサイズだけ**です。
  出力には**どのファイルから読んだ数字か**を必ず書きます。仮定を置きたくなければ
  `--probe all` で全部のヘッダを読めます（本数ぶん転送が増えます）。
  `--probe none` はヘッダを読まず、**ファイルサイズだけ**の粗い判定になります
  （そうと分かるように書きます）。
* **`bpw` 列は実測です。**ファイルサイズ ÷ パラメータ数。パラメータ数は量子化で
  変わらないので、代表1本のヘッダから全行ぶん出ます。**このリポジトリの出発点は
  「ファイル名はビット幅について嘘をつく」でした。その嘘が数字で見えます:**
  `UD-Q6_K_XL` は名前に 6 と入っていて実測 **7.41 bpw**、`UD-Q8_K_XL` は **9.21**
  で `Q8_0` (8.51) より重い。検算になるのが `BF16` で、定義上ちょうど 16.00 に
  なるはず（実測 16.00 / 16.01）。ずれていたら数え方が間違っています。
* **候補を絞る3つの手。**

  | | |
  | :-- | :-- |
  | `--min-bpw 4.5` | **中身**で切る。1bit 量子化を比較相手から外すのに |
  | `--spread` | 上から N 本ではなく **bpw の幅を取って** N 本 |
  | `--only 'UD-Q?_K_*'` / `--exclude 'IQ*'` | **名前**で切る。品質の判断ではない |

  `--spread` が要るのは実物で困ったからです。unsloth/Qwen3.8-27B の `--top 3` は
  26.1 / 27.0 / 29.3 GiB を返しますが、これは 8.21 / 8.51 / 9.21 bpw で
  **3本とも同じビット帯**。83 GiB 落として比較できるのは1段ぶんもありません。
* **`--fit` が選ぶもの。** 載るもののうち大きいほうから、既定で `--top 3` 本。
  1本に絞らないのは、**予算の境界付近では見積りはあくまで見積り**だからです。
  1段下をディスクに置いておく価値は、そのディスク代より大きい。
  `--min-ctx`（既定 16,384）に届かないものは「載る」と数えません。
  起動はするが ctx 4k しか取れない、は答えになっていないからです。
* **`mmproj`。** ビジョン投影は自動で付けます（複数あれば最小の1本）。
  `--mmproj all` / `--mmproj none` で変えられます。
* **分割 GGUF。** `-00001-of-00003` は1つの候補にまとめ、サイズを合計します。
  別々に数えると「小さいモデルが3本あって全部載る」ように見えます。
* **サブディレクトリの GGUF は候補にしません。**`MTP/`・`imatrix/` などに入って
  いるものは本体ではないからです。ここは実物で踏みました → [そのままにしたバグ](#簡略化する前に)
* **書き込む前に**ディスクの空きを確かめ、実行する `hf download` をそのまま出して、
  聞きます。`--dry-run` は出すだけ、`-y` で確認を飛ばします。

落とすのは `hf download` の仕事なので、ファイル名の一覧を渡したら手を引きます。
設定ファイルに `models_dir` を書いておけば `--dir` は要りません。`HF_ENDPOINT` と
`HF_TOKEN` を見るので、ミラーでも gated リポジトリでも動きます。

</details>

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

`--target {llama-server,ollama,lmstudio}`（既定 `llama-server`）で出力形式を切り替えます。

```bash
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072 --target ollama
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072 --target lmstudio
```

見積り・余りの警告・「実測値か計算値か」の行は3形式とも同じです。変わるのは起動設定の
書式だけ。ただし Ollama の `Modelfile` も LM Studio の公開インターフェースも、
`llama-server` の CLI 全部を書けるわけではありません。`llama-server` には正確な値を、
残り2つには**目安と分かる形で**出せる範囲の近似値を出します。

* **GPU に何層乗せるか** — `num_gpu` は `docs.ollama.com/modelfile` から消えましたが、
  Ollama 側が「機能は残っている」と認めています
  ([ollama/ollama#13986](https://github.com/ollama/ollama/issues/13986))。gguf-fit は
  全層オフロードしか計画しないので、`Modelfile` には目安として `PARAMETER num_gpu 99`
  （`-ngl 99` と同じ考え方）を書きます — 保証はしません。LM Studio のロード API と
  `lms` CLI は 0〜1 の割合（`lms load --gpu 0.5`）しか受け付けず、層数そのものは指定
  できないので、出力は `--gpu max` のままにして、総層数はコメントとして添えます。
* **KV キャッシュの量子化（`q8_0`）** — Ollama で決めるのはサーバ全体の環境変数
  `OLLAMA_KV_CACHE_TYPE` で、モデル単位ではありません。出力する `Modelfile` は存在しない
  PARAMETER をでっち上げる代わりに、そう明記します。LM Studio の公開ロード API・CLI には
  相当する項目自体が見当たりません。計画が予算に収めるため `q8_0` を前提にしている場合、
  LM Studio の出力には「この予算は達成できないかもしれない」とはっきり書きます。
* **MTP / `--spec-type draft-mtp`** — Ollama には投機的デコード用の `draft_num_predict`
  というパラメータがありますが、それが llama.cpp の `--spec-type draft-mtp` と同じ仕組みで
  MTP テンソルを使うかどうかは確認していないので、何も設定しません。

</details>

<details open>
<summary><b>gguf-calibrate</b> — 測る。見積りを推測でなくす</summary>

```bash
gguf-calibrate --model /models/Qwen3.8-27B-Q5_K_M.gguf \
  --ctx 32768,65536 --kv f16,q8_0 \
  --extra "--spec-type draft-mtp"        # 本番と同じフラグを渡すこと
```

ctx ごとに `llama-server` を起動し、`/health` を待ち、バッチ1つ分のプロンプト
（`--warmup-tokens`、既定 2048）を持つリクエストを1回通し、`nvidia-smi` で増分を読み、
サーバを落とします。4点で数分。ベンチマークは回しません。

```console
calibration result

  f16     69.1 KB/token   intercept 19.15 GiB   (2 points, max error 0 MiB)
  q8_0    43.1 KB/token   intercept 19.11 GiB   (2 points, max error 0 MiB)

  q8_0 / f16 = 0.624   (the naive 34/64-byte figure would say 0.531)

  one request added 110 MiB on top of the load-time figure; that is what these
  numbers include

--- paste into gguf-fit.toml ---
kv_f16_bytes = 70720   # 69.1 KB/token, 2 points
kv_q8_bytes = 44096    # 43.1 KB/token, 2 points
```

`--write-config` を付ければ、貼り付けずに設定ファイルへ直接書きます。再実行すると
較正ブロックだけを差し替え、**手で書いたコメントや他のキーはそのまま残します**。
書く前に TOML として読み直すので、失敗して壊れたファイルが残ることはありません。

```bash
gguf-calibrate --model /models/your.gguf --write-config
# [updated] gguf-fit.toml
```

これで `gguf-plan` は GGUF からの計算値ではなく実測値を使います。`nvidia-smi` が
要ります。`--no-warmup` でリクエストを省けますが、その分 110 MiB ほど低く出ます。

同じ条件で2回通して4点とも 1 MiB の差もなく一致したので、**1回測れば足ります**。
あとで値が動いたら、それは環境のほうが変わったということです。

> [!NOTE]
> `--ctx` は**2つ以上必要**です。1点では傾きと切片を分離できません
> （「KV 比が 0.61」と「比は 0.531 のままオーバーヘッドが増えた」は、ctx 1点では
> 同じ数字を出します）。`gguf-calibrate` はどちらかに決めず、**測り直しを求めます**。

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

**まだ落としていないなら、落とす前に絞れます。** `gguf-fetch <repo>` はヘッダだけを
HTTP で読んで同じ表を出すので、200 GB を落としてから4本消す、をしなくて済みます。

</details>

<details>
<summary><b>ctx が足りない。何を削るか</b></summary>

| 手段 | 効果 | 代償 |
| :-- | :-- | :-- |
| **KV を q8_0 にする** | ctx 65,536 で実測 1.62 GiB 減。Q5_K_M で 64k → 96k | 品質影響は未測定 |
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
CLI フラグ  >  環境変数  >  設定ファイル  >  実測  >  組み込みの既定値
```

### 何を自動で見るか

| | 方法 | 使い道 |
| :-- | :-- | :-- |
| **VRAM** | `nvidia-smi`。複数枚なら一番大きいカード | `--vram` |
| **統合メモリ** | Apple Silicon は RAM の 75%（macOS が GPU に回せる量に上限がある） | `--vram` |
| **物理コア** | `/proc/cpuinfo` の `(physical id, core id)`、macOS は `hw.physicalcpu` | `--threads` |
| **CUDA が要るか** | NVIDIA が無ければ `--device` を**推測せず省略** | 起動コマンド |

`--threads` は**論理コアではなく物理コア**です。SMT の相方まで渡すと、行列積は速くならず
むしろ奪い合って遅くなることがあります。

> [!WARNING]
> NVIDIA が複数枚あるとき、`gguf-plan` は `CUDA0` を出しつつ注意書きを添えます。
> **`nvidia-smi` の並びは PCI 順で、CUDA のデバイス番号とは一致しません。**
> ここで番号を推測すると静かに別のカードを掴むので、推測しない方を選んでいます。
> `llama-server --list-devices` で確認してください。

### 書き留める

自動検出は楽ですが、そのマシンでしか効きません。`--write-config` で**いまの値をファイルに
固定**できます。`nvidia-smi` が無い環境でも、他人のハードを想定するときでも、同じ計画が出ます。

```console
$ gguf-plan --write-config
[written] gguf-fit.toml

# detected hardware
#   GPU         none; Apple Silicon unified memory
#   RAM         64.0 GiB
#   CPU         16 physical

lang = "ja"   # <- config
vram = 48.0   # <- detected
overhead = 1.0   # <- default
# device = ...   # 取得できませんでした
threads = 16   # <- detected
```

**各行に由来が残ります。**半年後に「この値は自分で決めたのか、マシンが言ってきたのか」を
見分けられるようにするためです。取得できなかった項目は `None` を書くと TOML が壊れるので、
コメントにして残します。既存ファイルは `--force` なしでは上書きしません。

```toml
# ./gguf-fit.toml  （または ~/.config/gguf-fit/config.toml、$GGUF_FIT_CONFIG）
lang     = "ja"
vram     = 24.0
overhead = 1.0
device   = "CUDA0"
port     = 8085

# gguf-fetch の落とし先。この下にリポジトリ名のサブディレクトリを作ります。
models_dir = "/mnt/data/models"

# gguf-calibrate の結果。あれば GGUF からの計算値より優先されます。
kv_f16_bytes = 70720
kv_q8_bytes  = 44096
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
**実測を設定ファイルより下に置いたのは意図的**です。書いた値がマシンに勝手に上書きされたら、
書く意味がありません。

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
- **1つのリポジトリに別のモデルが同居している場合。** `gguf-fetch` の既定
  (`--probe one`) は代表1本のヘッダを全行に当てます。**同じモデルの量子化違い**なら
  正しいですが、無関係なモデルが混ざっていれば外れます。`--probe all` で全部読めます

## 開発

```bash
uv sync            # .venv と開発ツール
uv run pytest -q   # 265件。GGUF ファイルもネットワークも不要
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
7. **リポジトリの GGUF が全部「本体」ではない。** unsloth/Qwen3.8-27B-GGUF には
   `MTP/mtp-...-Q4_0.gguf` (1.28 GiB) があります。`gguf-fetch` はこれを候補に入れ、
   ラベル `Q4_0` が本物の `Qwen3.8-27B-Q4_0.gguf` (14.95 GiB) と衝突したうえ、
   **一番小さいので代表に選ばれ**、その 4.0 KB/token (KV層 1/65) を全行に当てて
   いました。本物は 68.0 KB/token (17/65) で **17倍**。表はそれらしく出ます。
   → ルート直下だけを候補にし、代表は大きいほうから試し、`n_tensors >= block_count`
   を満たさないものは代表にしません
8. **較正値はモデル固有なのに設定ファイルは全モデルに効く。** Qwen3.8-27B で測った
   69.1 KB/token が、KV が 22.0 KB/token しかない Ornith-1.5-35B の計画にも使われ、
   最大 ctx が3倍近く低く出ました。`gguf-calibrate` が `kv_measured_on` と
   `kv_derived_f16_bytes` を書き残し、計算値どうしが 1.15 倍以上ずれたら警告します

</details>

## ライセンス

MIT
