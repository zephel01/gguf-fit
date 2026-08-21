# 設定リファレンス

`gguf-fit.toml` に書けるキー全部と、どこから来るか、どれが勝つか。いま使っているマシンでの
答えは `--show-config` が由来つきで出します。このファイルはその裏側の地図です。

## 優先順位

```
CLI フラグ  >  環境変数  >  設定ファイル  >  実測  >  組み込みの既定値
```

実測を設定ファイルより**下**に置いたのは意図的です。書いた値がマシンに黙って上書きされたら、
書く意味がありません。ハードを入れ替えたときは、`--refresh` が `vram` / `threads` / `device`
についてこれを逆にします。

## ファイルを探す場所

**最初に見つかった1つだけを使います。**マージはしません。重ねると「この値はどこから来たのか」
が答えられなくなり、それを答えられることがこのツールの売りだからです。

1. `--config PATH`
2. `$GGUF_FIT_CONFIG`
3. `./gguf-fit.toml`
4. `~/.config/gguf-fit/config.toml`（`$XDG_CONFIG_HOME` を見ます）— Windows では `%APPDATA%\gguf-fit\config.toml`

未知のキー・型違い・壊れた TOML は、黙って無視せず stderr に出します。設定ファイルが効いて
いないことに気づけないのが一番困ります。

## 環境変数

どのキーも `GGUF_FIT_` + キーの大文字にしたものになります: `GGUF_FIT_LANG`、
`GGUF_FIT_VRAM`、`GGUF_FIT_MODELS_DIR`。リスト値のキーはカンマ区切りです。

`gguf-fetch` はさらに、このツールではなく `huggingface_hub` のものである2つを見ます:
`HF_ENDPOINT`（ミラー）と `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `HUGGINGFACE_TOKEN`
（gated リポジトリ）。トークンは `$HF_HOME/token` か `~/.cache/huggingface/token` からも
読みます。

## キー

### 出力

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `lang` | str | `"en"` | `"en"` か `"ja"`。フラグ名・JSON のキー・生成する config の中身は翻訳しません。読むのは機械です。 |

### 予算

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `vram` | float | 自動検出 | **1台**ぶんの GiB。`llama-server --list-devices` から、無ければ `nvidia-smi` から取ります。Apple Silicon では RAM の 75%。 |
| `overhead` | float | `1.0` | GiB。あるマシンでの実測は 0.68。連続運転で出てくる約 0.03 GiB 分を吸収するために切り上げてあります。 |

### 較正した KV の単価

`gguf-calibrate --write-config` が書きます。あれば GGUF からの計算値より優先されます。計算値は
低めに出ます: f16 は計算 68.0 に対して実測 69.1 KB/token、q8_0 は計算 36.1 に対して実測 43.1。

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `kv_f16_bytes` | float | *未設定* | トークンあたりのバイト数、f16。 |
| `kv_q8_bytes` | float | *未設定* | トークンあたりのバイト数、q8_0。 |
| `kv_measured_on` | str | *未設定* | どのファイルで測った数字か。人が読むためのものです。 |
| `kv_derived_f16_bytes` | float | *未設定* | そのモデルの GGUF からの計算値。機械が比べるためのものです。 |

> **これはモデルごとの値です。** KV の単価は層構造で決まるので、Qwen3.8-27B で測った
> 69.1 KB/token は、KV が 22.0 しかない Ornith-1.5-35B を説明しません。当てはめると最大 ctx が
> 3× 近く低く出ました。`kv_derived_f16_bytes` があり、いま扱っているモデルの計算値がそこから
> 1.15× 以上ずれていれば、`gguf-plan` も `gguf-fetch` もそう書きます。このキーが無い古い設定
> ファイルには警告が出ません。比べる相手がいないからです。

### ハードウェア検出

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `llama_servers` | list | `["llama-server"]` | 使うビルドを全部。**CUDA ビルドの `--list-devices` には ROCm デバイスが出ません。**1つだけ書くと、実在する GPU を見落とします。 |
| `llama_server` | str | — | 同じものの単数形。ビルドが1つの人向けです。 |
| `device` | str | 自動検出 | `CUDA0` / `ROCm0` / `Vulkan0`。 |
| `threads` | int | 物理コア | |

### 生成する起動コマンド

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `port` | int | `8085` | |
| `model_path` | str | probe の JSON から | まず要りません。 |

### gguf-fetch

| キー | 型 | 既定 | 補足 |
| :-- | :-- | :-- | :-- |
| `models_dir` | str | `.` | この下にリポジトリごとのサブディレクトリを作って落とします。書いておく価値があります。既定はカレントディレクトリなので、ソースを展開した中で実行すると数十 GB がそこに入り、しかも `.gitignore` に `*.gguf` があるので `git status` は綺麗なまま、そのまま忘れます。 |
| `hf_bin` | str | `hf`、次に `huggingface-cli` | |

## そもそもマシン固有のもの

`gguf-fit.toml` は `.gitignore` に入れてあり、そのままにしてください。`--write-config` は
VRAM とコア数を書き残すので、別のマシンに持っていくと、そこに無い GPU 向けの計画を出します。
追跡してあるテンプレートのほうが `gguf-fit.example.toml` です。

`--write-config` が書く各行には、由来がコメントで残ります (`<- detected`、`<- default`、
`<- cli`)。半年後に、どの数字を自分で決めて、どれをマシンが言ってきたのかを見分けるためです。
