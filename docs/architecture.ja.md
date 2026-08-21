# アーキテクチャ

どこに何が置いてあるか、そしてどの境界が効いているか。半年後に戻ってきて何かを「簡略化」
したくなる自分に向けて書いてあります。

## モジュール地図

```
src/gguf_fit/
  __init__.py     public API surface
  probe.py        read a local GGUF                 -> gguf-probe
  plan.py         budget arithmetic, launch output  -> gguf-plan
  calibrate.py    measure this machine              -> gguf-calibrate
  fetch.py        decide, then download             -> gguf-fetch
  _ggufhdr.py     GGUF header parser over bytes
  _hardware.py    GPU / RAM / CPU detection
  _config.py      config resolution with provenance
  _messages.py    en/ja message catalogue, display width
```

| モジュール | 依存 | 外の世界に触るか |
| :-- | :-- | :-- |
| `_messages` | — | 触りません |
| `_ggufhdr` | —（知らない型名のために `gguf` を任意で使います） | 触りません |
| `_config` | `_messages` | 設定ファイルを読みます |
| `_hardware` | — | `nvidia-smi`、`llama-server --list-devices` を実行し、sysfs を読みます |
| `probe` | `_config`, `_messages`, `gguf` | GGUF を mmap します |
| `plan` | `probe`, `_config`, `_hardware`, `_messages` | probe の JSON を読みます |
| `calibrate` | `_hardware`, `probe`（config を書くときだけ） | `llama-server` を起動し、HTTP、`nvidia-smi` |
| `fetch` | `plan`, `probe`, `_ggufhdr`, `_config`, `_hardware`, `_messages` | Hugging Face への HTTP、`hf download` の実行 |

`fetch` が `plan` を import しているのは意図的です。見積りは1か所にしか無いようにします。
式が2つあると、同じモデルに答えが2つ出ます。

## 純粋な中核

分類も予算計算も純粋関数です。テストがモデルファイル1本もネットワークもなしで走るのは
そのためです。

- `probe.summarize_tensors(tensors, meta)` — `(name, type)` の組を受け取り、量子化の配合・ハイブリッド注意の判定・KV サイズを返します
- `plan.max_ctx` / `recommended_ctx` / `kv_gib` / `file_gib` / `headroom_gib` — レコードの dict と予算を受け取ります
- `_ggufhdr.parse_header(data, want)` — `bytes` を受け取ります
- `fetch.group_files` / `quant_label` / `bits_per_weight` / `choose` / `filter_candidates` — ただのデータを受け取ります

ネットワークとプロセスの境界は、その上の薄いラッパです。薄いまま保ってください。

## GGUF リーダが2つあるのは意図的

`probe.py` は `gguf` パッケージの `GGUFReader` を使い、これは mmap するのでファイルが
ディスクに要ります。`gguf-fetch` はファイルが存在する*前*に決めなければならないので、
`_ggufhdr.py` が HTTP で取ったバイト範囲からヘッダを解析します。

2つは `summarize_tensors` で合流します。どちらも `(tensor name, type name)` の組を渡すので、
その先の KV 計算はどちらの経路でも同じです。

`_ggufhdr` は「まだバイトが足りない」を `TruncatedGGUF` で、何バイト要るかを添えて伝え、
「これは永久に解析できない」を `ValueError` で伝えます。この2つを1つの例外にまとめると、
あと1回 range リクエストを投げれば読めたファイルを諦めるか、永遠に取り続けるかのどちらかに
なります。

## 効いている境界

**`_messages` が人間向けの文字列を全部持ちます。**1つ足すなら両方の言語に足します。片方だけ
だとテストが落ちます。フラグ名・JSON のキー・生成する config の中身は翻訳しません。読むのは
機械です。

**`_config.resolve()` は値*と*その由来を返します。**「この数字はどこから来たのか」に答えられる
ことが売りです。設定を裏で解決するものは、それを壊します。

**`_hardware` は `_messages` を import しません。**報告するのはマシンの事実であって、文章では
ありません。その事実を使う翻訳ずみの警告は `plan.budget_warnings()` にあり、`gguf-plan` も
`gguf-fetch` も同じものを呼びます。片方だけがバックエンド混在を警告して、もう片方は黙っている、
が起きないようにするためです。

**`probe` や `fetch` を `__init__` に名前として持ち上げないでください。**サブモジュール名を
関数に束ね直すと、`from gguf_fit import probe` が関数を返します。テストが一度これを捕まえて
います。関数は `read_gguf`、`group_files`、`quant_label`、`parse_gguf_header` として export
してあります。

## 見積りの出どころ

```
使用量 = モデルファイル + KVキャッシュ + オーバーヘッド
```

- **モデルファイル** — ファイルサイズ、単位は GiB。GGUF の `size_gb` はバイト ÷ 10⁹ で別の単位です。ここでは全部 GiB に揃えます。
- **KV キャッシュ** — bytes/token × ctx。bytes/token が数えるのは `attn_k`/`attn_v` を実際に持つ層だけで、`block_count` ではありません。間違えると、ハイブリッド注意のモデルで 4× の過大評価になります。
- **オーバーヘッド** — 既定 1.0 GiB。実測は 0.68 で、切り上げてあります。

`gguf-calibrate` は前の2つを実測値に置き換えます。どちらを使ったかは出力に必ず書きます。

## テスト

テストは 295 件。GGUF ファイルもネットワークも要りません。

- ヘッダ解析は、テストファイルの中で組み立てた合成 GGUF バイト列で動かします。
- `gguf-fetch` の HTTP 経路は、`HF_ENDPOINT` を向けた localhost の `http.server` に対して走ります。わざと `Range` を無視するハンドラも含めて、クライアントが止まることを示します。
- ハードウェア検出は、記録しておいた `nvidia-smi` / `--list-devices` の出力に対して走ります。

テスト名には、呼ぶ関数ではなく、防いでいる欠陥を書きます。いくつかは、一度バグを出荷したから
存在しています。README の「簡略化する前に」を見てください。

## コマンドを足す

1. `src/gguf_fit/` に `main() -> int` を持つモジュール
2. `[project.scripts]` にエントリ
3. `_messages.py` にメッセージを両方の言語で
4. 設定を読むなら `_config.KNOWN_KEYS` にキー
5. `_config.resolve_llama_servers()` 経由で `_hardware.detect()`。他と同じデバイスを見るように
6. VRAM 予算を使うなら `plan.budget_warnings()`
