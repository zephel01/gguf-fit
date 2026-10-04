# Pirate Face / swarm からのダウンロード

`gguf-fetch` は、Hugging Face (以下 HF) からモデルが消えたときに、[Pirate Face](https://pirateface.co) の索引と BitTorrent の swarm から落とせます。ここは、そのためのインストールと使い方です。全フラグの一覧は [commands.ja.md](commands.ja.md) にあります。

## いつ使うか

- HF から消えたモデルを落としたい。
- HF のリポジトリが 401/403/404/410/451 を返すようになった。

HF が生きているあいだは、今までどおり HF から落とします。何も変えなくて構いません。

## インストール

必要なものは3つで、**用途ごとに分かれています**。

| 用途 | 必要なもの | 入れ方 |
| :-- | :-- | :-- |
| 基本 | Python 3.10 以上と `gguf-fit` | 下の「本体」 |
| swarm から落とす | `aria2c` | macOS: `brew install aria2` / Debian 系: `apt install aria2`。他の OS は、パッケージ名 `aria2` で入れてください |
| HF から落とす (`--via hf`、`--via auto` の切り替え先) | `hf` (旧名 `huggingface-cli`) | `pip install -U huggingface_hub` |

### 本体

```bash
uv tool install git+https://github.com/zephel01/gguf-fit
```

インストールせず1回だけ試すなら:

```bash
uvx --from git+https://github.com/zephel01/gguf-fit gguf-fetch --help
```

pip なら `pip install git+https://github.com/zephel01/gguf-fit` です。

### 入ったか確認する

```bash
aria2c --version | head -1
gguf-fetch --help | grep -E -- '--(source|via|aria2-bin)'
```

2行目に3つのフラグが出れば、Pirate Face 対応が入っています。

## 使い方

```bash
# 1. 判定だけ (何も落ちない。swarm のメタデータだけ読む)
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF --dry-run --fit

# 2. 載るものを swarm から落とす
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF --fit --dir ~/models

# 3. 量子化を指定して、swarm だけを使う
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF \
  --pick Q4_K_M --mmproj none --via swarm --dir ~/models
```

`--fit`、`--pick`、`--top`、`--mmproj` などの落とし方の指定は、HF のときと同じです。

### 取得元の決まり方 (`--source`)

| 渡したもの | `auto` (既定) の動作 |
| :-- | :-- |
| `https://pirateface.co/owner/name` | Pirate Face |
| `https://huggingface.co/owner/name` | HF |
| `owner/name` | まず HF。401/403/404/410/451 のときだけ Pirate Face に切り替える。ネットワークエラーでは切り替えない |

`--source hf` か `--source pirateface` で固定できます。

### 落とし方の決まり方 (`--via`)

| `--via` | 動作 |
| :-- | :-- |
| `auto` (既定) | swarm を先に試し、失敗したら `hf download --revision <固定した commit>` に切り替える。`aria2c` が無ければ最初から `hf` |
| `swarm` | magnet と `aria2c` だけを使う。失敗しても切り替えない |
| `hf` | 固定したリビジョンで `hf download` する。swarm は使わない |

### 何が起きるか

1. Pirate Face のページから、固定リビジョン・magnet・ファイルごとの SHA-256・概算サイズを読みます。
2. swarm が取得経路になるときは、**判定の前に** torrent のメタデータ (数十 KiB、数秒) を取り、**torrent に実際に入っているファイルだけ**を候補にします。入っていないものは名前を出して外します。
3. 選んだファイルだけを `aria2c --select-file` で取ります。
4. 保存先の直下に整理して、ページの SHA-256 と照合します。一致しなければ「完了」とは言わず、終了コード 1 で終わります。

**ページに載っているファイルが torrent にあるとは限りません。** 2026-10 に確認した `mradermacher/Ornith-1.5-9B-uncensored-GGUF` は、ページに 12 量子化が載っていましたが、torrent には `Q4_K_M` の 1 本 (5.2 GiB) しか入っていませんでした。そのため、手順 2 で候補を絞ります。`--pick` で torrent に無い量子化を指定すると、転送を始める前に「当たるものがありません」で止まります。

HF にそのリビジョンが残っているあいだは、サイズも GGUF ヘッダも HF から読み、判定は今までどおりです。消えた後はヘッダを読めないので、**ファイルサイズだけの判定**になり、そう表示します (サイズは 0.1 GB 刻み)。KV キャッシュと最大 ctx は、落としてから測るまで分かりません。

`--via auto` で HF が生きているときは、どちらの経路でも取れるので、torrent は先に読みません。

### `--dry-run`

実行するコマンドを表示して終わります。本体は落としません。ただし swarm が経路になるときは、判定のために torrent のメタデータを読みます。

### 保存先

```
<dir>/<リポジトリ名>/
    Model.Q4_K_M.gguf          ← 選んだファイル (hf download と同じ配置)
    <torrent の名前>/…          ← 選ばなかった隣のファイルの欠片 (出たときだけ)
```

aria2c は、選んだファイルと piece を共有する隣のファイルの端も取るため、欠片が残ることがあります。使えないので、不要なら `<torrent の名前>/` ごと手で消してください。ツールは消しません。

同じコマンドをもう一度流すと、SHA-256 が記録と合うファイルは落とし直しません (メタデータだけ読み直します)。

## 設定ファイル

毎回フラグを付けたくないときは、`gguf-fit.toml` に書きます。

```toml
models_dir = "~/models"
source = "auto"        # auto / hf / pirateface
aria2_bin = "aria2c"   # 別の場所の aria2c を使うとき
```

環境変数は、`HF_ENDPOINT` と `HF_TOKEN` (HF のミラー・gated リポジトリ) と、`PIRATEFACE_ENDPOINT` (Pirate Face のミラー) を見ます。優先順位は [configuration.ja.md](configuration.ja.md) にあります。

## うまくいかないとき

| 症状 | 原因と対処 |
| :-- | :-- |
| `aria2c が見つかりません` (終了コード 127) | `aria2c` を入れるか、`--aria2-bin` で場所を渡します。`--via hf` なら不要です |
| メタデータ取得で 180 秒待ったあと `torrent のメタデータを取得できませんでした` | peer が応答しないか、swarm が空です。aria2c の終了コードがそのまま返ります。`--via auto` なら HF に切り替わります (HF から消えていれば取れません) |
| `ページに載っている GGUF が torrent には1本も入っていません` | その torrent に GGUF が無い。別の取得元を探してください |
| `--pick X に当たるものがありません。あるのは: …` | X は torrent に入っていません。「あるのは」に出たものから選びます |
| `SHA-256 が一致しません` (終了コード 1) | 壊れたファイルを「完了」にはしません。使わずに消して、もう一度流してください |
| `Pirate Face で … を読めませんでした` | ネットワークの問題か、Pirate Face のページ構造が変わった可能性があります。ページを読めないと Pirate Face 経由では進めません。HF にまだあるなら `--source hf` を使ってください |

## 注意

- Pirate Face の web-seed は、固定した commit の HF への 302 リダイレクトです。HF から消えたあとに効くのは swarm だけです。
- swarm から取ったファイルの出所とライセンスの確認は、利用者の責任です。SHA-256 の照合は、Pirate Face の記録と合っているかを確かめるもので、モデルの安全性を保証しません。
- 実行されるコマンドは、そのまま表示されます。不安なときは `--dry-run` で確認してから流してください。
