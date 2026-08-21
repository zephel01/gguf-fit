# gguf-fit ドキュメント

[プロジェクトの README](../README.ja.md) が入口です。このツールが何を決めるのか、
数字がなぜその値なのか、そして動かし始めるのに足りるだけの使い方が書いてあります。
ここに置いてあるのは、その裏側のリファレンスです。

| | |
| :-- | :-- |
| [commands.ja.md](commands.ja.md) | 4本のコマンドの全フラグ。既定値と、その既定値である理由つき |
| [configuration.ja.md](configuration.ja.md) | `gguf-fit.toml` の全キー、優先順位、環境変数 |
| [architecture.ja.md](architecture.ja.md) | モジュール地図、純粋な中核、どの境界が効いているか |
| [../CHANGELOG.ja.md](../CHANGELOG.ja.md) | 何がいつ変わったか |

## どのページが何に答えるか

- *「このフラグは何をするのか、なぜ既定が 3 なのか」* → commands
- *「`vram` を書いたのに効かない」* → configuration、それから `--show-config`
- *「なぜ見積りがこうなるのか」* → README の VRAM の見積りの節、それから architecture
- *「メッセージ / 設定キー / コマンドはどこに足すのか」* → architecture

## わざと書いていないこと

コードにすでに書いてあって、実行時に出るもの。`--help` がフラグを並べ、`--show-config` が
どの値がどこから来たかを言い、`gguf-plan` の出力が KV の数字は実測か計算かを書きます。
それを繰り返すドキュメントは黙って古くなります。実行時の出力は古くなりません。

English: [README.md](README.md) · [commands.md](commands.md) ·
[configuration.md](configuration.md) ·
[architecture.md](architecture.md) ·
[../CHANGELOG.md](../CHANGELOG.md)
