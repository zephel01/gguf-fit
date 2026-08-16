"""gguf-fit — GGUF を読んで「自分の GPU に載るか」を、回さずに決める.

推論を1トークンも行わずに、GGUF のヘッダとテンソル一覧から
``--ctx-size`` の上限・KVキャッシュの VRAM・``--spec-type draft-mtp`` の
可否を確定させる。mmap で開くだけなので 27B のファイルでも1秒。

コマンドは2本:

    gguf-probe   ファイルに何が書いてあるかを読む
    gguf-plan    読んだ結果と VRAM 予算から起動コマンドと config を出す

ライブラリとしても使える。中核はどちらも純粋関数で、GGUF ファイルも
``gguf`` パッケージも要らない。

    from gguf_fit import summarize_tensors, recommended_ctx

    # テンソル一覧 (名前, 型) から量子化の配分と KV サイズを出す
    s = summarize_tensors(
        [("blk.3.attn_k.weight", "Q5_K"), ("blk.0.attn_qkv.weight", "Q5_K")],
        {"block_count": 65, "head_count_kv": 4,
         "key_length": 256, "value_length": 256},
    )
    s["kv_cache"]["bytes_per_token_f16"]

    # VRAM 予算から ctx を逆算する
    recommended_ctx(rec, vram_gib=24.0, kv_mode="q8_0", overhead=1.0)
"""

from .plan import (
    Q8_FACTOR,
    file_gib,
    headroom_gib,
    kv_gib,
    max_ctx,
    max_tokens_for,
    recommended_ctx,
)

# ``probe`` という名前でここに関数を持ち上げてはいけない。
# サブモジュール ``gguf_fit.probe`` を隠してしまい、
# ``from gguf_fit import probe`` がモジュールではなく関数を返すようになる。
# (実際にテストがそれで落ちた)。関数は read_gguf として公開する。
from .probe import probe as read_gguf
from .probe import render, summarize_tensors

__version__ = "0.1.0"

__all__ = [
    "Q8_FACTOR",
    "__version__",
    "file_gib",
    "headroom_gib",
    "kv_gib",
    "max_ctx",
    "max_tokens_for",
    "read_gguf",
    "recommended_ctx",
    "render",
    "summarize_tensors",
]
