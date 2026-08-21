"""GGUF の**先頭だけ**を読むパーサ。ファイル全体が手元に無くても動く.

``gguf-probe`` は ``gguf`` パッケージの ``GGUFReader`` に任せている。あれは
ファイルを mmap するので、**ローカルに実体があること**が前提になる。

``gguf-fetch`` はダウンロードする前に判断したい。GGUF はヘッダ（メタデータ +
テンソル一覧）がファイルの**先頭に固まっている**ので、HTTP Range で数MB〜
数十MB だけ取れば、KVキャッシュの単価も native ctx も MTP の有無も分かる。
21 GB のファイルを落とさずに済む。

そのために、必要な範囲だけを自前で読む。設計は3点:

  * **純粋関数**。``bytes`` を受けて結果を返すだけ。ネットワークもファイルも
    触らないので、合成したバイト列でテストできる（実際そうしている）。
  * **足りなければ「何バイト要るか」を言って止まる**（``TruncatedGGUF``）。
    呼び出し側はその分だけ追加で取ってやり直せばよい。全部取ってから
    「足りませんでした」と言うより、転送量が読める。
  * **要らない値は読み飛ばす**。トークナイザの語彙は 15万語あって
    10 MB を超えるが、``gguf-fetch`` に必要なのは層数と head 数だけ。
    配列はバイト数だけ計算して飛ばす（文字列配列は長さが可変なので歩く）。

実測: Ornith-1.5-35B-Q4_K_M（21.7 GB）のヘッダは **10.48 MiB**。
12 MiB 取れば足りた。ファイル本体の 0.05% で済む。
"""

from __future__ import annotations

import struct
from typing import Any, NamedTuple

#: GGUF のマジック
MAGIC = b"GGUF"

#: ggml のテンソル型。**gguf パッケージが無くても名前を出せるように**ここに持つ。
#: ここに無い番号（将来増える型）は gguf パッケージに聞きにいき、それも
#: 駄目なら ``TYPE_<番号>`` にする。**黙って F32 などに丸めてはいけない**。
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0",
}

#: メタデータ値の型 -> (struct フォーマット, バイト数)
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

#: 壊れたバイト列で天文学的なループに入らないための上限。
#: 実在する GGUF はテンソル数千・KV 数百のオーダーなので、けた違いに広く取ってある。
MAX_TENSORS = 1_000_000
MAX_KV = 100_000
MAX_ARRAY = 100_000_000

#: 文字列配列から手元に残す最大要素数。語彙 15万語を丸ごと持っても使い道が無い。
#: **飛ばすのではなく先頭だけ残す**のは、chat_template のような「中身を見たい」
#: 配列がまれにあるため。
KEEP_ARRAY_ITEMS = 64


class TruncatedGGUF(Exception):
    """バイト列が途中で終わっている。``need`` バイト目まであれば続けられる.

    「壊れている」ではなく「**まだ足りない**」を表す例外。``gguf-fetch`` は
    これを受けて追加のレンジ取得に行く。``ValueError``（本当に壊れている）と
    区別できないと、取り直せば読めるものを諦めることになる。
    """

    def __init__(self, need: int) -> None:
        super().__init__(f"need at least {need} bytes")
        self.need = need


class Header(NamedTuple):
    """ヘッダから読めたもの."""

    version: int
    #: メタデータ。``want`` に合致したキーだけが入る（既定は全部）
    metadata: dict[str, Any]
    #: ``(テンソル名, 型名)`` の並び。``summarize_tensors`` にそのまま渡せる
    tensors: list[tuple[str, str]]
    #: ヘッダが何バイトで終わったか。ここから先が実データ
    header_bytes: int
    #: 全テンソルの要素数の合計 = **パラメータ数**。
    #: 量子化しても変わらない（型が変わるだけ）ので、代表1本読めば
    #: 同じリポジトリの全量子化について「実測 bpw」が出せる
    n_params: int = 0


def type_name(code: int) -> str:
    """ggml のテンソル型番号を名前にする."""
    known = GGML_TYPE_NAMES.get(code)
    if known:
        return known
    try:
        from gguf import GGMLQuantizationType  # noqa: PLC0415 - 未知の型のときだけ聞く

        return GGMLQuantizationType(code).name
    except Exception:  # noqa: BLE001 - import 失敗も未知の値も同じ扱いでよい
        return f"TYPE_{code}"


class _Cursor:
    """読み進める位置を持つだけの器。**足りなければ TruncatedGGUF**."""

    __slots__ = ("buf", "i", "n")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.i = 0
        self.n = len(buf)

    def skip(self, count: int) -> int:
        """``count`` バイト読み飛ばして、飛ばす前の位置を返す（コピーしない）."""
        if count < 0:
            raise ValueError("negative length in GGUF header")
        start = self.i
        end = start + count
        if end > self.n:
            raise TruncatedGGUF(end)
        self.i = end
        return start

    def take(self, count: int) -> bytes:
        start = self.skip(count)
        return self.buf[start:self.i]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self, keep: bool = True) -> str | None:
        length = self.u64()
        if length > MAX_ARRAY:
            raise ValueError(f"implausible string length {length}")
        start = self.skip(length)
        if not keep:
            return None
        return self.buf[start:self.i].decode("utf-8", "replace")


def _read_value(cur: _Cursor, vtype: int, keep: bool) -> Any:
    """1つの値を読む。``keep`` が偽なら**位置だけ進めて中身は捨てる**."""
    if vtype == _TYPE_STRING:
        return cur.string(keep)
    if vtype == _TYPE_ARRAY:
        elem = cur.u32()
        count = cur.u64()
        if count > MAX_ARRAY:
            raise ValueError(f"implausible array length {count}")
        if elem == _TYPE_ARRAY:
            # GGUF の仕様上ありうるが実物で見たことがない。**推測で読まない**
            raise ValueError("nested arrays are not supported")
        if elem == _TYPE_STRING:
            out: list[str] = []
            for _ in range(count):
                # 文字列は長さが可変なので、飛ばすにも1つずつ歩くしかない
                value = cur.string(keep and len(out) < KEEP_ARRAY_ITEMS)
                if value is not None:
                    out.append(value)
            return out if keep else None
        if elem not in _SCALARS:
            raise ValueError(f"unknown array element type {elem}")
        fmt, size = _SCALARS[elem]
        start = cur.skip(size * count)
        if not keep:
            return None
        return list(struct.unpack(f"<{count}{fmt[1]}", cur.buf[start:cur.i]))
    if vtype not in _SCALARS:
        raise ValueError(f"unknown metadata value type {vtype}")
    fmt, size = _SCALARS[vtype]
    return struct.unpack(fmt, cur.take(size))[0]


def parse_header(data: bytes, want: tuple[str, ...] | None = None) -> Header:
    """GGUF の先頭バイト列からメタデータとテンソル一覧を読む.

    ``want`` を渡すと、**そのいずれかで終わるキーだけ**を手元に残す
    （残りは位置だけ進めて捨てる）。語彙の配列を持ち歩かずに済む。

    :raises TruncatedGGUF: バイト列が途中で終わっている（追加取得すれば読める）
    :raises ValueError: マジックが違う、型が未知など、取り直しても直らないもの
    """
    cur = _Cursor(data)
    if cur.take(4) != MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    version = cur.u32()
    n_tensors = cur.u64()
    n_kv = cur.u64()
    if n_tensors > MAX_TENSORS or n_kv > MAX_KV:
        raise ValueError(f"implausible header: {n_tensors} tensors, {n_kv} kv")

    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = cur.string()
        vtype = cur.u32()
        keep = want is None or any(key.endswith(suffix) for suffix in want)
        value = _read_value(cur, vtype, keep)
        if keep:
            metadata[str(key)] = value

    tensors: list[tuple[str, str]] = []
    n_params = 0
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.u32()
        # 形は要素数を数えるためだけに読む。**ここを飛ばすと bpw が出せない**
        elements = 1
        for _dim in range(n_dims):
            elements *= cur.u64()
        n_params += elements
        code = cur.u32()
        cur.skip(8)                   # データオフセット。ここでは要らない
        tensors.append((str(name), type_name(code)))

    return Header(version, metadata, tensors, cur.i, n_params)
