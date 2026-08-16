"""このマシンの GPU / メモリ / CPU を見て、既定値を自分で決める.

``--vram`` を毎回打たせない、``--threads`` を出せるようにする、が目的。
**検出は必ず失敗しうる**ので、どれも「取れなかった (None)」を返せるようにし、
呼び側が既定値や明示指定に落とせる形にしてある。

検出の手段:

  GPU     nvidia-smi (無ければ諦める)
  統合メモリ  macOS の Apple Silicon は VRAM が独立していないので RAM から推定
  RAM     Linux: sysconf / macOS: sysctl / Windows: GlobalMemoryStatusEx
  CPU     物理コア数。llama.cpp の --threads は**論理ではなく物理**に合わせる
          のが定石 (SMT の相方を使っても行列積は速くならない)

**Apple Silicon の 0.75 について**: macOS は GPU に回せる量に上限があり、
既定はおおよそ実装メモリの 3/4。全部使える前提で計算すると確実に外すので、
ここで削っておく。``iogpu.wired_limit_mb`` を上げている人は ``--vram`` で
明示すること。
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from typing import NamedTuple

#: Apple Silicon で GPU に回せると見なす割合
UNIFIED_MEMORY_SHARE = 0.75
#: 外部コマンドを待つ秒数。検出のために起動が止まるのは本末転倒
PROBE_TIMEOUT_S = 5


class Gpu(NamedTuple):
    index: int
    name: str
    total_gib: float
    used_gib: float


class Hardware(NamedTuple):
    gpus: list[Gpu]
    ram_gib: float | None
    physical_cores: int | None
    logical_cores: int | None
    unified_memory: bool  # Apple Silicon のように VRAM と RAM が同じか

    @property
    def largest_gpu(self) -> Gpu | None:
        return max(self.gpus, key=lambda g: g.total_gib) if self.gpus else None

    def suggested_vram_gib(self) -> float | None:
        """``--vram`` の既定として使える値。取れなければ None."""
        if self.gpus:
            return self.largest_gpu.total_gib
        if self.unified_memory and self.ram_gib:
            return round(self.ram_gib * UNIFIED_MEMORY_SHARE, 1)
        return None

    def suggested_threads(self) -> int | None:
        """``--threads`` の既定。物理コアを優先し、無ければ論理コア."""
        return self.physical_cores or self.logical_cores

    def suggested_device(self) -> str | None:
        """``--device`` の既定。**分からないときは None を返して省略させる**.

        NVIDIA が見えないのに ``CUDA0`` を書くと、そのコマンドは起動しない。
        Apple Silicon は Metal が既定で選ばれるので ``--device`` は要らない。

        NVIDIA があるときも ``CUDA0`` 固定にしている。**nvidia-smi の並び順と
        CUDA のデバイス番号は一致しない** (nvidia-smi は PCI 順、CUDA は既定で
        性能順)。番号を勝手に振ると静かに別のカードを掴むので、
        ここは推測せず、複数枚あるときは注意書きを出す方を選ぶ。
        """
        return "CUDA0" if self.gpus else None

    def device_index_is_ambiguous(self) -> bool:
        """複数枚あって、nvidia-smi と CUDA の番号がずれうるか."""
        return len(self.gpus) > 1


def _run(cmd: list[str]) -> str | None:
    """外部コマンドを叩く。取れなければ None。**例外は投げない**."""
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def detect_gpus() -> list[Gpu]:
    """nvidia-smi から GPU を拾う。無ければ空リスト."""
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            # MiB / 1024 をそのまま持つと 31.8427734375 のような値が
            # 設定ファイルにまで流れる。ここで丸めておく。
            gpus.append(Gpu(int(parts[0]), parts[1],
                            round(int(parts[2]) / 1024, 2),
                            round(int(parts[3]) / 1024, 2)))
        except ValueError:
            continue
    return gpus


def detect_ram_gib() -> float | None:
    """搭載メモリ (GiB)。取れなければ None."""
    if sys.platform == "win32":  # pragma: no cover - Windows でのみ通る
        import ctypes  # noqa: PLC0415 - Windows でしか使わない

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return round(st.ullTotalPhys / 1024 ** 3, 1)
        return None

    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            return round(int(out.strip()) / 1024 ** 3, 1)
        return None

    try:  # Linux
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                     / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def detect_cores() -> tuple[int | None, int | None]:
    """(物理コア, 論理コア) を返す.

    llama.cpp の ``--threads`` は物理コアに合わせるのが定石。論理コア数を
    渡すと SMT の相方を奪い合って**遅くなる**ことがある。
    """
    logical = os.cpu_count()

    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out and out.strip().isdigit():
            return int(out.strip()), logical
        return None, logical

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return None, logical
        # (physical id, core id) の組を数えるのが唯一まともな方法。
        # "cpu cores" だけ見るとソケット数を掛け忘れる。
        seen = set()
        pkg = core = None
        for line in text.splitlines():
            if line.startswith("physical id"):
                pkg = line.split(":")[-1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[-1].strip()
            elif not line.strip() and pkg is not None and core is not None:
                seen.add((pkg, core))
                pkg = core = None
        if pkg is not None and core is not None:
            seen.add((pkg, core))
        return (len(seen) or None), logical

    if sys.platform == "win32":  # pragma: no cover - Windows でのみ通る
        out = _run(["wmic", "cpu", "get", "NumberOfCores"])
        if out:
            nums = [int(x) for x in re.findall(r"\d+", out)]
            if nums:
                return sum(nums), logical
        return None, logical

    return None, logical


def is_unified_memory() -> bool:
    """Apple Silicon のように VRAM と RAM が同じ物理メモリか."""
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def detect() -> Hardware:
    """このマシンを一度だけ調べる。**失敗しても例外は出さない**."""
    gpus = detect_gpus()
    physical, logical = detect_cores()
    return Hardware(gpus=gpus, ram_gib=detect_ram_gib(),
                    physical_cores=physical, logical_cores=logical,
                    unified_memory=is_unified_memory() and not gpus)


def render(hw: Hardware) -> str:
    """検出結果を人間に見せる (``--show-config`` 用)."""
    lines = ["detected hardware"]
    if hw.gpus:
        for g in hw.gpus:
            lines.append(f"  GPU {g.index}      {g.name}  "
                         f"{g.total_gib:.1f} GiB ({g.used_gib:.1f} used)")
    elif hw.unified_memory:
        lines.append("  GPU         none; Apple Silicon unified memory")
    else:
        lines.append("  GPU         not detected (no nvidia-smi)")
    lines.append(f"  RAM         {hw.ram_gib:.1f} GiB" if hw.ram_gib
                 else "  RAM         not detected")
    cores = "  CPU         "
    if hw.physical_cores:
        cores += f"{hw.physical_cores} physical"
        if hw.logical_cores and hw.logical_cores != hw.physical_cores:
            cores += f" / {hw.logical_cores} logical"
    elif hw.logical_cores:
        cores += f"{hw.logical_cores} logical (physical count unavailable)"
    else:
        cores += "not detected"
    lines.append(cores)
    return "\n".join(lines)


#: 指定値と実測がこの割合を超えて食い違ったら警告する
VRAM_MISMATCH_TOLERANCE = 0.10


def vram_disagrees(given: float | None, hw: Hardware) -> bool:
    """指定された VRAM が、このマシンの実測と食い違っているか.

    設定ファイルを別のマシンに持っていったときに気づけるようにするための照合。
    実際に、Mac で書いた ``vram = 48.0`` (統合メモリ 64GiB の 75%) を
    NVIDIA 2枚の Linux 機で読み、**載らない ctx を勧める**事故を踏んだ。

    実測が取れないときは何も言わない (判断材料が無いのに騒がない)。
    """
    detected = hw.suggested_vram_gib()
    if given is None or detected is None:
        return False
    return abs(given - detected) > detected * VRAM_MISMATCH_TOLERANCE
