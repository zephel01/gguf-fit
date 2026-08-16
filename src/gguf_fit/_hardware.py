"""このマシンの GPU / メモリ / CPU を見て、既定値を自分で決める.

``--vram`` を毎回打たせない、``--threads`` を出せるようにする、が目的。
**検出は必ず失敗しうる**ので、どれも「取れなかった (None)」を返せるようにし、
呼び側が既定値や明示指定に落とせる形にしてある。

検出の手段:

  GPU     llama-server --list-devices が第一候補。**llama.cpp 自身が認識して
          いる値**なので、CUDA / ROCm / Vulkan / Metal のどれでも同じ形で
          取れるうえ、--device にそのまま書ける識別子が手に入る。
          見つからないときだけ nvidia-smi に落とす
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

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
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
    #: llama.cpp が使う識別子 ("CUDA0" / "ROCm0" / "Vulkan0" ...)。
    #: --list-devices から取れたときだけ入る
    device_id: str | None = None
    #: いま空いている量。--list-devices から取れたときだけ入る
    free_gib: float | None = None
    #: このデバイスを報告した llama-server のパス。
    #: **ビルドごとに見えるデバイスが違う** (build-cuda は CUDA しか見えない)
    reported_by: str | None = None


class Hardware(NamedTuple):
    gpus: list[Gpu]
    ram_gib: float | None
    physical_cores: int | None
    logical_cores: int | None
    unified_memory: bool  # Apple Silicon のように VRAM と RAM が同じか
    #: ドライバ (sysfs) が言う VRAM の空き。ランタイムの申告を裏取りするため。
    #: amdgpu 以外では None
    driver_free_gib: float | None = None

    @property
    def largest_gpu(self) -> Gpu | None:
        return max(self.gpus, key=lambda g: g.total_gib) if self.gpus else None

    def gpu_by_device_id(self, device_id: str | None) -> Gpu | None:
        """識別子 ("CUDA0" / "ROCm0") でデバイスを引く."""
        if not device_id:
            return None
        for g in self.gpus:
            if g.device_id == device_id:
                return g
        return None

    def suggested_vram_gib(self, device_id: str | None = None) -> float | None:
        """``--vram`` の既定として使える値。取れなければ None.

        ``device_id`` を渡すと**そのデバイスの容量**を返す。渡さなければ
        一番大きいデバイス。

        **起動するデバイスと予算を取るデバイスは一致していないといけない。**
        実機 (CUDA 5090 / 31.8 GiB と ROCm 8060S / 96 GiB が同居) で、
        largest が ROCm0 なのに device が CUDA0 という不整合が起こりかけた。
        96 GiB を前提に計画して 31.8 GiB のカードで起動すれば、当然落ちる。
        """
        picked = self.gpu_by_device_id(device_id)
        if picked is not None:
            return picked.total_gib
        if self.gpus:
            return self.largest_gpu.total_gib
        if self.unified_memory and self.ram_gib:
            return round(self.ram_gib * UNIFIED_MEMORY_SHARE, 1)
        return None

    def carved_out_from_system_memory(self) -> float | None:
        """VRAM がシステムメモリからの切り出しなら、合計の実装量 (GiB) を返す.

        AMD Strix Halo (AI MAX+ 395) の実機で確認した構成::

            実装 128 GB
              ├─ 96.0 GiB  BIOS で VRAM に割り当て
              └─ 31.0 GiB  OS から見えるぶん

        ``RAM 31.0 GiB`` だけを見ると「メモリの少ない機械」に見えるが、実際は
        96 GiB を GPU に渡した後の残り。**両者は同じ物理メモリを分けたもの**で、
        BIOS で割合を変えられる。この関係が見えないと誤解するので出す。

        判定は「VRAM がシステムメモリ以上」。専用メモリのカードでこうなる構成は
        現実にはほぼ無い (96 GiB のカードに 31 GiB のシステムメモリ、など)。
        """
        big = self.largest_gpu
        if big is None or self.ram_gib is None or self.unified_memory:
            return None
        if big.total_gib < self.ram_gib:
            return None
        return round(big.total_gib + self.ram_gib, 1)

    def suggested_threads(self) -> int | None:
        """``--threads`` の既定。物理コアを優先し、無ければ論理コア."""
        return self.physical_cores or self.logical_cores

    def suggested_device(self) -> str | None:
        """``--device`` の既定。**分からないときは None を返して省略させる**.

        ``--list-devices`` から取れていれば、**一番大きいデバイスの識別子**を
        そのまま返す。llama.cpp 自身が名乗っている名前なので推測が要らない。

        取れていない (nvidia-smi しか無い) ときは ``CUDA0`` に落とすが、
        **nvidia-smi の並び順と CUDA のデバイス番号は一致しない**
        (nvidia-smi は PCI 順、CUDA は既定で性能順)。番号を勝手に振ると
        静かに別のカードを掴むので、複数枚あるときは注意書きを出す。

        NVIDIA が見えないのに ``CUDA0`` を書くと、そのコマンドは起動しない。
        Apple Silicon は Metal が既定で選ばれるので ``--device`` は要らない。
        """
        big = self.largest_gpu
        if big is None:
            return None
        return big.device_id or "CUDA0"

    def device_index_is_ambiguous(self) -> bool:
        """番号を推測している状態か.

        ``--list-devices`` から識別子が取れていれば曖昧さは無い。
        nvidia-smi しか無くて複数枚あるときだけ、番号がずれうる。
        """
        if any(g.device_id for g in self.gpus):
            return False
        return len(self.gpus) > 1

    def free_figures_disagree(self) -> tuple[Gpu, float] | None:
        """ランタイムの「空き」がドライバの実測と食い違っていたら返す.

        実機 (Strix Halo / Radeon 8060S) で見た値::

            llama-server --list-devices : 98304 MiB total, 16642 MiB free
            amdgpu_top / sysfs         : VRAM 482 / 98304 MiB used
                                         GTT   58 /  15860 MiB used

        **VRAM は 96 GiB のうち 482 MiB しか使っていない。**
        llama.cpp の言う「16642 MiB free」は VRAM の空きではなく、GTT 側
        (15860 MiB) に対応する数字。APU で ``hipMemGetInfo()`` が返す値の癖。

        ここでランタイム側を信じて「--vram 16 にしろ」と言うと、96 GiB 使える
        マシンを 16 GiB に切り詰めさせることになる。**逆の助言になる。**

        なので「少ないほうを採る」ことはしない。**食い違っている事実だけを
        言って、判断は人に返す。**戻り値は (デバイス, ドライバ側の空き)。
        """
        big = self.largest_gpu
        if big is None or big.free_gib is None or self.driver_free_gib is None:
            return None
        # ランタイムの空きが、ドライバの実測より目立って小さいとき
        if big.free_gib < self.driver_free_gib * 0.8:
            return (big, self.driver_free_gib)
        return None

    def tight_on_free_memory(self) -> Gpu | None:
        """本当に空きが少ないデバイス。**裏取りが取れているときだけ言う**.

        ドライバ側でも空きが少ないと確認できた場合に限る。ランタイムの数字
        だけで判断すると、上の APU のケースで誤った助言をする。
        """
        big = self.largest_gpu
        if big is None or big.free_gib is None:
            return None
        if self.driver_free_gib is not None:
            # 裏が取れている。両方が「少ない」と言うときだけ警告する
            if self.driver_free_gib < big.total_gib * 0.5:
                return big
            return None
        # 裏が取れない場合は、ランタイムの言い分を採る (保守的)
        return big if big.free_gib < big.total_gib * 0.5 else None


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


#: `  ROCm0: AMD Radeon 8060S Graphics (98304 MiB, 16642 MiB free)`
_DEVICE_LINE = re.compile(
    r"^\s*(?P<id>[A-Za-z]+(?P<index>\d+)):\s*(?P<name>.+?)\s*"
    r"\((?P<total>\d+)\s*MiB(?:,\s*(?P<free>\d+)\s*MiB free)?\)\s*$")


def detect_llama_devices(binary: str = "llama-server") -> list[Gpu]:
    """``llama-server --list-devices`` から拾う。**これが第一候補**.

    llama.cpp 自身が認識している値なので、

      ・CUDA / ROCm / Vulkan / Metal のどれでも同じ形で取れる
      ・``--device`` にそのまま書ける識別子が手に入る (番号の推測が要らない)
      ・総量だけでなく**空き**も分かる

    実際の出力::

        Available devices:
          ROCm0: AMD Radeon 8060S Graphics (98304 MiB, 16642 MiB free)

    バイナリがビルドごとに別 (build-cuda / build-rocm) なので、
    パスは設定で差し替えられるようにしてある。
    """
    out = _run([binary, "--list-devices"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        total = round(int(m["total"]) / 1024, 2)
        free = round(int(m["free"]) / 1024, 2) if m["free"] else None
        gpus.append(Gpu(index=int(m["index"]), name=m["name"],
                        total_gib=total,
                        used_gib=round(total - free, 2) if free is not None else 0.0,
                        device_id=m["id"], free_gib=free))
    return gpus


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


def detect_amd_driver_free_gib() -> float | None:
    """amdgpu の sysfs から VRAM の空きを読む (GiB)。取れなければ None.

    ランタイム (llama.cpp / HIP) の申告を**裏取りする**ためだけに使う。
    APU では両者が食い違うことがあり、そこを黙って片方に寄せると事故る。
    """
    if not sys.platform.startswith("linux"):
        return None
    best = None
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        try:
            with open(path, encoding="utf-8") as fh:
                total = int(fh.read().strip())
            with open(path.replace("_total", "_used"), encoding="utf-8") as fh:
                used = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        free = round((total - used) / 1024 ** 3, 2)
        if best is None or free > best:
            best = free
    return best


def is_unified_memory() -> bool:
    """Apple Silicon のように VRAM と RAM が同じ物理メモリか."""
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def detect_all_llama_devices(binaries: Sequence[str]) -> list[Gpu]:
    """**複数のビルド**を順に叩いて、見えるデバイスを全部集める.

    llama.cpp はビルドごとにバックエンドが別なので、``build-cuda`` の
    ``--list-devices`` には CUDA しか出ず、ROCm や Vulkan のデバイスは
    **丸ごと見えない**。1本しか指定できないと、そのマシンに実在する GPU を
    見落とす。実機 (CUDA 2枚 + ROCm 1枚) で問題になった。

    同じデバイスが複数のビルドから見えることもあるので
    (識別子, 名前) で重複を除く。
    """
    seen: dict[tuple[str | None, str], Gpu] = {}
    for binary in binaries:
        for g in detect_llama_devices(binary):
            key = (g.device_id, g.name)
            if key not in seen:
                seen[key] = g._replace(reported_by=binary)
    return list(seen.values())


def detect(llama_server: str | Sequence[str] = "llama-server") -> Hardware:
    """このマシンを一度だけ調べる。**失敗しても例外は出さない**."""
    binaries = [llama_server] if isinstance(llama_server, str) else list(llama_server)
    # llama.cpp が名乗る値を優先。無ければ nvidia-smi に落とす。
    gpus = detect_all_llama_devices(binaries) or detect_gpus()
    physical, logical = detect_cores()
    return Hardware(gpus=gpus, ram_gib=detect_ram_gib(),
                    physical_cores=physical, logical_cores=logical,
                    unified_memory=is_unified_memory() and not gpus,
                    driver_free_gib=detect_amd_driver_free_gib())


def render(hw: Hardware) -> str:
    """検出結果を人間に見せる (``--show-config`` 用)."""
    lines = ["detected hardware"]
    if hw.gpus:
        for g in hw.gpus:
            label = g.device_id or f"GPU {g.index}"
            detail = (f"{g.free_gib:.1f} free" if g.free_gib is not None
                      else f"{g.used_gib:.1f} used")
            lines.append(f"  {label:<11} {g.name}  {g.total_gib:.1f} GiB ({detail})")
            if g.reported_by and len(hw.gpus) > 1:
                lines.append(f"              via {g.reported_by}")
    elif hw.unified_memory:
        lines.append("  GPU         none; Apple Silicon unified memory")
    else:
        lines.append("  GPU         not detected (no nvidia-smi)")
    lines.append(f"  RAM         {hw.ram_gib:.1f} GiB" if hw.ram_gib
                 else "  RAM         not detected")
    carved = hw.carved_out_from_system_memory()
    if carved is not None:
        lines.append(f"              (VRAM + RAM = {carved:.1f} GiB shared pool; "
                     f"the split is set in firmware)")
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


def has_mixed_backends(hw: Hardware) -> bool:
    """種類の違うバックエンドが同居しているか (CUDA と ROCm など).

    **一番大きい = 一番速い、ではない。**実機では 8060S (96 GiB, APU) が
    5090 (31.8 GiB) より大きいが、生成速度は比べるまでもない。容量だけで
    自動的に選ぶと遅いほうを勧めることになるので、そうと言う。
    """
    kinds = {g.device_id.rstrip("0123456789")
             for g in hw.gpus if g.device_id}
    return len(kinds) > 1


def vram_disagrees(given: float | None, hw: Hardware,
                   device_id: str | None = None) -> bool:
    """指定された VRAM が、このマシンの実測と食い違っているか.

    設定ファイルを別のマシンに持っていったときに気づけるようにするための照合。
    実際に、Mac で書いた ``vram = 48.0`` (統合メモリ 64GiB の 75%) を
    NVIDIA 2枚の Linux 機で読み、**載らない ctx を勧める**事故を踏んだ。

    実測が取れないときは何も言わない (判断材料が無いのに騒がない)。
    """
    detected = hw.suggested_vram_gib(device_id)
    if given is None or detected is None:
        return False
    return abs(given - detected) > detected * VRAM_MISMATCH_TOLERANCE
