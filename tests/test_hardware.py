"""ハードウェア検出のテスト.

**検出は必ず失敗しうる**。nvidia-smi が無い、権限が無い、出力形式が変わる。
そのたびにツールごと落ちてはいけないので、「取れなかった」を返せることを固定する。

実機に依存しないよう、外部コマンドは monkeypatch で差し替える。
"""

from __future__ import annotations

import pytest

from gguf_fit import _hardware as hw

NVIDIA_TWO_GPUS = (
    "0, NVIDIA GeForce RTX 3090, 24576, 279\n"
    "1, NVIDIA GeForce RTX 5090, 32607, 22362\n"
)


@pytest.fixture
def fake_run(monkeypatch):
    """``_run`` を辞書で差し替える。キーはコマンド名."""
    def _install(mapping):
        def _fake(cmd):
            return mapping.get(cmd[0])
        monkeypatch.setattr(hw, "_run", _fake)
    return _install


# --- GPU -----------------------------------------------------------------

def test_parses_nvidia_smi(fake_run):
    fake_run({"nvidia-smi": NVIDIA_TWO_GPUS})
    gpus = hw.detect_gpus()
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].name == "NVIDIA GeForce RTX 3090"
    assert gpus[0].total_gib == pytest.approx(24.0, abs=0.01)
    assert gpus[1].total_gib == pytest.approx(31.84, abs=0.01)
    assert gpus[1].used_gib == pytest.approx(21.84, abs=0.01)


def test_no_nvidia_smi_is_not_an_error(fake_run):
    """nvidia-smi が無いのは普通のこと。落ちてはいけない."""
    fake_run({})
    assert hw.detect_gpus() == []


def test_garbage_lines_are_skipped(fake_run):
    fake_run({"nvidia-smi": "0, GPU, 24576, 279\nnot,a,valid,row\nshort,row\n"})
    gpus = hw.detect_gpus()
    assert len(gpus) == 1


def test_largest_gpu_is_used_for_the_budget():
    """複数刺さっているとき、モデルを載せるのは一番大きいカード."""
    h = hw.Hardware(gpus=[hw.Gpu(0, "3090", 24.0, 0.3), hw.Gpu(1, "5090", 31.8, 0.0)],
                    ram_gib=64.0, physical_cores=16, logical_cores=32,
                    unified_memory=False)
    assert h.largest_gpu.name == "5090"
    assert h.suggested_vram_gib() == 31.8


def test_no_gpu_and_no_unified_memory_gives_no_budget():
    """勝手に RAM を VRAM 扱いしない。ここは黙って外すより None が正しい."""
    h = hw.Hardware(gpus=[], ram_gib=64.0, physical_cores=8, logical_cores=16,
                    unified_memory=False)
    assert h.suggested_vram_gib() is None


# --- Apple Silicon の統合メモリ ------------------------------------------

def test_unified_memory_reserves_a_share_for_the_system():
    """macOS は GPU に回せる量に上限がある。全部使える前提だと必ず外す."""
    h = hw.Hardware(gpus=[], ram_gib=64.0, physical_cores=12, logical_cores=12,
                    unified_memory=True)
    assert h.suggested_vram_gib() == pytest.approx(48.0, abs=0.1)
    assert h.suggested_vram_gib() < h.ram_gib


def test_unified_memory_detection_requires_arm_mac(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert hw.is_unified_memory() is False
    monkeypatch.setattr(hw.sys, "platform", "darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: "x86_64")
    assert hw.is_unified_memory() is False
    monkeypatch.setattr(hw.platform, "machine", lambda: "arm64")
    assert hw.is_unified_memory() is True


def test_a_discrete_gpu_wins_over_unified_memory(monkeypatch, fake_run):
    """eGPU などが見えているなら、そちらが正しい予算."""
    fake_run({"nvidia-smi": NVIDIA_TWO_GPUS})
    monkeypatch.setattr(hw, "detect_ram_gib", lambda: 64.0)
    monkeypatch.setattr(hw, "is_unified_memory", lambda: True)
    monkeypatch.setattr(hw, "detect_cores", lambda: (16, 32))
    h = hw.detect()
    assert h.unified_memory is False
    assert h.suggested_vram_gib() == pytest.approx(31.84, abs=0.01)


# --- CPU -----------------------------------------------------------------

def test_threads_prefers_physical_cores():
    """llama.cpp の --threads は論理コアに合わせると遅くなることがある."""
    h = hw.Hardware(gpus=[], ram_gib=None, physical_cores=16, logical_cores=32,
                    unified_memory=False)
    assert h.suggested_threads() == 16


def test_threads_falls_back_to_logical_cores():
    h = hw.Hardware(gpus=[], ram_gib=None, physical_cores=None, logical_cores=8,
                    unified_memory=False)
    assert h.suggested_threads() == 8


def test_threads_can_be_unknown():
    h = hw.Hardware(gpus=[], ram_gib=None, physical_cores=None, logical_cores=None,
                    unified_memory=False)
    assert h.suggested_threads() is None


def test_linux_physical_cores_count_sockets_too(monkeypatch, tmp_path):
    """`cpu cores` だけ見るとソケット数を掛け忘れる。(physical id, core id) で数える."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "physical id\t: 0\ncore id\t\t: 0\n\n"
        "physical id\t: 0\ncore id\t\t: 0\n\n"   # SMT の相方 → 数えない
        "physical id\t: 0\ncore id\t\t: 1\n\n"
        "physical id\t: 1\ncore id\t\t: 0\n\n"   # 2つ目のソケット → 別物
        "physical id\t: 1\ncore id\t\t: 1\n",    # 末尾に空行が無い場合
        encoding="utf-8")
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 8)
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(cpuinfo if p == "/proc/cpuinfo"
                                                     else p, *a, **k))
    physical, logical = hw.detect_cores()
    assert physical == 4
    assert logical == 8


def test_unreadable_cpuinfo_is_not_fatal(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 8)

    def _boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr("builtins.open", _boom)
    assert hw.detect_cores() == (None, 8)


# --- 全体 ----------------------------------------------------------------

def test_detect_never_raises(monkeypatch):
    """何が失敗しても例外を投げない。検出のためにツールが死んでは本末転倒."""
    def _boom(*a, **k):
        raise OSError("everything is broken")

    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    monkeypatch.setattr(hw.os, "sysconf", _boom)
    h = hw.detect()
    assert h.gpus == []
    assert h.suggested_vram_gib() is None


def test_render_is_readable_with_gpus():
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 5090", 31.8, 21.8)], ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    out = hw.render(h)
    assert "RTX 5090" in out
    assert "31.8 GiB" in out
    assert "16 physical / 32 logical" in out


def test_render_says_so_when_nothing_was_detected():
    h = hw.Hardware(gpus=[], ram_gib=None, physical_cores=None, logical_cores=None,
                    unified_memory=False)
    out = hw.render(h)
    assert "not detected" in out


# --- --device の既定 ------------------------------------------------------

def test_no_nvidia_means_no_device_flag():
    """NVIDIA が無いのに --device CUDA0 を書くと、そのコマンドは起動しない.

    Apple Silicon で実際に踏んだ。Metal は llama.cpp が自分で選ぶので、
    分からないときは黙って省略するのが正しい。
    """
    h = hw.Hardware(gpus=[], ram_gib=64.0, physical_cores=16, logical_cores=16,
                    unified_memory=True)
    assert h.suggested_device() is None


def test_nvidia_present_gives_cuda0():
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 5090", 31.8, 0.0)], ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    assert h.suggested_device() == "CUDA0"
    assert h.device_index_is_ambiguous() is False


def test_multiple_gpus_are_flagged_as_ambiguous():
    """nvidia-smi は PCI 順、CUDA は既定で性能順。番号は一致しない."""
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 3090", 24.0, 0.3),
                          hw.Gpu(1, "RTX 5090", 31.8, 0.0)],
                    ram_gib=31.0, physical_cores=16, logical_cores=32,
                    unified_memory=False)
    assert h.device_index_is_ambiguous() is True
