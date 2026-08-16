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


# --- 指定値と実測の照合 ----------------------------------------------------

LINUX_2GPU = hw.Hardware(gpus=[hw.Gpu(0, "RTX 3090", 24.0, 0.3),
                               hw.Gpu(1, "RTX 5090", 31.8, 0.0)],
                         ram_gib=31.0, physical_cores=16, logical_cores=32,
                         unified_memory=False)


def test_config_from_another_machine_is_caught():
    """Mac で書いた vram = 48.0 を NVIDIA 機で読んだ実際の事故.

    そのままだと「載らない ctx」を勧めてしまう。
    """
    assert hw.vram_disagrees(48.0, LINUX_2GPU) is True


def test_matching_value_is_not_flagged():
    assert hw.vram_disagrees(31.8, LINUX_2GPU) is False


def test_small_difference_is_tolerated():
    """ドライバの取り分などで数百 MiB はずれる。そこで騒いでも意味がない."""
    assert hw.vram_disagrees(31.0, LINUX_2GPU) is False


def test_planning_for_a_smaller_card_is_flagged_too():
    """24GiB 機向けに計画するのは正当だが、意図的かどうかは本人しか知らない.

    だから「止める」のではなく「言う」。
    """
    assert hw.vram_disagrees(24.0, LINUX_2GPU) is True


def test_nothing_is_said_when_detection_failed():
    """判断材料が無いのに警告するとノイズになる."""
    blind = hw.Hardware(gpus=[], ram_gib=None, physical_cores=None,
                        logical_cores=None, unified_memory=False)
    assert hw.vram_disagrees(48.0, blind) is False


def test_nothing_is_said_when_no_value_was_given():
    assert hw.vram_disagrees(None, LINUX_2GPU) is False


def test_gpu_sizes_are_rounded_at_detection(fake_run):
    """32607 MiB / 1024 = 31.8427734375。この生値を持ち回らない."""
    fake_run({"nvidia-smi": NVIDIA_TWO_GPUS})
    gpus = hw.detect_gpus()
    assert gpus[1].total_gib == 31.84
    assert gpus[1].used_gib == 21.84
    assert len(str(gpus[1].total_gib)) <= 6


# --- llama-server --list-devices（第一候補の情報源）----------------------

#: 実機の出力そのまま (Strix Halo / ROCm ビルド)
LIST_DEVICES_ROCM = """Available devices:
  ROCm0: AMD Radeon 8060S Graphics (98304 MiB, 16642 MiB free)
"""
LIST_DEVICES_CUDA = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32607 MiB, 32100 MiB free)
  CUDA1: NVIDIA GeForce RTX 3090 (24576 MiB, 24297 MiB free)
"""


def test_parses_amd_rocm_devices(fake_run):
    """nvidia-smi では AMD が丸ごと見えない。--list-devices なら取れる."""
    fake_run({"llama-server": LIST_DEVICES_ROCM})
    gpus = hw.detect_llama_devices()
    assert len(gpus) == 1
    g = gpus[0]
    assert g.device_id == "ROCm0"
    assert g.name == "AMD Radeon 8060S Graphics"
    assert g.total_gib == 96.0
    assert g.free_gib == pytest.approx(16.25, abs=0.01)


def test_device_id_comes_from_llama_not_from_a_guess(fake_run):
    """--device に書く値を推測しなくてよくなる。これが一番の利点."""
    fake_run({"llama-server": LIST_DEVICES_ROCM})
    h = hw.Hardware(gpus=hw.detect_llama_devices(), ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    assert h.suggested_device() == "ROCm0"
    assert h.device_index_is_ambiguous() is False


def test_known_device_ids_remove_the_numbering_ambiguity(fake_run):
    """llama.cpp 自身の番号なので、複数枚でもずれようがない."""
    fake_run({"llama-server": LIST_DEVICES_CUDA})
    h = hw.Hardware(gpus=hw.detect_llama_devices(), ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    assert h.suggested_device() == "CUDA0"       # 一番大きいのは 5090
    assert h.largest_gpu.name.endswith("5090")
    assert h.device_index_is_ambiguous() is False


def test_llama_devices_win_over_nvidia_smi(monkeypatch, fake_run):
    fake_run({"llama-server": LIST_DEVICES_ROCM, "nvidia-smi": NVIDIA_TWO_GPUS})
    monkeypatch.setattr(hw, "detect_ram_gib", lambda: 31.0)
    monkeypatch.setattr(hw, "detect_cores", lambda: (16, 32))
    h = hw.detect()
    assert [g.device_id for g in h.gpus] == ["ROCm0"]


def test_falls_back_to_nvidia_smi_without_llama_server(monkeypatch, fake_run):
    """llama.cpp がパスに無い環境も普通にある."""
    fake_run({"nvidia-smi": NVIDIA_TWO_GPUS})
    monkeypatch.setattr(hw, "detect_ram_gib", lambda: 31.0)
    monkeypatch.setattr(hw, "detect_cores", lambda: (16, 32))
    h = hw.detect()
    assert len(h.gpus) == 2
    assert all(g.device_id is None for g in h.gpus)
    assert h.device_index_is_ambiguous() is True   # 番号を推測している状態


def test_noise_around_the_device_lines_is_ignored(fake_run):
    fake_run({"llama-server": "ggml_cuda_init: found 1 device\n"
                              "Available devices:\n"
                              "  CUDA0: RTX 5090 (32607 MiB, 32100 MiB free)\n"
                              "load_backend: loaded RPC backend\n"})
    assert [g.device_id for g in hw.detect_llama_devices()] == ["CUDA0"]


def test_missing_free_field_is_tolerated(fake_run):
    """llama.cpp のバージョンで書式が変わりうる。総量だけでも使う."""
    fake_run({"llama-server": "  Metal0: Apple M4 Max (49152 MiB)\n"})
    g = hw.detect_llama_devices()[0]
    assert g.device_id == "Metal0"
    assert g.total_gib == 48.0
    assert g.free_gib is None


# --- 総量と空きの乖離 ------------------------------------------------------

def test_runtime_and_driver_free_figures_are_reconciled(fake_run):
    """実機 (Strix Halo) で踏んだ、危うく逆の助言をしかけたケース.

        llama.cpp : 98304 MiB total / 16642 MiB free
        amdgpu_top: VRAM 482 / 98304 MiB used  (= 95.5 GiB 空いている)
                    GTT   58 /  15860 MiB used

    llama.cpp の「16642 free」は VRAM ではなく GTT 側の数字。ここで
    ランタイムを信じて --vram 16 を勧めると、96 GiB 使えるマシンを
    16 GiB に切り詰めさせることになる。
    """
    fake_run({"llama-server": LIST_DEVICES_ROCM})
    h = hw.Hardware(gpus=hw.detect_llama_devices(), ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False,
                    driver_free_gib=95.53)   # sysfs 実測
    # 「空きが少ない」とは言わない
    assert h.tight_on_free_memory() is None
    # 代わりに「食い違っている」とだけ言う
    disagree = h.free_figures_disagree()
    assert disagree is not None
    dev, driver_free = disagree
    assert dev.total_gib == 96.0
    assert dev.free_gib == pytest.approx(16.25, abs=0.01)
    assert driver_free == 95.53


def test_a_genuinely_full_card_is_still_flagged():
    """両方が「少ない」と言うなら、それは本当に少ない."""
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 5090", 31.8, 30.0,
                                 device_id="CUDA0", free_gib=1.8)],
                    ram_gib=31.0, physical_cores=16, logical_cores=32,
                    unified_memory=False, driver_free_gib=1.9)
    assert h.tight_on_free_memory() is not None
    assert h.free_figures_disagree() is None


def test_without_a_driver_figure_the_runtime_is_believed(fake_run):
    """裏取りが無ければランタイムの言い分を採る (保守的な側に倒す)."""
    fake_run({"llama-server": LIST_DEVICES_ROCM})
    h = hw.Hardware(gpus=hw.detect_llama_devices(), ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False,
                    driver_free_gib=None)
    assert h.tight_on_free_memory() is not None
    assert h.free_figures_disagree() is None


def test_a_mostly_idle_card_is_not_flagged(fake_run):
    fake_run({"llama-server": LIST_DEVICES_CUDA})
    h = hw.Hardware(gpus=hw.detect_llama_devices(), ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    assert h.tight_on_free_memory() is None


def test_nothing_is_flagged_without_a_free_figure():
    """nvidia-smi 経由だと空きが分からない。分からないものは言わない."""
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 5090", 31.8, 21.8)], ram_gib=31.0,
                    physical_cores=16, logical_cores=32, unified_memory=False)
    assert h.tight_on_free_memory() is None


# --- BIOS で切り出した VRAM ------------------------------------------------

def test_carve_out_is_recognised():
    """実機 (AI MAX+ 395): 実装 128GB を VRAM 96 + OS 31 に分けている.

    RAM 31.0 GiB だけ見ると「メモリの少ない機械」に見えるが、そうではない。
    """
    h = hw.Hardware(gpus=[hw.Gpu(0, "AMD Radeon 8060S Graphics", 96.0, 0.5,
                                 device_id="ROCm0", free_gib=16.25)],
                    ram_gib=31.0, physical_cores=16, logical_cores=32,
                    unified_memory=False, driver_free_gib=95.5)
    assert h.carved_out_from_system_memory() == 127.0
    assert "shared pool" in hw.render(h)


def test_a_discrete_card_is_not_treated_as_a_carve_out():
    """5090 (31.8) + システム 31.0 は切り出しではない (VRAM < RAM ではないが僅差)."""
    h = hw.Hardware(gpus=[hw.Gpu(0, "RTX 3090", 24.0, 0.3)],
                    ram_gib=64.0, physical_cores=16, logical_cores=32,
                    unified_memory=False)
    assert h.carved_out_from_system_memory() is None
    assert "shared pool" not in hw.render(h)


def test_apple_silicon_is_handled_by_its_own_path():
    """統合メモリは別の扱いなので、ここでは二重に言わない."""
    h = hw.Hardware(gpus=[], ram_gib=64.0, physical_cores=16, logical_cores=16,
                    unified_memory=True)
    assert h.carved_out_from_system_memory() is None


# --- 複数ビルドの走査 ------------------------------------------------------

def test_one_build_cannot_see_another_backend(fake_run, monkeypatch):
    """build-cuda の --list-devices に ROCm は出ない。1本だけだと見落とす."""
    calls = {"/b/build-cuda/bin/llama-server": LIST_DEVICES_CUDA,
             "/b/build-rocm/bin/llama-server": LIST_DEVICES_ROCM}
    monkeypatch.setattr(hw, "_run", lambda cmd: calls.get(cmd[0]))

    cuda_only = hw.detect_all_llama_devices(["/b/build-cuda/bin/llama-server"])
    assert [g.device_id for g in cuda_only] == ["CUDA0", "CUDA1"]   # ROCm が消える

    both = hw.detect_all_llama_devices(list(calls))
    assert [g.device_id for g in both] == ["CUDA0", "CUDA1", "ROCm0"]


def test_each_device_records_which_build_reported_it(monkeypatch):
    calls = {"/b/cuda": LIST_DEVICES_CUDA, "/b/rocm": LIST_DEVICES_ROCM}
    monkeypatch.setattr(hw, "_run", lambda cmd: calls.get(cmd[0]))
    gpus = hw.detect_all_llama_devices(["/b/cuda", "/b/rocm"])
    by_id = {g.device_id: g.reported_by for g in gpus}
    assert by_id["CUDA0"] == "/b/cuda"
    assert by_id["ROCm0"] == "/b/rocm"


def test_the_same_device_seen_twice_is_not_duplicated(monkeypatch):
    """Vulkan ビルドと ROCm ビルドが同じカードを見ることがある."""
    monkeypatch.setattr(hw, "_run", lambda cmd: LIST_DEVICES_CUDA)
    gpus = hw.detect_all_llama_devices(["/b/one", "/b/two", "/b/three"])
    assert len(gpus) == 2


def test_missing_binaries_are_skipped_quietly(monkeypatch):
    calls = {"/b/rocm": LIST_DEVICES_ROCM}
    monkeypatch.setattr(hw, "_run", lambda cmd: calls.get(cmd[0]))
    gpus = hw.detect_all_llama_devices(["/nope", "/b/rocm", "/also-nope"])
    assert [g.device_id for g in gpus] == ["ROCm0"]


def test_detect_accepts_a_single_string_too(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: LIST_DEVICES_ROCM)
    monkeypatch.setattr(hw, "detect_ram_gib", lambda: 31.0)
    monkeypatch.setattr(hw, "detect_cores", lambda: (16, 32))
    monkeypatch.setattr(hw, "detect_amd_driver_free_gib", lambda: None)
    assert [g.device_id for g in hw.detect("llama-server").gpus] == ["ROCm0"]


# --- 予算と起動デバイスの一致 ----------------------------------------------

MIXED = hw.Hardware(
    gpus=[hw.Gpu(0, "RTX 5090", 31.8, 0.4, device_id="CUDA0", free_gib=31.4),
          hw.Gpu(1, "RTX 3090", 23.6, 0.6, device_id="CUDA1", free_gib=23.0),
          hw.Gpu(0, "AMD Radeon 8060S", 96.0, 79.7, device_id="ROCm0",
                 free_gib=16.3)],
    ram_gib=31.0, physical_cores=16, logical_cores=32, unified_memory=False)


def test_budget_follows_the_chosen_device():
    """96 GiB を前提に 31.8 GiB のカードで起動したら当然落ちる.

    実機で起こりかけた: largest は ROCm0 (96) なのに device は CUDA0 (31.8)。
    """
    assert MIXED.suggested_vram_gib("CUDA0") == 31.8
    assert MIXED.suggested_vram_gib("CUDA1") == 23.6
    assert MIXED.suggested_vram_gib("ROCm0") == 96.0


def test_without_a_device_the_largest_is_used():
    assert MIXED.suggested_vram_gib() == 96.0
    assert MIXED.suggested_device() == "ROCm0"


def test_unknown_device_id_falls_back_to_largest():
    assert MIXED.suggested_vram_gib("Vulkan9") == 96.0


def test_mixed_backends_are_reported():
    """一番大きい 8060S は、5090 より遅い。容量だけで選ばせない."""
    assert hw.has_mixed_backends(MIXED) is True


def test_same_backend_is_not_reported_as_mixed():
    cuda_only = MIXED._replace(gpus=MIXED.gpus[:2])
    assert hw.has_mixed_backends(cuda_only) is False


def test_vram_check_uses_the_chosen_device():
    """CUDA0 (31.8) を選んでいるなら、96 との比較で騒がない."""
    assert hw.vram_disagrees(31.8, MIXED, "CUDA0") is False
    assert hw.vram_disagrees(96.0, MIXED, "CUDA0") is True
    assert hw.vram_disagrees(96.0, MIXED, "ROCm0") is False
