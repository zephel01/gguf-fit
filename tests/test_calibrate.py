"""較正のテスト.

実測4点 (Qwen3.8-27B-Q5_K_M / RTX 5090 / CUDA / draft-mtp あり) を
そのまま使う。この4点は誤差 0 MiB で1本の直線に乗ることを確認済み。
"""

from __future__ import annotations

import pytest

from gguf_fit import calibrate as cal

#: 実測値 (MiB)。起動前後の差ではなく nvidia-smi の絶対値だが、
#: デスクトップ分 (279 MiB) は両方に乗るので傾きには影響しない。
F16 = [cal.Point("f16", 32768, 21716), cal.Point("f16", 65536, 24068)]
Q8 = [cal.Point("q8_0", 32768, 20842), cal.Point("q8_0", 65536, 22362)]


# --- 直線の当てはめ --------------------------------------------------------

def test_two_points_reproduce_the_real_measurements():
    f = cal.fit_points(F16)
    assert f.bytes_per_token == pytest.approx(75264, abs=1)      # 73.5 KB/token
    assert f.max_error_mib == pytest.approx(0, abs=0.5)
    for p in F16:
        assert f.predict_gib(p.ctx) * 1024 == pytest.approx(p.used_mib, abs=1)


def test_q8_slope_is_not_the_theoretical_one():
    """理論 (34/64 バイト) なら 36.1 KB/token のはずが、実測は 47.5."""
    f = cal.fit_points(Q8)
    assert f.bytes_per_token == pytest.approx(48640, abs=1)
    theoretical = 69632 * 0.53125
    assert f.bytes_per_token > theoretical * 1.2


def test_measured_ratio_differs_from_the_naive_figure():
    """0.531 ではなく 0.646。ここを理論値のままにすると節約を過大に見積もる."""
    ratio = cal.fit_points(Q8).bytes_per_token / cal.fit_points(F16).bytes_per_token
    assert ratio == pytest.approx(0.646, abs=0.005)
    assert ratio > 0.531


def test_intercepts_agree_between_the_two_kv_modes():
    """切片はモデル本体 + 固定バッファ。KV の型では大きく変わらないはず."""
    a, b = cal.fit_points(F16).intercept_gib, cal.fit_points(Q8).intercept_gib
    assert abs(a - b) < 0.1


def test_one_point_is_refused():
    """1点では傾きと切片を分離できない。実際にそれで判断できず詰まった."""
    with pytest.raises(ValueError, match="two different"):
        cal.fit_points([F16[0]])


def test_repeated_ctx_is_refused():
    same = [cal.Point("f16", 32768, 21716), cal.Point("f16", 32768, 21720)]
    with pytest.raises(ValueError, match="two different"):
        cal.fit_points(same)


def test_mixed_kv_modes_are_refused():
    with pytest.raises(ValueError, match="one kv_mode"):
        cal.fit_points([F16[0], Q8[0]])


def test_three_points_use_least_squares():
    pts = [*F16, cal.Point("f16", 49152, 22892)]     # ちょうど中間
    f = cal.fit_points(pts)
    assert f.n_points == 3
    assert f.max_error_mib < 5


def test_noise_shows_up_as_error_not_as_a_silent_fit():
    """当てはまりの悪さを黙って飲み込まない。max_error で見えるようにする."""
    pts = [*F16, cal.Point("f16", 49152, 23500)]     # 明らかに外れた点
    f = cal.fit_points(pts)
    assert f.max_error_mib > 100


# --- 一番増えた GPU を見る -------------------------------------------------

def test_the_gpu_that_grew_is_the_one_measured():
    """nvidia-smi の並びと CUDA の番号は一致しない。「増えたほう」で見る."""
    before = {0: 279, 1: 4}
    after = {0: 279, 1: 21716}
    assert cal._biggest_delta(before, after) == 21712


def test_other_processes_do_not_leak_into_the_figure():
    """差分で測るので、デスクトップや他プロセスの使用量は混ざらない."""
    before = {0: 279, 1: 4}
    after = {0: 512, 1: 20846}      # GPU0 も増えたが、GPU1 のほうが大きい
    assert cal._biggest_delta(before, after) == 20842


def test_no_nvidia_smi_gives_zero():
    """測れないときに負の値やゴミを返さない。0 = 測れなかった."""
    assert cal._biggest_delta({}, {}) == 0
    assert cal._biggest_delta({0: 10}, {}) == 0
    assert cal._biggest_delta({}, {0: 10}) == 0


# --- 出力 ------------------------------------------------------------------

def test_result_states_the_measured_ratio():
    out = cal.render_fits([cal.fit_points(F16), cal.fit_points(Q8)])
    assert "73.5 KB/token" in out
    assert "47.5 KB/token" in out
    assert "0.646" in out
    assert "0.531" in out          # 理論値との対比を残す


def test_toml_fragment_is_valid_and_says_it_was_measured():
    import tomllib  # noqa: PLC0415

    text = cal.render_toml_fragment([cal.fit_points(F16), cal.fit_points(Q8)])
    parsed = tomllib.loads(text)
    assert parsed["kv_f16_bytes"] == pytest.approx(75264, abs=1)
    assert parsed["kv_q8_bytes"] == pytest.approx(48640, abs=1)
    assert "Measured, not derived" in text


def test_prediction_matches_a_held_out_expectation():
    """較正した式で、測っていない ctx を予測する."""
    f = cal.fit_points(Q8)
    # 24GiB カードに ctx 131,072 は入らない (以前 23.97 と誤って書いた)
    assert f.predict_gib(131072) == pytest.approx(24.81, abs=0.05)
    assert f.predict_gib(131072) > 24.0
