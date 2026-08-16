"""較正のテスト.

実測4点 (Qwen3.8-27B-Q5_K_M / RTX 5090 / CUDA / draft-mtp あり) を
そのまま使う。この4点は誤差 0 MiB で1本の直線に乗ることを確認済み。

**数字はすべて起動前からの増分 (MiB)。**素の nvidia-smi の絶対値ではない
(このマシンは待機時に 4 MiB 使っていた)。
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from gguf_fit import calibrate as cal

#: 実測 (gguf-calibrate、条件をそろえたもの)
F16 = [cal.Point("f16", 32768, 21712), cal.Point("f16", 65536, 23922)]
Q8 = [cal.Point("q8_0", 32768, 20838), cal.Point("q8_0", 65536, 22216)]

#: 同じサーバ・同じ ctx を**推論中に**読んだ値 (絶対値から待機分 4 MiB を引いた)。
#: ロード直後より 142 MiB 高い。f16 でも q8_0 でも同じ 142。
F16_RUNNING_64K = 24064
Q8_RUNNING_64K = 22358


# --- 直線の当てはめ --------------------------------------------------------

def test_two_points_reproduce_the_real_measurements():
    f = cal.fit_points(F16)
    assert f.bytes_per_token == pytest.approx(70720, abs=1)      # 69.1 KB/token
    assert f.max_error_mib == pytest.approx(0, abs=0.5)
    for p in F16:
        assert f.predict_gib(p.ctx) * 1024 == pytest.approx(p.used_mib, abs=1)


def test_the_measured_slope_lands_near_the_gguf_arithmetic():
    """GGUF から計算した 68.0 KB/token に対して 1.016 倍。

    ここが 1.08 倍に見えていたのは測り方のせいだった (下のテスト参照)。
    """
    theoretical = 69632          # 4 heads x (256+256) x 2 bytes x 17 layers
    assert cal.fit_points(F16).bytes_per_token / theoretical == pytest.approx(
        1.016, abs=0.005)


def test_q8_slope_is_not_the_theoretical_one():
    """理論 (34/64 バイト) なら 36.1 KB/token のはずが、実測は 43.1."""
    f = cal.fit_points(Q8)
    assert f.bytes_per_token == pytest.approx(44096, abs=1)
    theoretical = 69632 * 0.53125
    assert f.bytes_per_token > theoretical * 1.15


def test_measured_ratio_differs_from_the_naive_figure():
    """0.531 ではなく 0.624。ここを理論値のままにすると節約を過大に見積もる."""
    ratio = cal.fit_points(Q8).bytes_per_token / cal.fit_points(F16).bytes_per_token
    assert ratio == pytest.approx(0.624, abs=0.005)
    assert ratio > 0.531


def test_intercepts_agree_between_the_two_kv_modes():
    """切片はモデル本体 + 固定バッファ。KV の型では大きく変わらないはず."""
    a, b = cal.fit_points(F16).intercept_gib, cal.fit_points(Q8).intercept_gib
    assert abs(a - b) < 0.1


def test_mixing_load_time_and_running_points_inflates_the_slope():
    """**一度これで間違えた。**条件をそろえないと、切片に乗るべき差が傾きに化ける.

    ctx 32,768 をロード直後に、ctx 65,536 を推論中に測ると 73.5 KB/token に
    なる。差の 142 MiB は ctx に比例しないので、傾きに入れてはいけない。
    """
    mixed = [F16[0], cal.Point("f16", 65536, F16_RUNNING_64K)]
    inflated = cal.fit_points(mixed).bytes_per_token
    assert inflated == pytest.approx(75264, abs=64)        # 73.5 KB/token
    assert inflated > cal.fit_points(F16).bytes_per_token * 1.06


def test_the_running_offset_does_not_depend_on_the_kv_type():
    """+142 MiB は f16 でも q8_0 でも同じ。だから KV ではなく推論の分."""
    assert F16_RUNNING_64K - F16[1].used_mib == Q8_RUNNING_64K - Q8[1].used_mib


def test_one_point_is_refused():
    """1点では傾きと切片を分離できない。実際にそれで判断できず詰まった."""
    with pytest.raises(ValueError, match="two different"):
        cal.fit_points([F16[0]])


def test_repeated_ctx_is_refused():
    same = [cal.Point("f16", 32768, 21712), cal.Point("f16", 32768, 21716)]
    with pytest.raises(ValueError, match="two different"):
        cal.fit_points(same)


def test_mixed_kv_modes_are_refused():
    with pytest.raises(ValueError, match="one kv_mode"):
        cal.fit_points([F16[0], Q8[0]])


def test_three_points_use_least_squares():
    pts = [*F16, cal.Point("f16", 49152, 22817)]     # ちょうど中間
    f = cal.fit_points(pts)
    assert f.n_points == 3
    assert f.max_error_mib < 5


def test_noise_shows_up_as_error_not_as_a_silent_fit():
    """当てはまりの悪さを黙って飲み込まない。max_error で見えるようにする."""
    pts = [*F16, cal.Point("f16", 49152, 23500)]     # 明らかに外れた点
    f = cal.fit_points(pts)
    assert f.max_error_mib > 100


# --- ロード直後と推論後の差 ------------------------------------------------

def test_warmup_delta_is_reported_per_point():
    p = cal.Point("f16", 65536, F16_RUNNING_64K, 23922)
    assert p.warmup_mib == 142


def test_warmup_delta_is_none_when_it_was_not_measured():
    """測っていないものを 0 と言わない。**None は「測っていない」**."""
    assert cal.Point("f16", 65536, 23922).warmup_mib is None


def test_result_says_how_much_one_request_added():
    pts = [cal.Point("f16", 65536, F16_RUNNING_64K, 23922),
           cal.Point("q8_0", 65536, Q8_RUNNING_64K, 22216)]
    out = cal.render_fits([cal.fit_points(F16)], pts)
    assert "142 MiB" in out
    assert "one request" in out


def test_result_omits_the_warmup_line_when_nothing_was_warmed_up():
    out = cal.render_fits([cal.fit_points(F16)], list(F16))
    assert "one request" not in out


# --- 一番増えた GPU を見る -------------------------------------------------

def test_the_gpu_that_grew_is_the_one_measured():
    """nvidia-smi の並びと CUDA の番号は一致しない。「増えたほう」で見る."""
    before = {0: 279, 1: 4}
    after = {0: 279, 1: 21716}
    assert cal._biggest_delta(before, after) == 21712


def test_other_processes_do_not_leak_into_the_figure():
    """差分で測るので、デスクトップや他プロセスの使用量は混ざらない."""
    before = {0: 279, 1: 4}
    after = {0: 512, 1: 20842}      # GPU0 も増えたが、GPU1 のほうが大きい
    assert cal._biggest_delta(before, after) == 20838


def test_no_nvidia_smi_gives_zero():
    """測れないときに負の値やゴミを返さない。0 = 測れなかった."""
    assert cal._biggest_delta({}, {}) == 0
    assert cal._biggest_delta({0: 10}, {}) == 0
    assert cal._biggest_delta({}, {0: 10}) == 0


# --- 起動完了の判定 --------------------------------------------------------

class _FakeProc:
    """まだ生きているサーバ."""

    returncode = None

    def poll(self):
        return None


class _DeadProc:
    returncode = 1

    def poll(self):
        return 1


def test_ready_is_the_servers_own_answer_not_a_quiet_vram_reading(monkeypatch):
    """/health が返るまで待つ。**VRAM が静かになったことを完了と見なさない**.

    大きい ctx ほど確保に間があくので、静かさで判定すると途中の値を拾う。
    """
    seen = []
    calls = {"n": 0}

    def fake_ok(url, timeout=2.0):
        calls["n"] += 1
        seen.append(url)
        return calls["n"] >= 3

    monkeypatch.setattr(cal, "_http_ok", fake_ok)
    monkeypatch.setattr(cal.time, "sleep", lambda s: None)
    took = cal.wait_until_ready(8085, _FakeProc())
    assert took == pytest.approx(2 * cal.POLL_S)
    assert seen[0] == "http://127.0.0.1:8085/health"


def test_a_server_that_died_is_reported_not_waited_out(monkeypatch):
    monkeypatch.setattr(cal, "_http_ok", lambda *a, **k: False)
    monkeypatch.setattr(cal.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="exited early"):
        cal.wait_until_ready(8085, _DeadProc())


def test_waiting_forever_is_refused(monkeypatch):
    monkeypatch.setattr(cal, "_http_ok", lambda *a, **k: False)
    monkeypatch.setattr(cal.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="/health"):
        cal.wait_until_ready(8085, _FakeProc(), timeout_s=4)


# --- ウォームアップ --------------------------------------------------------

class _Resp:
    status = 200

    def read(self):
        return b"{}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_warmup_posts_a_tiny_completion(monkeypatch):
    sent = {}

    def fake_open(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(cal.urllib.request, "urlopen", fake_open)
    assert cal.warm_up(8085) is True
    assert sent["url"] == "http://127.0.0.1:8085/completion"
    assert sent["body"]["n_predict"] == cal.WARMUP_TOKENS


def test_warmup_falls_back_to_the_openai_route(monkeypatch):
    tried = []

    def fake_open(req, timeout=None):
        tried.append(req.full_url)
        if req.full_url.endswith("/completion"):
            raise urllib.error.HTTPError(req.full_url, 404, "no", None, None)
        return _Resp()

    monkeypatch.setattr(cal.urllib.request, "urlopen", fake_open)
    assert cal.warm_up(8085) is True
    assert tried[-1].endswith("/v1/completions")


def test_a_failed_warmup_is_reported_not_hidden(monkeypatch):
    """**投げられなかったのに投げた顔をしない。**呼び側がロード直後の値に戻す."""
    def fake_open(req, timeout=None):
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(cal.urllib.request, "urlopen", fake_open)
    assert cal.warm_up(8085) is False


# --- 出力 ------------------------------------------------------------------

def test_result_states_the_measured_ratio():
    out = cal.render_fits([cal.fit_points(F16), cal.fit_points(Q8)])
    assert "69.1 KB/token" in out
    assert "43.1 KB/token" in out
    assert "0.624" in out
    assert "0.531" in out          # 理論値との対比を残す


def test_toml_fragment_is_valid_and_says_it_was_measured():
    import tomllib  # noqa: PLC0415

    text = cal.render_toml_fragment([cal.fit_points(F16), cal.fit_points(Q8)])
    parsed = tomllib.loads(text)
    assert parsed["kv_f16_bytes"] == pytest.approx(70720, abs=1)
    assert parsed["kv_q8_bytes"] == pytest.approx(44096, abs=1)
    assert "Measured after one request" in text


def test_prediction_matches_a_held_out_expectation():
    """較正した式で、測っていない ctx を予測する."""
    f = cal.fit_points(Q8)
    # 24GiB カードに ctx 131,072 は入らない (以前 23.97 と誤って書いた)
    assert f.predict_gib(131072) == pytest.approx(24.39, abs=0.05)
    assert f.predict_gib(131072) > 24.0


# --- 起動コマンド ----------------------------------------------------------

def test_launch_command_matches_what_gguf_plan_emits():
    cmd = cal.build_launch_cmd("llama-server", "/m.gguf", 65536, "q8_0",
                               "CUDA0", 8, 18085, ["--spec-type", "draft-mtp"])
    assert cmd[:3] == ["llama-server", "-m", "/m.gguf"]
    assert "-ctk" in cmd and cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("--ctx-size") + 1] == "65536"
    assert cmd[-2:] == ["--spec-type", "draft-mtp"]


def test_f16_does_not_pass_a_kv_type_flag():
    """f16 は既定。**指定しないことが条件**なので、足してはいけない."""
    cmd = cal.build_launch_cmd("llama-server", "/m.gguf", 32768, "f16",
                               None, None, 18085)
    assert "-ctk" not in cmd
    assert "--device" not in cmd      # 分からないなら書かない
