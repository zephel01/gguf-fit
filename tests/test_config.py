"""設定の解決とメッセージカタログのテスト.

このツールの売りは「その値がどこから来たかを言えること」なので、
優先順位そのものをテストで固定する。
"""

from __future__ import annotations

import string

import pytest

from gguf_fit import _config, _messages

# --- メッセージカタログ --------------------------------------------------

def test_every_message_has_both_languages():
    """片方だけ書いて忘れるのを防ぐ。新しいメッセージを足したらここで落ちる."""
    missing = {k: sorted(set(_messages.LANGS) - set(v))
               for k, v in _messages.MESSAGES.items()
               if set(_messages.LANGS) - set(v)}
    assert not missing, f"訳が欠けています: {missing}"


def test_placeholders_match_across_languages():
    """en と ja で差し込み口が違うと、片方の言語だけ KeyError で落ちる."""
    bad = {}
    for key, entry in _messages.MESSAGES.items():
        fields = {
            lang: {f for _, f, _, _ in string.Formatter().parse(text) if f}
            for lang, text in entry.items()
        }
        if len(set(map(frozenset, fields.values()))) > 1:
            bad[key] = fields
    assert not bad, f"プレースホルダが言語間で不一致: {bad}"


def test_default_language_is_english():
    assert _messages.DEFAULT_LANG == "en"
    assert _messages.t("hdr_headroom", hr=1.0).startswith("headroom")


def test_unknown_key_is_visible_not_silent():
    """未知のキーで黙って空文字を返すと、抜けに気づけない."""
    assert "missing message" in _messages.t("no_such_key", "en")


def test_translation_actually_differs():
    en = _messages.t("hdr_estimate", "en", model=1, kv=2, overhead=3, used=6, vram=24)
    ja = _messages.t("hdr_estimate", "ja", model=1, kv=2, overhead=3, used=6, vram=24)
    assert en != ja
    assert "estimate" in en
    assert "見積り" in ja


# --- 言語の解決 ----------------------------------------------------------

def test_lang_falls_back_to_english(monkeypatch):
    monkeypatch.delenv("GGUF_FIT_LANG", raising=False)
    assert _config.resolve("lang", None, {}, "en").value == "en"


def test_lang_from_env(monkeypatch):
    monkeypatch.setenv("GGUF_FIT_LANG", "ja")
    r = _config.resolve("lang", None, {}, "en")
    assert (r.value, r.source) == ("ja", "env")


def test_cli_beats_env_and_config(monkeypatch):
    monkeypatch.setenv("GGUF_FIT_LANG", "ja")
    r = _config.resolve("lang", "en", {"lang": "ja"}, "en")
    assert (r.value, r.source) == ("en", "cli")


def test_env_beats_config(monkeypatch):
    monkeypatch.setenv("GGUF_FIT_LANG", "en")
    r = _config.resolve("lang", None, {"lang": "ja"}, "en")
    assert (r.value, r.source) == ("en", "env")


def test_config_beats_default(monkeypatch):
    monkeypatch.delenv("GGUF_FIT_LANG", raising=False)
    r = _config.resolve("lang", None, {"lang": "ja"}, "en")
    assert (r.value, r.source) == ("ja", "config")


def test_env_is_coerced_to_the_declared_type(monkeypatch):
    monkeypatch.setenv("GGUF_FIT_VRAM", "24")
    r = _config.resolve("vram", None, {})
    assert r.value == 24.0
    assert isinstance(r.value, float)


def test_bad_env_value_is_reported_and_skipped(monkeypatch, capsys):
    monkeypatch.setenv("GGUF_FIT_VRAM", "lots")
    r = _config.resolve("vram", None, {"vram": 16.0})
    assert (r.value, r.source) == (16.0, "config")   # 壊れた env は飛ばして次へ
    assert "not a float" in capsys.readouterr().err


# --- 設定ファイル --------------------------------------------------------

def write(tmp_path, text):
    p = tmp_path / "gguf-fit.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_reads_a_toml_file(tmp_path):
    p = write(tmp_path, 'lang = "ja"\nvram = 24.0\ndevice = "CUDA1"\n')
    cfg, path = _config.load_config(p)
    assert cfg == {"lang": "ja", "vram": 24.0, "device": "CUDA1"}
    assert path == p


def test_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nowhere"))
    monkeypatch.delenv("GGUF_FIT_CONFIG", raising=False)
    cfg, path = _config.load_config()
    assert cfg == {}
    assert path is None


def test_unknown_key_is_reported_not_silently_ignored(tmp_path, capsys):
    """設定が効いていないことに気づけないのが一番困る."""
    p = write(tmp_path, 'lang = "ja"\nvramm = 24.0\n')
    cfg, _ = _config.load_config(p)
    assert cfg == {"lang": "ja"}
    assert "unknown key" in capsys.readouterr().err


def test_wrong_type_is_reported(tmp_path, capsys):
    p = write(tmp_path, 'vram = "twenty four"\n')
    cfg, _ = _config.load_config(p)
    assert cfg == {}
    assert "should be float" in capsys.readouterr().err


def test_broken_toml_is_reported(tmp_path, capsys):
    p = write(tmp_path, 'lang = "ja"\nthis is not toml\n')
    cfg, path = _config.load_config(p)
    assert (cfg, path) == ({}, None)
    assert "could not read" in capsys.readouterr().err


def test_search_order_puts_cwd_before_home(monkeypatch):
    monkeypatch.delenv("GGUF_FIT_CONFIG", raising=False)
    paths = [str(p) for p in _config.config_search_paths()]
    assert paths[0] == "gguf-fit.toml"
    assert len(paths) >= 2


def test_explicit_path_wins_the_search(tmp_path, monkeypatch):
    monkeypatch.setenv("GGUF_FIT_CONFIG", str(tmp_path / "from-env.toml"))
    paths = _config.config_search_paths(tmp_path / "explicit.toml")
    assert paths[0].name == "explicit.toml"
    assert paths[1].name == "from-env.toml"


# --- --show-config -------------------------------------------------------

def test_show_config_names_the_source_of_each_value(tmp_path):
    resolved = {
        "lang": _config.Resolved("ja", "config"),
        "vram": _config.Resolved(24.0, "cli"),
        "port": _config.Resolved(None, "default"),
    }
    out = _config.render_show_config(resolved, tmp_path / "gguf-fit.toml")
    assert "lang" in out and "<- config" in out
    assert "24.0" in out and "<- cli" in out
    assert "(unset)" in out
    assert "gguf-fit.toml" in out


def test_show_config_lists_where_it_looked_when_nothing_found():
    out = _config.render_show_config({"lang": _config.Resolved("en", "default")}, None)
    assert "(none found)" in out
    assert "searched:" in out


@pytest.mark.parametrize("key", sorted(_config.KNOWN_KEYS))
def test_every_known_key_has_a_usable_type(key):
    assert callable(_config.KNOWN_KEYS[key])


# --- --write-config -------------------------------------------------------

def test_written_toml_can_be_read_back(tmp_path):
    """書いたものが自分で読めなければ意味がない。往復させて確かめる."""
    resolved = {
        "lang": _config.Resolved("ja", "config"),
        "vram": _config.Resolved(31.8, "detected"),
        "overhead": _config.Resolved(1.0, "default"),
        "port": _config.Resolved(8085, "default"),
        "device": _config.Resolved("CUDA0", "detected"),
        "threads": _config.Resolved(16, "detected"),
        "model_path": _config.Resolved(None, "default"),
    }
    text = _config.render_toml(resolved, "detected hardware\n  RAM  64.0 GiB")
    p = tmp_path / "gguf-fit.toml"
    p.write_text(text, encoding="utf-8")

    cfg, path = _config.load_config(p)
    assert path == p
    assert cfg == {"lang": "ja", "vram": 31.8, "overhead": 1.0,
                   "port": 8085, "device": "CUDA0", "threads": 16}


def test_written_toml_records_where_each_value_came_from(tmp_path):
    resolved = {"vram": _config.Resolved(31.8, "detected"),
                "port": _config.Resolved(8085, "default")}
    text = _config.render_toml(resolved)
    assert "vram = 31.8   # <- detected" in text
    assert "port = 8085   # <- default" in text


def test_unavailable_values_are_commented_out_not_written_as_none(tmp_path):
    """None をそのまま書くと TOML として壊れる。コメントにして残す."""
    text = _config.render_toml({"vram": _config.Resolved(None, "default")})
    assert "vram = ..." in text
    assert "None" not in text
    p = tmp_path / "gguf-fit.toml"
    p.write_text(text, encoding="utf-8")
    cfg, _ = _config.load_config(p)
    assert cfg == {}          # 読み返しても壊れない


def test_hardware_summary_is_embedded_as_comments(tmp_path):
    text = _config.render_toml({"vram": _config.Resolved(31.8, "detected")},
                               "detected hardware\n  GPU 0  RTX 5090  31.8 GiB")
    assert "# detected hardware" in text
    assert "#   GPU 0  RTX 5090  31.8 GiB" in text
    p = tmp_path / "gguf-fit.toml"
    p.write_text(text, encoding="utf-8")
    cfg, _ = _config.load_config(p)   # コメントなので読み込みを壊さない
    assert cfg == {"vram": 31.8}


def test_floats_are_rounded_before_writing(tmp_path):
    """MiB/1024 の生値 (31.8427734375) を設定ファイルに書かない."""
    text = _config.render_toml({"vram": _config.Resolved(31.8427734375, "detected")})
    assert "vram = 31.84   # <- detected" in text
    assert "31.8427" not in text
    p = tmp_path / "gguf-fit.toml"
    p.write_text(text, encoding="utf-8")
    cfg, _ = _config.load_config(p)
    assert cfg["vram"] == 31.84


# --- --refresh -------------------------------------------------------------

def test_refresh_drops_only_the_detectable_keys():
    """設定ファイルは実測より強いので、一度書いた vram は居座る.

    実機で踏んだ: nvidia-smi 由来の 31.84 が config に入っていて、
    llama.cpp が言う 31.4 に更新されなかった。
    """
    cfg = {"lang": "ja", "vram": 31.84, "threads": 16, "device": "CUDA0",
           "overhead": 1.0, "port": 8085}
    refreshed = _config.drop_detectable(cfg)
    assert refreshed == {"lang": "ja", "overhead": 1.0, "port": 8085}


def test_refreshed_config_lets_detection_win():
    cfg = _config.drop_detectable({"vram": 31.84})
    r = _config.resolve("vram", None, cfg, detected=31.4)
    assert (r.value, r.source) == (31.4, "detected")


def test_without_refresh_the_config_still_wins():
    r = _config.resolve("vram", None, {"vram": 31.84}, detected=31.4)
    assert (r.value, r.source) == (31.84, "config")


def test_cli_beats_refresh_too():
    cfg = _config.drop_detectable({"vram": 31.84})
    r = _config.resolve("vram", 24.0, cfg, detected=31.4)
    assert (r.value, r.source) == (24.0, "cli")


def test_byte_counts_are_shown_as_integers_but_gib_keeps_its_decimal():
    """**単位で決める。**24.0 GiB の .0 には意味があり、70720.0 バイトには無い."""
    out = _config.render_show_config({
        "vram": _config.Resolved(24.0, "cli"),
        "kv_f16_bytes": _config.Resolved(70720.0, "config"),
    }, None)
    assert "24.0" in out
    assert "70720 " in out
    assert "70720.0" not in out
