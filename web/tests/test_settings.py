import pytest

from tensorfold_web import settings


def test_secret_files_are_private_from_the_first_byte(tmp_path, monkeypatch):
    import os
    import stat

    created = []
    real_open = os.open

    def spy(path, flags, mode=0o777, *a, **k):
        created.append(mode)
        return real_open(path, flags, mode, *a, **k)

    monkeypatch.setattr(os, "open", spy)
    settings.write_private(tmp_path / "s.json", '{"api_key": "k"}')
    assert created == [0o600]
    assert stat.S_IMODE((tmp_path / "s.json").stat().st_mode) == 0o600
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("raw", ["[]", "null", '"old settings"', "17", "false", "{"])
def test_invalid_settings_root_uses_defaults_without_rewriting_file(tmp_path, monkeypatch, raw):
    config = tmp_path / "settings.json"
    config.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(settings, "path", lambda: config)

    assert settings.load() == settings.Settings()
    assert config.read_text(encoding="utf-8") == raw


def test_load_keeps_known_fields_from_a_settings_object(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    config.write_text('{"theme":"dark","unknown":true}', encoding="utf-8")
    monkeypatch.setattr(settings, "path", lambda: config)

    assert settings.load() == settings.Settings(theme="dark")
