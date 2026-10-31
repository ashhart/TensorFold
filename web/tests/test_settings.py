

def test_secret_files_are_private_from_the_first_byte(tmp_path, monkeypatch):
    import os
    import stat

    from tensorfold_web import settings as settings_mod

    created = []
    real_open = os.open

    def spy(path, flags, mode=0o777, *a, **k):
        created.append(mode)
        return real_open(path, flags, mode, *a, **k)

    monkeypatch.setattr(os, "open", spy)
    settings_mod.write_private(tmp_path / "s.json", '{"api_key": "k"}')
    assert created == [0o600]
    assert stat.S_IMODE((tmp_path / "s.json").stat().st_mode) == 0o600
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
