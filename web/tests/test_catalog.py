from tensorfold_web import catalog, pull


def test_a_download_cut_halfway_is_not_installed(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    repo = tmp_path / "models--org--m"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc")
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "blobs").mkdir()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(repo / "snapshots" / "abc"))
    assert catalog.installed("org/m")
    (repo / "blobs" / "deadbeef.incomplete").write_bytes(b"x")              # a pull TensorFold Web stopped halfway
    assert not catalog.installed("org/m")


def test_a_snapshot_lighter_than_the_catalog_says_is_not_installed(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    repo = tmp_path / "models--org--m"
    snap = repo / "snapshots" / "abc"
    (repo / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "blobs" / "b1").write_bytes(b"x" * 1000)
    (snap / "model.safetensors").symlink_to(repo / "blobs" / "b1")
    (repo / "blobs" / "stale.incomplete").write_bytes(b"y")                 # an orphan of another revision
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    assert catalog.installed("org/m", expected_bytes=1000)                   # whole, the orphan does not count
    assert not catalog.installed("org/m", expected_bytes=2000)               # half of it came down
    (snap / "tokenizer.json").symlink_to(repo / "blobs" / "never")           # a link to a blob that never finished
    assert not catalog.installed("org/m", expected_bytes=1000)


def test_a_checkpoint_the_engine_refuses_is_not_asked_again_on_every_view(monkeypatch):
    from tensorfold_web import catalog, native

    asked = []
    monkeypatch.setattr(native, "find_binary", lambda: "/x/tensorfold")
    monkeypatch.setattr(native, "info", lambda binary, model: asked.append(model) or None)
    cat = catalog.Catalog()
    cat.refresh_sizes_async = lambda: None
    e = next(x for x in cat.entries if x.native)
    assert cat.engine_facts(e) is None and cat.engine_facts(e) is None
    assert asked == [e.id]                                      # once, not on every Models or Home
    cat._facts_missed[e.id] -= catalog.FACTS_RETRY_S           # the memo aged out: asked again
    assert cat.engine_facts(e) is None and asked == [e.id, e.id]


def test_a_mixed_width_checkpoint_is_native_from_the_release_that_reads_it(monkeypatch):
    """`capabilities --json` lists the 27B's uniform format only: an oQ checkpoint counts as native from 1.0.10."""

    from tensorfold_web import catalog, native

    caps = {"families": {"qwen3_5": ["mlx-q4g64"]}}
    version = {"v": "1.0.9"}
    monkeypatch.setattr(native, "find_binary", lambda: "/x/tensorfold")
    monkeypatch.setattr(native, "version_of", lambda binary: version["v"])
    monkeypatch.setattr(native, "capabilities", lambda binary: caps)
    monkeypatch.setattr(native, "refused", lambda binary, model: None)
    cat = catalog.Catalog()
    cat.refresh_sizes_async = lambda: None
    oq2 = next(x for x in cat.entries if x.id == "TensorFold/Qwen3.8-27B-oQ2")
    assert oq2.native_from == "1.0.10" and not cat.native_here(oq2)
    version["v"] = "1.0.11"
    assert cat.native_here(oq2)
    monkeypatch.setattr(native, "refused", lambda binary, model: "QwenMetalQualificationRequired")   # an M6
    assert not cat.native_here(oq2)
