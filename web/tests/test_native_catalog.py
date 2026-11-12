import os

import pytest

from tensorfold_web import catalog, native


def test_every_curated_family_is_served_by_native():
    binary = os.environ.get("TENSORFOLD_WEB_TEST_NATIVE") or native.find_binary()
    if not binary:
        pytest.skip("native capabilities binary not available")
    caps = native.capabilities(binary)
    assert caps.get("families"), "capabilities --json must report actual native families"
    entries = catalog.Catalog().entries
    assert entries
    for entry in entries:
        assert entry.family in caps["families"], entry.id
        assert entry.native, entry.id
        assert "26b-a4b" not in entry.id.lower()
        assert "bonsai" not in entry.id.lower()
