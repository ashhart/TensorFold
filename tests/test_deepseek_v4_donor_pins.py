"""The packaged native donor is immutable, portable and correctly attributed."""

from tensorfold.families.deepseek_v4.cuda.build import PIN, VENDOR, verify_sources


def test_packaged_donor_revision_hashes_and_license():
    manifest = verify_sources(VENDOR)
    assert manifest["revision"] == PIN
    assert len(manifest["sha256"]) >= 40
    assert "MIT License" in (VENDOR / "LICENSE").read_text()
    assert (VENDOR / "ds4.h").is_file()
    assert (VENDOR / "ds4_cuda.cu").is_file()
