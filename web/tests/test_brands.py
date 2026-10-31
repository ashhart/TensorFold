import base64

import pytest

from tensorfold_web import brands

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


def test_a_maker_is_kept_completed_and_removed(tmp_path):
    store = brands.Brands(tmp_path / "brands.json")
    kept = store.save("falcon", {"label": " Falcon ", "kind": "model", "words": ["Falcon", "falcon", " tii "], "logo": PNG})
    assert kept == {"id": "falcon", "label": "Falcon", "kind": "model", "words": ["falcon", "tii"], "logo": PNG}
    # renamed without a logo in the body: the logo stays; null takes it away
    assert store.save("falcon", {"label": "TII Falcon", "kind": "model", "words": ["falcon"]})["logo"] == PNG
    assert store.save("falcon", {"label": "TII Falcon", "kind": "model", "words": ["falcon"], "logo": None})["logo"] is None
    # a built-in maker completed with only its logo
    assert store.save("microsoft", {"label": "Microsoft", "kind": "model", "logo": PNG})["words"] == []
    assert [b["id"] for b in store.all()] == ["falcon", "microsoft"]
    assert store.remove("falcon") and not store.remove("falcon")
    assert [b["id"] for b in brands.Brands(tmp_path / "brands.json").all()] == ["microsoft"]


@pytest.mark.parametrize("brand_id, body", [
    ("Bad Id", {"label": "X", "kind": "model", "words": ["x"]}),
    ("x", {"label": "", "kind": "model", "words": ["x"]}),
    ("x", {"label": "X", "kind": "software", "words": ["x"]}),
    ("x", {"label": "X", "kind": "model", "words": []}),
    ("x", {"label": "X", "kind": "model", "words": ["x"], "logo": "https://example.com/x.png"}),
    ("x", {"label": "X", "kind": "model", "words": ["x"], "logo": "data:text/html;base64,PGI+"}),
    ("x", {"label": "X", "kind": "model", "words": ["x"],
           "logo": "data:image/svg+xml;base64," + base64.b64encode(b'<svg onload="alert(1)"/>').decode()}),
    ("x", {"label": "X", "kind": "model", "words": ["x"],
           "logo": "data:image/png;base64," + base64.b64encode(b"0" * (brands.LOGO_BYTES + 1)).decode()}),
])
def test_an_entry_that_cannot_be_kept_is_refused(tmp_path, brand_id, body):
    with pytest.raises(brands.Invalid):
        brands.Brands(tmp_path / "brands.json").save(brand_id, body)
