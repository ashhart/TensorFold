"""The 27B's startup memory plan on any GPU: exact bytes for the weights and their load, sized from the budget."""

import pytest

from tensorfold.cuda import capacity, geometry
from tensorfold.families.qwen3_5.cuda.affine_memory import draft_bytes, draft_weights, weight_transform

from cuda_27b_headers import TEXT, drafter, target

GIB = capacity.GIB
HEAD = 248320 * 640 * 4 + 2 * 248320 * 80 * 2          # the 4-bit head: words, scales and biases


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("27b")
    return target(root / "target"), drafter(root / "drafter")


def test_one_gpu_counts_the_tiled_head_once(model):
    folder, _ = model
    twice = capacity.estimate_weights(folder, weight_transform(folder))
    once = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    assert twice.resident - once.resident == HEAD                  # the drafter reads the head's rows as views
    assert round(twice.resident / GIB, 2) == 14.76                  # the real checkpoint's figure, measured 14.20 held
    assert once.staging == 3 * 248320 * 640 * 4                     # the largest tensor as loaded: the head's words


def test_one_gpu_counts_an_8bit_head_once(tmp_path):
    """An 8-bit head in groups of 64 is tiled too, so the drafter's rows are views of it as well."""

    from cuda_27b_headers import _write

    n, k = 248320, TEXT["hidden_size"]
    head = [("lm_head.weight", "U32", [n, k // 4]), ("lm_head.scales", "BF16", [n, k // 64]),
            ("lm_head.biases", "BF16", [n, k // 64])]
    folder = _write(tmp_path / "head8", {"model_type": "qwen3_5", "text_config": TEXT,
                                         "quantization": {"group_size": 64, "bits": 8}}, head)
    twice = capacity.estimate_weights(folder, weight_transform(folder))
    once = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    assert twice.resident - once.resident == n * k + 2 * n * (k // 64) * 2



@pytest.mark.parametrize("bits", [5, 6])
def test_one_gpu_counts_a_5_or_6bit_head_once(tmp_path, bits):
    """5- and 6-bit heads in groups of 64 are tiled too, so the drafter's rows are views of them as well."""

    from cuda_27b_headers import _write

    n, k = 248320, TEXT["hidden_size"]
    head = [("lm_head.weight", "U32", [n, k * bits // 32]), ("lm_head.scales", "BF16", [n, k // 64]),
            ("lm_head.biases", "BF16", [n, k // 64])]
    folder = _write(tmp_path / f"head{bits}", {"model_type": "qwen3_5", "text_config": TEXT,
                                                "quantization": {"group_size": 64, "bits": bits}}, head)
    twice = capacity.estimate_weights(folder, weight_transform(folder))
    once = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    assert twice.resident - once.resident == n * k * bits // 8 + 2 * n * (k // 64) * 2

def test_the_drafter_load_peak_is_its_largest_quantize(model):
    _, folder = model
    held = capacity.estimate_weights(folder, draft_bytes)
    peak = draft_weights(folder)
    assert peak.resident == held.resident and round(held.resident / GIB, 2) == 0.97
    assert peak.staging == 14 * 5120 * 25600                        # fc: bf16 upload, fp32 copy, two fp32 temporaries


def test_the_drafter_loads_after_the_target(model, monkeypatch):
    folder, draft = model
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 100 * GIB)
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)
    geometry = capacity.Geometry(lambda slots: slots * 1024, 12)
    main = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    side = draft_weights(draft)
    receipt = capacity.admit(folder, 8192, True, None, geometry, weight_transform(folder, one_gpu=True),
                             draft_dir=draft, draft_weights=draft_weights)
    assert receipt["weight_bytes_estimate"] == main.resident + side.resident
    assert receipt["loading_bytes_estimate"] == max(main.staging - side.resident, side.staging)
    # measured on an RTX 4090: 16.73 GiB allocated at the load's peak (17.95 reserved: the budget's reserve covers it)
    assert round((receipt["weight_bytes_estimate"] + receipt["loading_bytes_estimate"]) / GIB, 2) == 16.78


@pytest.mark.parametrize("gib,rows", [(128, 4096), (80, 4096), (23.54, 2048), (16, 1024), (12, 1024), (8, 512)])
def test_a_prompt_chunk_is_sized_to_the_card(gib, rows):
    assert geometry.prompt_rows(int(gib * GIB), geometry.prompt_row_bytes(TEXT)) == rows


@pytest.mark.parametrize("slots", [0, 4108, 65536 + 128, 262144 + 128])
def test_a_gb10_keeps_todays_scratch(slots):
    """One stream on a GB10 verifies 128 rows and prompts 4096: the scratch and every cache size are today's."""

    gb10 = geometry.gdn_geometry(TEXT, 1, 128, rows=128, prompt=geometry.prompt_rows(128 * GIB, 360448))
    assert gb10.bytes_at(slots) == geometry.gdn_geometry(TEXT, 1, 128).bytes_at(slots)
    assert gb10.bytes_at(262144 + 128) == 74040369152


def test_twelve_rows_bound_the_scratch_and_a_prompt_chunk_shares_it():
    extent = 5120 + 248320 + 2 * (17408 + 5120) + 16480 + 24 * 256
    verify = geometry.gdn_geometry(TEXT, 1, 12, rows=12)
    prompt = geometry.gdn_geometry(TEXT, 1, 12, rows=12, prompt=2048)
    wide = geometry.gdn_geometry(TEXT, 1, 12, rows=128)
    replay = 48 * (128 - 12) * (16480 * 2 + 16 * 128 * 4 + 48 * 128 * 4 + 48 * 8) + 32 * (128 - 12) * 2560 * 4
    assert wide.bytes_at(4108) - verify.bytes_at(4108) >= replay + 16 * (128 - 12) * extent * 4
    assert prompt.bytes_at(4108) - verify.bytes_at(4108) == 2048 * 360448 - 16 * 12 * extent * 4
