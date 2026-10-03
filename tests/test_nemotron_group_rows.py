"""Nemotron windows that share their rows' expert reads from two rows keep every row's one-token-step bits."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import rows  # noqa: E402

from test_nemotron_pre_m5_streams import IDS, _same, _tiny  # noqa: E402


@pytest.fixture(scope="module")
def nem():
    from tensorfold.families.nemotron_h.model import NemotronH

    patch = pytest.MonkeyPatch()
    patch.setattr(K, "tensor_units", lambda: False)          # the same kernels on every Mac
    yield NemotronH(_tiny(), mtp_path=None, drafts=0, tokenizer=SimpleNamespace(encode=lambda text: IDS))
    patch.undo()


def test_tensor_unit_install_groups_from_two_rows(monkeypatch):
    from tensorfold.families.nemotron_h.model import NemotronH
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    monkeypatch.setattr(K, "tensor_units", lambda: True)
    monkeypatch.setattr(lane_qmm, "install", lambda *a, **k: None)
    monkeypatch.setattr(lane_qmm, "warm", lambda *a, **k: 0)
    fam = SimpleNamespace(model=object(), fused=SimpleNamespace(qkv={}, lane_xs=False, group_rows=rows.GROUP_ROWS))
    NemotronH._install_lane_matmul(fam)
    assert fam.fused.group_rows == 2 and rows.GROUP_ROWS == 8     # other Macs keep the module's threshold


@pytest.mark.parametrize("group_rows", [8, 2])
def test_windows_equal_serial_steps_at_either_threshold(nem, group_rows, monkeypatch):
    calls = []
    route_group = rows.route_group
    monkeypatch.setattr(rows, "route_group", lambda *a, **k: calls.append(1) or route_group(*a, **k))
    nem.fused.group_rows = group_rows
    nem.fused._compiled_blocks.clear()                        # a traced block keeps the threshold it saw
    copy = LaneEngine.copy_single_cache
    base = nem.make_cache()
    mx.eval(nem.hidden(mx.array([IDS[:48]], dtype=mx.uint32), base))
    one, serial = copy(base), []
    for t in IDS[48:57]:
        serial.append(nem.head(nem.hidden(mx.array([[t]], dtype=mx.uint32), one))[0, -1])
    mx.eval(serial)
    for width in range(2, 10):
        calls.clear()
        logits = nem.head(nem.hidden(mx.array([IDS[48:48 + width]], dtype=mx.uint32), copy(base)))[0]
        mx.eval(logits)
        assert all(_same(logits[i], serial[i]) for i in range(width)), f"{width} rows from {group_rows}"
        assert bool(calls) == (width >= group_rows), f"{width} rows grouped from {group_rows}"
    nem.fused.group_rows = rows.GROUP_ROWS
    nem.fused._compiled_blocks.clear()
