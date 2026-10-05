"""Real CUDA gates for the dynamic-layout helper, before any serving timing."""

from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest


def test_dynamic_layouts_actual_compute_and_original_commit():
    """The same qualification called at startup passes the real small-model kernels."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('requires Flash Next sm_12x CUDA')
    from test_flashnext_forward import _Rand, _bf16_table, _cfg, _model, _ple
    import _mixed_row_graph_cases as qualification
    c = _cfg(ple=True)
    with tempfile.TemporaryDirectory() as directory:
        table = _bf16_table(Path(directory)/'shard_0.safetensors', c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(19)))
        result = qualification.qualify(SimpleNamespace(w=w, streams={}, filling=[]))
        assert result['case_count'] == 12 and result['exact']
        assert all(case['exact'] for case in result['cases'])
        assert {(sum(case['widths']), case['parity']) for case in result['cases']
                if case['captures_delta']} == {(13, 0), (13, 1), (18, 0), (18, 1)}
        assert result['old_states_released'] and result['captures'] == 5
        assert result['replays'] == 9
