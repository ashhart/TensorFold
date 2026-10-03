"""Both HTTP layers report live footprint bytes once or omit the unavailable family."""

from types import SimpleNamespace

import pytest

from tensorfold.server import metrics
from tests.test_metrics import get, serve


@pytest.mark.parametrize('layer', ['mac', 'cuda'])
@pytest.mark.parametrize('reading', [None, 0, 4096])
def test_live_footprint_family_on_both_routes(layer, reading, tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, 'process_footprint', lambda: reading)

    def check(port):
        for path in ('/metrics', '/v1/metrics'):
            status, content_type, body = get(port, path)
            assert status == 200 and content_type.startswith('text/plain')
            prefix = metrics.PREFIX + 'process_footprint_bytes'
            samples = [line for line in body.splitlines() if line.startswith(prefix + ' ')]
            if reading is None:
                assert not samples and prefix not in body
            else:
                assert samples == [prefix + ' ' + str(reading)]
                assert body.count('# TYPE ' + prefix + ' gauge') == 1
                assert body.count('# HELP ' + prefix + ' ') == 1

    if layer == 'mac':
        app = SimpleNamespace(served_name='test', model_ids=['test'], max_batch_size=1)
        httpd, thread = serve(app)
        try:
            check(httpd.server_port)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(5)
    else:
        from tests.test_cuda_server_disconnect import PacedEngine, app_for, serving

        with serving(app_for(tmp_path, PacedEngine(hold_at=0))) as port:
            check(port)
