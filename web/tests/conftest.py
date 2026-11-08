import pytest

_missing = []
for dependency in ("fastapi", "httpx", "uvicorn", "psutil", "huggingface_hub"):
    try:
        pytest.importorskip(dependency, reason=f"Web tests need optional package {dependency}")
    except pytest.skip.Exception:
        _missing.append(dependency)

if _missing:
    def _web_dependencies():
        pytest.skip("Web tests need optional packages: " + ", ".join(_missing))

    # A skipped item per module avoids imports and gives a standalone optional suite a successful exit.
    class _OptionalWebModule(pytest.Module):
        def collect(self):
            return [pytest.Function.from_parent(self, name="web_dependencies", callobj=_web_dependencies)]

    def pytest_pycollect_makemodule(module_path, parent):
        return _OptionalWebModule.from_parent(parent, path=module_path)
else:
    import living_fixtures

    from tensorfold_web import server

    home = living_fixtures.home

    @pytest.fixture(autouse=True)
    def _testclient_is_local(monkeypatch):
        """FastAPI's test client is a local caller named testserver."""

        monkeypatch.setattr(server, "LOOPBACK_NAMES", server.LOOPBACK_NAMES | {"testserver"})
