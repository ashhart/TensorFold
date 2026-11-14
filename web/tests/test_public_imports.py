import pytest


@pytest.mark.parametrize(("module", "names"), [
    ("engine", {"Engine", "Health", "build_command", "memory_limit_env", "parse_metrics"}),
    ("bench", {"Bench", "Store", "PROTOCOL", "load_suites", "stream_once", "summarize"}),
    ("living", {"Teacher", "Target", "folder", "preview", "snapshot_of"}),
    ("report", {"findings", "share_text", "html_report"}),
    ("server", {"create_app", "frame_policy", "connect_ctx"}),
])
def test_existing_public_imports_survive_the_module_split(module, names):
    namespace = {}
    exec(f"from tensorfold_web.{module} import *", namespace)  # noqa: S102 - Fixed module names test star imports.
    assert names <= namespace.keys()
