"""The app's API keeps answering whatever the GPU driver is doing (no GPU, no network; nvidia-smi is stood in for)."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)


def test_the_state_never_waits_on_the_gpu_driver():
    """/api/state ran nvidia-smi twice a call with no time limit. It is asked once, with a limit, and kept."""

    if not server.SPARK:
        return
    real, calls = subprocess.run, []

    def stand_in(command, *args, **kwargs):
        if command and command[0] == "nvidia-smi":
            calls.append(kwargs.get("timeout"))
            if stand_in.hangs:  # the driver is busy: what subprocess does when the limit passes
                raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
            return subprocess.CompletedProcess(command, 0, "NVIDIA GB10\n", "")
        return real(command, *args, **kwargs)

    server.subprocess.run, had = stand_in, list(server.GPU_NAME)
    try:
        server.GPU_NAME.clear()
        stand_in.hangs = True
        started = time.time()
        state = client.get("/api/state").json()
        assert time.time() - started < 3 and state["platform"] == "spark" and "CUDA" in state["platform_label"]
        assert calls and all(limit and limit <= 5 for limit in calls)  # every call has a limit
        # the driver answers again: asked once more, then never again
        stand_in.hangs = False
        calls.clear()
        for _ in range(3):
            assert "GB10" in client.get("/api/state").json()["platform_label"]
        assert len(calls) == 1
        # and the job list and one job never ask it at all
        calls.clear()
        assert client.get("/api/jobs").status_code == 200 and client.get("/api/jobs/none").status_code == 404
        assert calls == []
    finally:
        server.subprocess.run = real
        server.GPU_NAME[:] = had


def test_no_request_runs_a_tool_without_a_time_limit():
    """Every subprocess a request handler can reach has a timeout; only the job thread's renders run unlimited."""

    import ast

    # server.py and the modules it was split into
    names = ("server", "settings", "prompt_text", "motion_prompt", "engines", "picture_models", "job_build", "job_run",
             "job_state", "api_jobs", "api_files")
    trees = [ast.parse((ROOT / "app" / f"{name}.py").read_text()) for name in names if (ROOT / "app" / f"{name}.py").is_file()]
    unlimited = []
    for function in [node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        for call in [node for node in ast.walk(function) if isinstance(node, ast.Call)]:
            name = ast.unparse(call.func)
            if name == "subprocess.run" and not any(keyword.arg == "timeout" for keyword in call.keywords):
                unlimited.append(function.name)
    assert sorted(set(unlimited)) == ["export", "hush", "long_take"], sorted(set(unlimited))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
