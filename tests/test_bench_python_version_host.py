"""Benchmark tools refuse old interpreters before defining annotated request helpers."""

from pathlib import Path
import subprocess
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / 'tools'


@pytest.mark.parametrize('tool', ['bench_openai.py', 'bench_concurrent.py'])
@pytest.mark.parametrize('version', [(3, 9, 0), (3, 10, 0)])
def test_old_python_refuses_before_tool_definitions(tool, version):
    source = (TOOLS / tool).read_text()
    prefix = f'import sys; sys.version_info = {version!r}; sys.version = {".".join(map(str, version))!r}\n'
    rc = subprocess.run([sys.executable, '-c', prefix + source, '--help'], capture_output=True, text=True)
    assert rc.returncode != 0
    assert 'Python 3.11+ is required' in rc.stderr
    assert '.'.join(map(str, version)) in rc.stderr
    assert f'python3.12 tools/{tool}' in rc.stderr
    assert 'Traceback' not in rc.stderr


@pytest.mark.parametrize('tool', ['bench_openai.py', 'bench_concurrent.py'])
def test_supported_python_keeps_the_help_command(tool):
    rc = subprocess.run([sys.executable, str(TOOLS / tool), '--help'], capture_output=True, text=True)
    assert rc.returncode == 0 and 'usage:' in rc.stdout
    assert 'Python 3.11+ is required' not in rc.stderr
