"""`tensorfold cluster|node` and `serve --cluster` exec the native engine; end to end on fake hosts when it is built."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tensorfold import cli, native
from tensorfold.native import cluster
from tests.test_native_contract import fake_binary

RUN = ("import sys; from pathlib import Path; import tensorfold.native as native; native.ROOT = Path(sys.argv[1]); "
       "from tensorfold.cli import main; raise SystemExit(main(sys.argv[2:]))")
BUILT = Path(os.environ.get("TENSORFOLD_NATIVE_TEST_BINARY", Path(__file__).parents[1] / "zig-out" / "bin" / "tensorfold"))


class Execd(Exception):
    """os.execv was reached; a real one never returns."""


@pytest.fixture
def execd(monkeypatch):
    calls = []

    def fake(path, argv):
        calls.append((Path(path), argv))
        raise Execd()

    monkeypatch.setattr(os, "execv", fake)
    return calls


@pytest.mark.parametrize("argv, routed", [
    (["cluster", "init", "--node", "a=h"], True),
    (["cluster"], True),
    (["node", "--name", "a", "--cluster", "c.json"], True),
    (["serve", "MODEL", "--cluster", "c.json"], True),
    (["serve", "--cluster=c.json"], True),
    (["serve", "MODEL", "--port", "9"], False),
    (["serve", "MODEL", "--name", "--cluster"], True),
    (["models"], False),
    ([], False),
])
def test_cluster_command_lines_go_to_the_native_engine(argv, routed):
    assert cluster.wanted(argv) is routed


def test_the_native_engine_gets_the_whole_command_line(tmp_path, monkeypatch, execd):
    binary = fake_binary(tmp_path / native.BINARY, "sys.exit(0)")
    monkeypatch.setattr(native, "ROOT", tmp_path)
    for argv in (["cluster", "check", "--cluster", "c.json"], ["serve", "kimi-k3", "--cluster", "c.json", "--dry-run"],
                 ["node", "--name", "node-b", "--cluster", "c.json"]):
        with pytest.raises(Execd):
            cli.main(argv)
    assert [call[0] for call in execd] == [binary] * 3
    assert execd[1][1] == ["tensorfold", "serve", "kimi-k3", "--cluster", "c.json", "--dry-run"]


def test_without_a_native_engine_a_cluster_command_says_why(tmp_path, monkeypatch, capsys, execd):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    assert cli.main(["cluster", "status", "--cluster", "c.json"]) == 1
    assert "clusters run on the native engine" in capsys.readouterr().err and not execd


def _domain(n: int, bus: int) -> str:
    return f"{n + 1:02X}000000-0000-4000-8000-0000000000{bus:02X}"


MESH = [(0, 1, 3, 1), (0, 2, 2, 2), (0, 3, 1, 3), (1, 1, 2, 1), (1, 2, 3, 2), (2, 3, 3, 3)]


def _peer(n: int, bus: int) -> tuple[int, int] | None:
    for a, ab, b, bb in MESH:
        if (a, ab) == (n, bus):
            return b, bb
        if (b, bb) == (n, bus):
            return a, ab
    return None


def write_hosts(root: Path, names: list[str]) -> None:
    """Saved probe output for four Macs cabled in a full mesh, in the formats macOS 27 prints."""

    for n, name in enumerate(names):
        host = root / name
        host.mkdir(parents=True)
        (host / "sysctl.txt").write_text("hw.memsize: 549755813888\nhw.pagesize: 16384\niogpu.wired_limit_mb: 0\n"
                                         "vm.page_free_count: 24000000\nvm.page_speculative_count: 200000\n"
                                         "machdep.cpu.brand_string: Apple M3 Ultra\nkern.osproductversion: 27.0\n")
        (host / "displays.txt").write_text("Graphics/Displays:\n\n    Apple M3 Ultra:\n\n      Chipset Model: Apple M3 Ultra\n"
                                           "      Type: GPU\n      Total Number of Cores: 80\n")
        (host / "df.txt").write_text("Filesystem 1024-blocks Used Available Capacity iused ifree %iused Mounted on\n"
                                     "/dev/disk3s5 7811085600 4385517656 1000000 57% 1 2 0% /System/Volumes/Data\n")
        tb, dev = ["Thunderbolt/USB4:\n\n"], []
        for bus in range(4):
            far = _peer(n, bus)
            tb.append(f"    Thunderbolt/USB4 Bus {bus}:\n\n      Domain UUID: {_domain(n, bus)}\n      Port:\n"
                      f"          Status: {'Device connected' if far else 'No device connected'}\n"
                      f"          Speed: {'80 Gb/s' if far else 'Up to 120 Gb/s'}\n          Receptacle: {bus + 1}\n\n")
            if far:
                tb.append(f"        Mac Studio:\n\n          Domain UUID: {_domain(*far)}\n\n")
            raw = bytes.fromhex(_domain(n, bus).replace("-", ""))
            gid = ":".join(raw[i:i + 2].hex() for i in range(0, 16, 2))
            dev.append(f"hca_id:\trdma_en{bus + 2}\n\ttransport:\t\t\tThunderbolt (100)\n\tatomic_cap:\t\t\tATOMIC_NONE (0)\n"
                       f"\t\t\tstate:\t\t\t{'PORT_ACTIVE (4)' if far else 'PORT_DOWN (1)'}\n\t\t\tGID[  0]:\t\t{gid}\n\n")
        (host / "thunderbolt.txt").write_text("".join(tb))
        (host / "ibv_devinfo.txt").write_text("".join(dev))


GLM = {"model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 78, "vocab_size": 151552,
       "num_attention_heads": 64, "num_key_value_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512,
       "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "n_routed_experts": 256,
       "num_experts_per_tok": 8, "n_shared_experts": 1, "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
       "intermediate_size": 12288, "num_nextn_predict_layers": 1, "index_topk": 2048, "index_n_heads": 32,
       "index_head_dim": 128, "max_position_embeddings": 202752,
       "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}


@pytest.mark.skipif(not BUILT.is_file(), reason="needs the native engine built (zig build tensorfold)")
def test_end_to_end_on_fake_hosts_through_the_tensorfold_command(tmp_path):
    root = tmp_path / "native"
    (root / "bin").mkdir(parents=True)
    shutil.copy2(BUILT, root / native.BINARY)
    probes = tmp_path / "probes"
    write_hosts(probes, ["s1", "s2", "s3", "s4"])
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "TENSORFOLD_NO_UPDATE_CHECK": "1"}

    def tensorfold(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", RUN, str(root), *argv], capture_output=True, text=True, env=env,
                              timeout=600)

    init = tensorfold("cluster", "init", "--node", "s1=192.0.2.10", "--node", "s2=s2.local@s1", "--node", "s3=s3.local@s1",
                      "--node", "s4=s4.local@s1", "--name", "lab", "--probes", str(probes))
    assert init.returncode == 0, init.stderr
    found = json.loads(init.stdout)
    assert [len(n["ports"]) for n in found["nodes"]] == [3, 3, 3, 3] and found["nodes"][1]["via"] == "s1"
    found["models"] = {"glm-5.3": {"path": "/models/glm53", "parallel": {"tensor": 4, "expert": 4}, "streams": 32,
                                   "context": 8192}}
    (tmp_path / "cluster.json").write_text(json.dumps(found))
    (tmp_path / "glm.json").write_text(json.dumps(GLM))
    common = ["--cluster", str(tmp_path / "cluster.json"), "--model-config", str(tmp_path / "glm.json"), "--probes",
              str(probes)]
    check = tensorfold("cluster", "check", *common)
    assert check.returncode == 0 and "admission: experts spread, MLA by" in check.stdout and "128 rows" in check.stdout
    serve = tensorfold("serve", *common)
    assert serve.returncode == 0 and "fake cluster up: 4 nodes agree on plan" in serve.stdout
    assert (probes / "launches.log").read_text().count("tensorfold node --name") == 3
    node = tensorfold("node", "--name", "s2", "--cluster", str(tmp_path / "cluster.json"), "--probes", str(probes))
    assert node.returncode == 3 and "s2" in node.stdout
