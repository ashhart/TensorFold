"""Tests for TensorFold Cluster Studio (Master/Slave APIs, heterogeneous TP/PP/Hybrid planner)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import cli as cli_mod
from tensorfold_web import cluster_rank, cluster_ssh, settings as settings_mod, slave as slave_mod
from tensorfold_web.cluster_studio import ClusterStudio, _is_glm53_full_layer
from tensorfold_web.cluster_tls import issue_worker_cert
from tensorfold_web.server import create_app
from tensorfold_web.slave import build_slave_server, create_slave_app


def _fake_node(nid: str, name: str, role: str, ram_gb: int, bw: float, rdma: list[dict]) -> dict:
    return {
        "id": nid,
        "name": name,
        "hostname": name,
        "role": role,
        "online": True,
        "ping_ms": 0.2,
        "error": None,
        "enabled": True,
        "weight": 1.0,
        "color": "#3d6bff",
        "hardware": {
            "chip": "Apple M3 Ultra",
            "gpu_cores": 80,
            "ram_total_bytes": ram_gb * (1024**3),
            "memory_bandwidth_gbs": bw,
            "peak_tflops_fp16": 54.0,
        },
        "memory": {
            "total_bytes": ram_gb * (1024**3),
            "used_bytes": 24 * (1024**3),
            "available_bytes": (ram_gb - 24) * (1024**3),
            "pressure_pct": 5.0,
        },
        "gpu": {"util_pct": 0.0, "alloc_bytes": 0, "in_use_bytes": 0},
        "cpu": {"util_pct": 1.0, "load_1m": 0.5},
        "rdma": rdma,
        "network": [{"iface": "en0", "ip": "198.51.100.10", "kind": "lan"}],
        "models": [],
        "rank": {"state": "idle"},
    }


def _four_mac_mesh() -> list[dict]:
    return [
        _fake_node("master", "studio-1", "master", 512, 800.0, [
            {"device": "rdma_en5", "iface": "en5", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.12.1", "subnet": "10.10.12.0/30", "peer_ip": "10.10.12.2"},
            {"device": "rdma_en4", "iface": "en4", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.13.1", "subnet": "10.10.13.0/30", "peer_ip": "10.10.13.2"},
            {"device": "rdma_en3", "iface": "en3", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.14.1", "subnet": "10.10.14.0/30", "peer_ip": "10.10.14.2"},
        ]),
        _fake_node("n2", "studio-2", "slave", 512, 800.0, [
            {"device": "rdma_en5", "iface": "en5", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.12.2", "subnet": "10.10.12.0/30", "peer_ip": "10.10.12.1"},
            {"device": "rdma_en3", "iface": "en3", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.23.1", "subnet": "10.10.23.0/30", "peer_ip": "10.10.23.2"},
            {"device": "rdma_en4", "iface": "en4", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.24.1", "subnet": "10.10.24.0/30", "peer_ip": "10.10.24.2"},
        ]),
        _fake_node("n3", "studio-3", "slave", 512, 800.0, [
            {"device": "rdma_en4", "iface": "en4", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.13.2", "subnet": "10.10.13.0/30", "peer_ip": "10.10.13.1"},
            {"device": "rdma_en3", "iface": "en3", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.23.2", "subnet": "10.10.23.0/30", "peer_ip": "10.10.23.1"},
            {"device": "rdma_en5", "iface": "en5", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.34.1", "subnet": "10.10.34.0/30", "peer_ip": "10.10.34.2"},
        ]),
        _fake_node("n4", "studio-4", "slave", 512, 800.0, [
            {"device": "rdma_en3", "iface": "en3", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.14.2", "subnet": "10.10.14.0/30", "peer_ip": "10.10.14.1"},
            {"device": "rdma_en4", "iface": "en4", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.24.2", "subnet": "10.10.24.0/30", "peer_ip": "10.10.24.1"},
            {"device": "rdma_en5", "iface": "en5", "state": "PORT_ACTIVE", "mtu": 4096, "speed_gbps": 80, "ip": "10.10.34.2", "subnet": "10.10.34.0/30", "peer_ip": "10.10.34.1"},
        ]),
    ]


def _write_glm53_cfg(tmp_path: Path) -> Path:
    d = tmp_path / "GLM-5.3"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "num_attention_heads": 64,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "intermediate_size": 12288,
        "vocab_size": 154880,
        "n_routed_experts": 256,
        "num_nextn_predict_layers": 1,
    }))
    return d


def test_slave_daemon_auth_and_telemetry():
    app = create_slave_app(token="secret-cluster-token")
    c = TestClient(app)
    assert c.get("/api/slave/health").status_code == 200
    assert c.get("/api/slave/telemetry").status_code == 401
    r = c.get("/api/slave/telemetry", headers={"X-TensorFold-Token": "secret-cluster-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "slave"
    assert "hardware" in body and "memory" in body and "rdma" in body


def test_glm53_4node_tp_pp_hybrid_and_heterogeneous_plans(tmp_path: Path, monkeypatch):
    studio = ClusterStudio(state_dir=tmp_path / "state")
    model_dir = _write_glm53_cfg(tmp_path)
    nodes = _four_mac_mesh()
    monkeypatch.setattr(studio, "state", lambda: {
        "nodes": nodes,
        "links": studio._build_mesh_links(nodes),
        "models": [],
    })

    # Verify all 6 full-mesh TB5 links reconstructed
    links = studio._build_mesh_links(nodes)
    assert len(links) == 6
    assert all(lnk["state"] == "active" for lnk in links)

    # 1. 4-node TP=4: even heads, 128-aligned MoE channels, 20-port spaced RDMA peers
    tp4 = studio.plan_cluster(model_path=str(model_dir), mode="tp")
    assert tp4["world"] == 4 and tp4["tp"] == 4 and tp4["pp"] == 1 and tp4["all_fit"]
    assert tp4["effective_bandwidth_gbs"] == 3200.0
    assert len(tp4["layers"]) == 79  # 78 + MTP
    for rp in tp4["ranks"]:
        assert (rp["head_range"][1] - rp["head_range"][0]) % 2 == 0
        assert (rp["moe_half_range"][1] - rp["moe_half_range"][0]) % 128 == 0
        assert len(rp["peers"]) == 3
        assert rp["missing_tb5_links"] == []
        ports = sorted(p["port"] for p in rp["peers"])
        assert all(ports[i + 1] - ports[i] >= 20 for i in range(len(ports) - 1))

    # 2. 4-node PP=4: every stage boundary > 0 MUST snap onto a full DSA indexer layer!
    pp4 = studio.plan_cluster(model_path=str(model_dir), mode="pp")
    assert pp4["tp"] == 1 and pp4["pp"] == 4 and pp4["all_fit"]
    for rp in pp4["ranks"]:
        l0 = rp["layer_range"][0]
        if l0 > 0:
            assert _is_glm53_full_layer(l0), f"PP boundary {l0} must be a full DSA indexer layer"

    # 3. 4-node Hybrid TP2 x PP2
    hyb = studio.plan_cluster(model_path=str(model_dir), mode="hybrid")
    assert hyb["tp"] == 2 and hyb["pp"] == 2 and hyb["resolved_mode"] == "hybrid"
    assert hyb["effective_bandwidth_gbs"] == 1600.0

    # 4. Heterogeneous weighted split (2-node 2:1 ratio)
    het = studio.plan_cluster(
        node_ids=["master", "n2"],
        model_path=str(model_dir),
        mode="tp",
        custom_weights={"master": 2.0, "n2": 1.0},
    )
    assert het["heterogeneous"] is True
    assert het["ranks"][0]["head_count"] > het["ranks"][1]["head_count"]
    assert het["ranks"][0]["head_count"] % 2 == 0
    assert het["ranks"][0]["head_count"] + het["ranks"][1]["head_count"] == 64


def test_master_cluster_routes_mounted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "web_cfg")
    app = create_app()
    c = TestClient(app)
    r = c.get("/api/cluster/state")
    assert r.status_code == 200
    body = r.json()
    assert body["master_id"] == "master"
    assert body["summary"]["nodes_online"] >= 1
    assert "slave_join_command" in body
    assert "cluster_token" not in body
    assert "--host 127.0.0.1" in body["slave_join_command"]
    assert "--token-file" in body["slave_join_command"]
    assert "--host 0.0.0.0" not in body["slave_join_command"]


def test_worker_sync_bundle_runs_standalone(tmp_path: Path):
    """Copying ONLY cluster_ssh.SLAVE_SYNC_FILES into an isolated dir must start the worker cleanly."""
    assert "cluster_rank.py" in cluster_ssh.SLAVE_SYNC_FILES
    pkg_src = Path(cluster_ssh.__file__).resolve().parent
    bundle_pkg = tmp_path / "isolated_root" / "tensorfold_web"
    bundle_pkg.mkdir(parents=True)
    for fname in cluster_ssh.SLAVE_SYNC_FILES:
        shutil.copy2(pkg_src / fname, bundle_pkg / fname)

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path / "isolated_root")}
    pythons = [sys.executable]
    if Path("/usr/bin/python3").is_file() and Path("/usr/bin/python3").resolve() != Path(sys.executable).resolve():
        pythons.append("/usr/bin/python3")
    for py in pythons:
        # -S suppresses site-packages so only stdlib + isolated_root are importable
        res_help = subprocess.run([py, "-S", "-m", "tensorfold_web.slave", "--help"], env=env, capture_output=True, text=True, timeout=10)
        assert res_help.returncode == 0, f"{py} --help failed: {res_help.stderr}"
        assert "--tls-cert" in res_help.stdout and "--token-file" in res_help.stdout
        res_imp = subprocess.run(
            [
                py, "-S", "-c",
                "from tensorfold_web.slave import build_slave_server, serve_slave_stdlib; "
                "from tensorfold_web.cluster_telemetry import RankManager, collect_local_telemetry; "
                "print('BUNDLE_OK')",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert res_imp.returncode == 0, f"{py} import failed: {res_imp.stderr}"
        assert "BUNDLE_OK" in res_imp.stdout


def test_slave_cli_and_server_require_tls_and_token_for_non_loopback():
    with pytest.raises(SystemExit):
        slave_mod.main(["--host", "0.0.0.0", "--port", "8091"])
    with pytest.raises(SystemExit):
        slave_mod.main(["--host", "0.0.0.0", "--port", "8091", "--token", "secret"])
    with pytest.raises(ValueError, match="TLS"):
        build_slave_server(host="0.0.0.0", port=0, token="secret")
    with pytest.raises(ValueError, match="together"):
        build_slave_server(host="127.0.0.1", port=0, tls_cert="/tmp/only-cert.pem")
    with pytest.raises(SystemExit):
        cli_mod.main(["--slave", "--host", "0.0.0.0", "--port", "8091", "--token", "secret"])


def test_slave_https_transport_and_plaintext_refusal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "web_cfg")
    studio = ClusterStudio(state_dir=tmp_path / "state")
    assert stat.S_IMODE(studio.config_path.stat().st_mode) == 0o600

    # Plaintext non-loopback URLs must be refused before any network call
    with pytest.raises(ValueError, match="unencrypted http://"):
        studio.register_node(name="bad-node", slave_url="http://10.10.12.2:8091", token="tok")
    with pytest.raises(ValueError, match="unencrypted http://"):
        studio._slave_request("GET", "http://10.10.12.2:8091", "/api/slave/telemetry", token="tok")
    with pytest.raises(ValueError, match="token is required"):
        studio.register_node(name="no-tok", slave_url="https://10.10.12.2:8091")

    # API route also refuses plaintext non-loopback and hides raw tokens on success
    app = create_app()
    c = TestClient(app)
    assert c.post("/api/cluster/nodes", json={"slave_url": "http://10.10.12.2:8091", "token": "tok"}).status_code == 400
    assert c.post("/api/cluster/nodes", json={"slave_url": "https://10.10.12.2:8091"}).status_code == 400
    reg = c.post("/api/cluster/nodes", json={"slave_url": "10.10.12.2:8091", "token": "node-secret-xyz", "name": "tls-peer"})
    assert reg.status_code == 200
    reg_body = reg.json()
    assert reg_body["slave_url"] == "https://10.10.12.2:8091"
    assert reg_body["token_set"] is True
    assert "token" not in reg_body

    # End-to-end real TLS handshake with CA-signed worker cert + per-node token
    cert_path, key_path = issue_worker_cert(studio.tls_dir, "worker-local", ["127.0.0.1", "localhost"])
    assert stat.S_IMODE((studio.tls_dir / "ca.key").stat().st_mode) == 0o600
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600

    worker_tok = "per-node-secret-token-42"
    srv, mgr = build_slave_server(
        host="127.0.0.1",
        port=0,
        token=worker_tok,
        tls_cert=str(cert_path),
        tls_key=str(key_path),
        rank_manager=cluster_rank.RankManager(tmp_path / "slave_rank"),
    )
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        r_bad = studio._slave_request("GET", f"https://127.0.0.1:{port}", "/api/slave/telemetry", token="wrong-token")
        assert r_bad.status_code == 401
        r_ok = studio._slave_request("GET", f"https://127.0.0.1:{port}", "/api/slave/telemetry", token=worker_tok)
        assert r_ok.status_code == 200
        assert r_ok.json()["role"] == "slave"
    finally:
        srv.shutdown()
        mgr.stop()
        srv.server_close()


def test_bootstrap_slave_uses_tls_token_file_and_syncs_all_modules(tmp_path: Path, monkeypatch):
    studio = ClusterStudio(state_dir=tmp_path / "state")
    monkeypatch.setattr(studio, "test_ssh", lambda *a, **kw: {"ok": True, "hostname": "worker-2"})

    commands: list[tuple[list[str], bytes | None]] = []
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if cmd and Path(str(cmd[0])).name == "openssl":
            return real_run(cmd, *a, **kw)
        commands.append((list(cmd), kw.get("input")))
        if cmd[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="hostname 10.10.12.2\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"role": "slave", "hostname": "worker-2"}

    captured_req: dict = {}

    def fake_slave_req(method, url, path, *, token=None, json_body=None, timeout=5.0):
        captured_req.update(method=method, url=url, path=path, token=token)
        return _Resp()

    monkeypatch.setattr(cluster_ssh.subprocess, "run", fake_run)
    monkeypatch.setattr(studio, "_slave_request", fake_slave_req)

    res = studio.bootstrap_slave(host="worker-2", slave_port=8091)
    assert res["node"]["slave_url"] == "https://10.10.12.2:8091"
    assert res["node"]["token_set"] is True
    assert "token" not in res["node"]
    assert captured_req["url"] == "https://10.10.12.2:8091"
    assert captured_req["token"] and captured_req["token"] != studio.cluster_token

    # All required worker modules including cluster_rank.py were copied
    scp_srcs = [Path(cmd[-2]).name for cmd, _ in commands if cmd and cmd[0] == "scp"]
    for mod in cluster_ssh.SLAVE_SYNC_FILES:
        assert mod in scp_srcs, f"{mod} was not synced by bootstrap_slave"

    # Launch command uses --token-file, --tls-cert, --tls-key and NEVER leaks the raw token in argv
    launch_cmds = [cmd[-1] for cmd, _ in commands if cmd and cmd[0] == "ssh" and "tensorfold_web.slave" in cmd[-1]]
    assert len(launch_cmds) == 1
    assert "--token-file" in launch_cmds[0]
    assert "--tls-cert" in launch_cmds[0]
    assert "--tls-key" in launch_cmds[0]
    assert "--token " not in launch_cmds[0]
    for cmd, _ in commands:
        joined = " ".join(str(x) for x in cmd)
        assert captured_req["token"] not in joined
    # Token was written over SSH stdin
    stdin_payloads = [inp.decode("utf-8", errors="ignore") for _, inp in commands if inp]
    assert any(captured_req["token"] in p for p in stdin_payloads)


def _install_fake_native(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    record_file = tmp_path / "launched.json"
    fake_bin = tmp_path / "tensorfold-native"
    fake_bin.write_text(
        f"#!{sys.executable}\n"
        "import json, os, stat, sys, time\n"
        "args = sys.argv[1:]\n"
        "key_info = None\n"
        "if '--api-key-file' in args:\n"
        "    kp = args[args.index('--api-key-file') + 1]\n"
        "    key_info = {'path': kp, 'mode': stat.S_IMODE(os.stat(kp).st_mode), 'text': open(kp).read().strip()}\n"
        f"open({str(record_file)!r}, 'w').write(json.dumps({{'args': args, 'key': key_info, 'tf': os.environ.get('TF_TEST_FLAG')}}))\n"
        "print('listening on test server', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)
    monkeypatch.setattr(cluster_rank, "_find_rank_binaries", lambda: {"tensorfold_native": str(fake_bin)})
    return fake_bin, record_file


def _wait_record(record_file: Path) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if record_file.is_file() and record_file.stat().st_size > 0:
            return json.loads(record_file.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError("fake tensorfold-native did not record launch")


def test_rank_manager_blocks_binary_override_and_unsafe_env(tmp_path: Path, monkeypatch):
    _, record_file = _install_fake_native(tmp_path, monkeypatch)
    model_dir = _write_glm53_cfg(tmp_path)
    mgr = cluster_rank.RankManager(tmp_path / "rank_state")
    try:
        with pytest.raises(ValueError, match="custom binary"):
            mgr.start({"binary": "/bin/sh", "model_path": str(model_dir)})
        with pytest.raises(ValueError, match="disallowed rank environment variable"):
            mgr.start({"model_path": str(model_dir), "env": {"DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib"}})
        with pytest.raises(ValueError, match="disallowed rank environment variable"):
            mgr.start({"model_path": str(model_dir), "env": {"PATH": "/tmp"}})
        with pytest.raises(ValueError, match="API key"):
            mgr.start({"model_path": str(model_dir), "host": "0.0.0.0"})

        mgr.start({"model_path": str(model_dir), "env": {"TF_TEST_FLAG": "allowed-1"}})
        rec = _wait_record(record_file)
        assert rec["tf"] == "allowed-1"
        assert rec["args"][rec["args"].index("--host") + 1] == "127.0.0.1"
    finally:
        mgr.stop()


def test_deploy_defaults_to_loopback_and_enforces_api_key(tmp_path: Path, monkeypatch):
    _, record_file = _install_fake_native(tmp_path, monkeypatch)
    model_dir = _write_glm53_cfg(tmp_path)
    single_node = [_fake_node("master", "studio-1", "master", 1024, 800.0, [])]

    settings_key: dict[str, str | None] = {"value": None}
    studio = ClusterStudio(state_dir=tmp_path / "state", get_api_key=lambda: settings_key["value"])
    monkeypatch.setattr(studio, "state", lambda: {"nodes": single_node, "links": [], "models": []})

    try:
        # 1. Default deployment binds loopback 127.0.0.1
        dep_local = studio.deploy_cluster(model_path=str(model_dir), mode="auto")
        rec_local = _wait_record(record_file)
        assert dep_local["serve_host"] == "127.0.0.1"
        assert dep_local["api_auth_enabled"] is False
        assert rec_local["args"][rec_local["args"].index("--host") + 1] == "127.0.0.1"
        assert rec_local["key"] is None

        # 2. Non-loopback without API key is refused
        with pytest.raises(ValueError, match="API key"):
            studio.deploy_cluster(model_path=str(model_dir), serve_host="0.0.0.0")

        # 3. Non-loopback with explicit API key passes 0600 --api-key-file without leaking secret
        record_file.unlink(missing_ok=True)
        secret_key = "sk-cluster-auth-999"
        dep_pub = studio.deploy_cluster(model_path=str(model_dir), serve_host="0.0.0.0", api_key=secret_key)
        rec_pub = _wait_record(record_file)
        assert dep_pub["serve_host"] == "0.0.0.0"
        assert dep_pub["api_auth_enabled"] is True
        assert rec_pub["args"][rec_pub["args"].index("--host") + 1] == "0.0.0.0"
        assert "--api-key" not in rec_pub["args"]
        assert "--api-key-file" in rec_pub["args"]
        assert rec_pub["key"]["mode"] == 0o600
        assert rec_pub["key"]["text"] == secret_key
        assert secret_key not in json.dumps(dep_pub)
        assert secret_key not in json.dumps(studio.master_rank.status())
        assert all(secret_key not in line for line in studio.master_rank.logs())

        key_path = Path(rec_pub["key"]["path"])
        studio.stop_cluster()
        assert not key_path.exists()

        # 4. Non-loopback inherits API key from TensorFold Web settings when configured
        record_file.unlink(missing_ok=True)
        settings_key["value"] = "sk-settings-key-777"
        dep_settings = studio.deploy_cluster(model_path=str(model_dir), serve_host="0.0.0.0")
        rec_settings = _wait_record(record_file)
        assert dep_settings["api_auth_enabled"] is True
        assert rec_settings["key"]["mode"] == 0o600
        assert rec_settings["key"]["text"] == "sk-settings-key-777"
    finally:
        studio.stop_cluster()

    # 5. HTTP route /api/cluster/deploy also refuses non-loopback without API key
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "web_cfg")
    app = create_app()
    c = TestClient(app)
    r_refused = c.post("/api/cluster/deploy", json={"model_path": str(model_dir), "serve_host": "0.0.0.0"})
    assert r_refused.status_code == 400
    assert "API key" in r_refused.json()["detail"]
