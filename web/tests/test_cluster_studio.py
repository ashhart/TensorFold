"""Tests for TensorFold Cluster Studio (Master/Slave APIs, heterogeneous TP/PP/Hybrid planner)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tensorfold_web.cluster_studio import ClusterStudio, _is_glm53_full_layer
from tensorfold_web.server import create_app
from tensorfold_web.slave import create_slave_app


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


def test_master_cluster_routes_mounted():
    app = create_app()
    c = TestClient(app)
    r = c.get("/api/cluster/state")
    assert r.status_code == 200
    body = r.json()
    assert body["master_id"] == "master"
    assert body["summary"]["nodes_online"] >= 1
    assert "slave_join_command" in body
