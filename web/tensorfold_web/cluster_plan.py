"""Model geometry inspection, TB5 RDMA mesh graph, and heterogeneous TP/PP sharding planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GLM53_FULL_INDEXER_LAYERS = [0, 1, 2] + list(range(6, 78, 4))


def is_glm53_full_layer(layer: int) -> bool:
    return layer < 3 or (layer >= 6 and (layer - 2) % 4 == 0)


def _snap_even(val: int, lo: int, hi: int) -> int:
    v = max(lo, min(hi, val))
    return v - (v % 2)


def _snap_mult(val: int, mult: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(val / mult)) * mult))


def build_mesh_links(studio, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match TB5 /30 RDMA subnets across all online nodes to reconstruct the exact physical cable graph."""
    by_subnet: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for n in nodes:
        for port in n.get("rdma") or []:
            sub = port.get("subnet")
            if sub:
                by_subnet.setdefault(sub, []).append((n, port))

    links: list[dict[str, Any]] = []
    probes = studio._data.get("link_probes") or {}
    for subnet, endpoints in sorted(by_subnet.items()):
        if len(endpoints) == 2:
            (na, pa), (nb, pb) = endpoints
            key = f"{min(na['hostname'], nb['hostname'])}<->{max(na['hostname'], nb['hostname'])}:{subnet}"
            pr = probes.get(key) or {}
            active = (pa.get("state") == "PORT_ACTIVE" and pb.get("state") == "PORT_ACTIVE")
            tx_mbps = round(float(pa.get("tx_mbps", 0)) + float(pb.get("tx_mbps", 0)), 2)
            rx_mbps = round(float(pa.get("rx_mbps", 0)) + float(pb.get("rx_mbps", 0)), 2)
            links.append({
                "id": key,
                "subnet": subnet,
                "kind": "tb5_rdma",
                "speed_gbps": min(int(pa.get("speed_gbps", 80)), int(pb.get("speed_gbps", 80))),
                "mtu": min(int(pa.get("mtu", 4096)), int(pb.get("mtu", 4096))),
                "state": "active" if active else "degraded",
                "a_id": na["id"],
                "a_name": na["name"],
                "a_device": pa["device"],
                "a_iface": pa["iface"],
                "a_ip": pa["ip"],
                "b_id": nb["id"],
                "b_name": nb["name"],
                "b_device": pb["device"],
                "b_iface": pb["iface"],
                "b_ip": pb["ip"],
                "traffic_mbps": round(max(tx_mbps, rx_mbps), 2),
                "a_tx_mbps": pa.get("tx_mbps", 0.0),
                "b_tx_mbps": pb.get("tx_mbps", 0.0),
                "latency_us": pr.get("latency_us"),
                "bandwidth_gbs": pr.get("bandwidth_gbs"),
                "probed_at": pr.get("probed_at"),
            })
    return links


def model_geometry(studio, model_path: str) -> dict[str, Any]:
    p = Path(model_path)
    cfg: dict[str, Any] = {}
    if (p / "config.json").is_file():
        try:
            cfg = json.loads((p / "config.json").read_text())
        except (OSError, ValueError):
            pass
    model_type = str(cfg.get("model_type") or "glm_moe_dsa")
    n_layers = int(cfg.get("num_hidden_layers") or 78)
    n_heads = int(cfg.get("num_attention_heads") or 64)
    hidden = int(cfg.get("hidden_size") or 6144)
    moe_inter = int(cfg.get("moe_intermediate_size") or 2048)
    dense_inter = int(cfg.get("intermediate_size") or 12288)
    vocab = int(cfg.get("vocab_size") or 154880)
    n_experts = int(cfg.get("n_routed_experts") or 256)
    n_dense_layers = int(cfg.get("first_k_dense_replace") or 3)
    has_mtp = int(cfg.get("num_nextn_predict_layers") or (1 if model_type == "glm_moe_dsa" else 0)) > 0

    # Estimate exact checkpoint bytes (prefer real index/disk size, else known GLM-5.3 FP8 = 703.77 GiB)
    total_weight_bytes = 0
    dl_file = p / ".download_status.json"
    if dl_file.is_file():
        try:
            total_weight_bytes = int(json.loads(dl_file.read_text()).get("total_bytes") or 0)
        except (OSError, ValueError):
            pass
    if total_weight_bytes == 0 and p.is_dir():
        st_sz = sum(f.stat().st_size for f in p.glob("*.safetensors") if f.is_file())
        if st_sz > 0:
            total_weight_bytes = st_sz
    if total_weight_bytes == 0:
        total_weight_bytes = int(703.77 * (1024**3)) if model_type == "glm_moe_dsa" else int(32 * (1024**3))

    # Per-token KV cache bytes per layer (GLM-5.3 MLA latent 576 bf16 + indexer K 128 fp8).
    if model_type == "glm_moe_dsa":
        kv_bytes_per_token_per_layer = (512 + 64) * 2 + 128 + 4  # 1,284 bytes/tok/layer
    else:
        kv_heads = int(cfg.get("num_key_value_heads") or max(1, n_heads // 4))
        head_dim = int(cfg.get("head_dim") or (hidden // max(1, n_heads)))
        kv_bytes_per_token_per_layer = kv_heads * head_dim * 2 * 2

    return {
        "name": p.name or "GLM-5.3",
        "path": str(p),
        "model_type": model_type,
        "num_layers": n_layers,
        "num_dense_layers": n_dense_layers,
        "num_heads": n_heads,
        "hidden_size": hidden,
        "moe_intermediate": moe_inter,
        "dense_intermediate": dense_inter,
        "vocab_size": vocab,
        "n_experts": n_experts,
        "has_mtp": has_mtp,
        "total_weight_bytes": total_weight_bytes,
        "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
    }


def link_between(studio, a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Find matching TB5 /30 RDMA interface pair between node a and node b."""
    b_by_sub = {p["subnet"]: p for p in (b.get("rdma") or []) if p.get("subnet")}
    for pa in a.get("rdma") or []:
        sub = pa.get("subnet")
        if sub and sub in b_by_sub:
            return pa, b_by_sub[sub]
    return None, None


def plan_cluster(studio,
    *,
    node_ids: list[str] | None = None,
    model_path: str | None = None,
    mode: str = "auto",
    context: int = 8192,
    parallel: int = 1,
    tp_override: int | None = None,
    pp_override: int | None = None,
    custom_weights: dict[str, float] | None = None,
    base_port: int = 19740,
) -> dict[str, Any]:
    """Compute an exact heterogeneous TP / PP / Hybrid TPxPP sharding plan + rankN.json configs."""
    st = studio.state()
    nodes_by_id = {n["id"]: n for n in st["nodes"] if n["online"]}
    if node_ids:
        selected = [nodes_by_id[nid] for nid in node_ids if nid in nodes_by_id]
    else:
        selected = [n for n in st["nodes"] if n["online"] and n.get("enabled", True)]
    if not selected:
        raise ValueError("No online nodes selected for cluster plan")
    # Ensure master is rank 0 if included
    selected.sort(key=lambda n: (0 if n["id"] == "master" else 1, n["name"]))

    if not model_path:
        if st["models"]:
            model_path = st["models"][0]["path"]
        else:
            model_path = str(Path.home() / "models" / "GLM-5.3")
    geo = model_geometry(studio, model_path)

    world = len(selected)
    # Determine (tp, pp)
    if mode == "tp":
        tp, pp = world, 1
    elif mode == "pp":
        tp, pp = 1, world
    elif mode == "hybrid":
        if tp_override and pp_override and tp_override * pp_override == world:
            tp, pp = tp_override, pp_override
        elif world == 4:
            tp, pp = 2, 2
        elif world == 6:
            tp, pp = 3, 2
        elif world == 8:
            tp, pp = 4, 2
        else:
            tp, pp = world, 1
    else:  # auto
        # Check if hardware is heterogeneous in RAM (>25% spread)
        rams = [max(1, n["hardware"].get("ram_total_bytes", 1)) for n in selected]
        hetero = (max(rams) / min(rams)) > 1.25 if world > 1 else False
        if world == 1:
            tp, pp = 1, 1
        elif hetero and world >= 2:
            # Heterogeneous nodes can use proportional TP (or PP if very skewed)
            tp, pp = (1, world) if (max(rams) / min(rams)) > 2.5 else (world, 1)
        else:
            # Full TB5 mesh with identical RAM: pure TP maximizes BOTH decode & prefill!
            tp, pp = world, 1

    # Compute normalized compute/memory weights per node (allows heterogeneous splits!)
    raw_w: list[float] = []
    for n in selected:
        if custom_weights and n["id"] in custom_weights and custom_weights[n["id"]] > 0:
            raw_w.append(float(custom_weights[n["id"]]))
        else:
            ram_gb = max(16.0, n["hardware"].get("ram_total_bytes", 0) / (1024**3))
            bw = max(64.0, float(n["hardware"].get("memory_bandwidth_gbs", 100.0)))
            w_override = float(n.get("weight", 1.0))
            # Blend RAM capacity & memory bandwidth for heterogeneous balance
            raw_w.append(((ram_gb * 0.6) + (bw * 0.4)) * w_override)

    # Partition layers across PP stages (snapping to full DSA indexer layers for GLM-5.3!)
    n_layers = geo["num_layers"]
    stage_weights = [sum(raw_w[s * tp : (s + 1) * tp]) for s in range(pp)]
    stage_total_w = sum(stage_weights) or 1.0
    layer_bounds = [0]
    acc_w = 0.0
    for s in range(pp - 1):
        acc_w += stage_weights[s]
        target_l = int(round(n_layers * (acc_w / stage_total_w)))
        target_l = max(layer_bounds[-1] + 1, min(n_layers - (pp - 1 - s), target_l))
        if geo["model_type"] == "glm_moe_dsa":
            # Snap to nearest full DSA indexer layer >= layer_bounds[-1] + 1
            valid = [ly for ly in GLM53_FULL_INDEXER_LAYERS if layer_bounds[-1] < ly < n_layers]
            if valid:
                target_l = min(valid, key=lambda ly: abs(ly - target_l))
        layer_bounds.append(target_l)
    layer_bounds.append(n_layers)

    # Partition TP slices inside each PP stage proportionally to node weights
    ranks_plan: list[dict[str, Any]] = []
    total_w_gib = geo["total_weight_bytes"] / (1024**3)
    # Replicated shared overhead (embeddings + norms + routers + indexers + lm_head/MTP share)
    replicated_gib = 3.6 if geo["model_type"] == "glm_moe_dsa" else total_w_gib * 0.03
    sharded_layers_gib = max(0.0, total_w_gib - replicated_gib)

    for r, node in enumerate(selected):
        pp_r = r // tp
        tp_r = r % tp
        l0, l1 = layer_bounds[pp_r], layer_bounds[pp_r + 1]

        # Weights within this PP stage's TP group
        grp_w = raw_w[pp_r * tp : (pp_r + 1) * tp]
        grp_sum = sum(grp_w) or 1.0
        frac_lo = sum(grp_w[:tp_r]) / grp_sum
        frac_hi = sum(grp_w[: tp_r + 1]) / grp_sum

        # Snap heads to even (required by GLM-5.3 kv_b_proj 128-row FP8 blocks)
        h_lo = 0 if tp_r == 0 else _snap_even(int(round(geo["num_heads"] * frac_lo)), 2, geo["num_heads"] - 2)
        h_hi = geo["num_heads"] if tp_r == tp - 1 else _snap_even(int(round(geo["num_heads"] * frac_hi)), h_lo + 2, geo["num_heads"] - 2 * (tp - 1 - tp_r))

        # Snap MoE intermediate channels to 128 (FP8 block-128 aligned)
        m_lo = 0 if tp_r == 0 else _snap_mult(int(round(geo["moe_intermediate"] * frac_lo)), 128, 128, geo["moe_intermediate"] - 128)
        m_hi = geo["moe_intermediate"] if tp_r == tp - 1 else _snap_mult(int(round(geo["moe_intermediate"] * frac_hi)), 128, m_lo + 128, geo["moe_intermediate"] - 128 * (tp - 1 - tp_r))

        # Snap dense MLP channels to 128
        d_lo = 0 if tp_r == 0 else _snap_mult(int(round(geo["dense_intermediate"] * frac_lo)), 128, 128, geo["dense_intermediate"] - 128)
        d_hi = geo["dense_intermediate"] if tp_r == tp - 1 else _snap_mult(int(round(geo["dense_intermediate"] * frac_hi)), 128, d_lo + 128, geo["dense_intermediate"] - 128 * (tp - 1 - tp_r))

        # Snap vocab rows to 128
        v_lo = 0 if tp_r == 0 else _snap_mult(int(round(geo["vocab_size"] * frac_lo)), 128, 128, geo["vocab_size"] - 128)
        v_hi = geo["vocab_size"] if tp_r == tp - 1 else _snap_mult(int(round(geo["vocab_size"] * frac_hi)), 128, v_lo + 128, geo["vocab_size"] - 128 * (tp - 1 - tp_r))

        # Memory accounting for this rank
        layer_share = (l1 - l0) / max(1, n_layers)
        tp_share = (m_hi - m_lo) / max(1, geo["moe_intermediate"])
        weights_gib = round(sharded_layers_gib * layer_share * tp_share + replicated_gib * (1.0 if pp_r == 0 or pp_r == pp - 1 else 0.25), 2)
        kv_gib = round((geo["kv_bytes_per_token_per_layer"] * (l1 - l0) * context * parallel) / (1024**3), 2)
        runtime_gib = round(2.8 + (0.6 if pp_r == 0 else 0.0), 2)
        ram_total_gib = round(node["hardware"].get("ram_total_bytes", 0) / (1024**3), 1)
        load_limit_gib = round(ram_total_gib * (0.85 if ram_total_gib >= 400 else 0.75), 1)
        total_needed_gib = round(weights_gib + kv_gib + runtime_gib, 2)
        headroom_gib = round(load_limit_gib - total_needed_gib, 2)
        fits = headroom_gib > 0

        # Build exact TB5 RDMA peers list for `rankN.json`
        peers_cfg: list[dict[str, Any]] = []
        missing_links: list[str] = []
        for p_rank, p_node in enumerate(selected):
            if p_rank == r:
                continue
            pa, pb = link_between(studio, node, p_node)
            a_idx, b_idx = min(r, p_rank), max(r, p_rank)
            link_port = base_port + (a_idx * 8 + b_idx) * 20
            if pa and pb:
                peers_cfg.append({
                    "rank": p_rank,
                    "device": pa["device"],
                    "via": f"{pa['iface']}/{pb['ip']}",
                    "port": link_port,
                    "peer_port": link_port,
                    "name": p_node["name"],
                })
            else:
                # Fallback over LAN IP if no direct TB5 link found
                lan = next((x["ip"] for x in p_node.get("network") or [] if x["kind"] == "lan"), p_node["hostname"])
                missing_links.append(p_node["name"])
                peers_cfg.append({
                    "rank": p_rank,
                    "device": "rdma_en0",
                    "via": lan,
                    "port": link_port,
                    "peer_port": link_port,
                    "name": p_node["name"],
                    "fallback": True,
                })

        # Exact `rankN.json` understood by our upgraded Zig `tp_settings.zig` & `pair.zig`
        links_cfg = [
            {
                "peer": p["rank"],
                "device": p["device"],
                "via": p["via"],
                "port": p["port"],
                "peer_port": p["peer_port"],
                "name": f"mesh-{min(r, p['rank'])}-{max(r, p['rank'])}",
                "gid": -1,
            }
            for p in peers_cfg
        ]
        rank_json: dict[str, Any] = {
            "rank": r,
            "ranks": world,
            "tp": tp,
            "pp": pp,
            "tp_rank": tp_r,
            "pp_rank": pp_r,
            "library": str(Path.home() / "lib" / "libmcdma-fabric.dylib"),
            "links": links_cfg,
            "layer_range": [l0, l1],
            "head_range": [h_lo, h_hi],
            "moe_half_range": [m_lo // 128, m_hi // 128],
            "dense_range": [d_lo, d_hi],
            "vocab_range": [v_lo, v_hi],
            "load_limit_gb": load_limit_gib,
        }

        ranks_plan.append({
            "rank": r,
            "node_id": node["id"],
            "name": node["name"],
            "role": node["role"],
            "color": node["color"],
            "chip": node["hardware"].get("chip"),
            "gpu_cores": node["hardware"].get("gpu_cores", 0),
            "bandwidth_gbs": node["hardware"].get("memory_bandwidth_gbs", 0.0),
            "tp_rank": tp_r,
            "pp_rank": pp_r,
            "layer_range": [l0, l1],
            "layer_count": l1 - l0,
            "head_range": [h_lo, h_hi],
            "head_count": h_hi - h_lo,
            "moe_half_range": [m_lo, m_hi],
            "moe_channels": m_hi - m_lo,
            "dense_range": [d_lo, d_hi],
            "vocab_range": [v_lo, v_hi],
            "share_pct": round(tp_share * 100.0, 1),
            "weights_gib": weights_gib,
            "kv_gib": kv_gib,
            "runtime_gib": runtime_gib,
            "total_needed_gib": total_needed_gib,
            "ram_total_gib": ram_total_gib,
            "load_limit_gib": load_limit_gib,
            "headroom_gib": headroom_gib,
            "fits": fits,
            "peers": peers_cfg,
            "missing_tb5_links": missing_links,
            "rank_json": rank_json,
        })

    # Build visual layer-by-layer shard allocation map (all layers + MTP!)
    layer_map: list[dict[str, Any]] = []
    for ly in range(n_layers):
        owners = [rp for rp in ranks_plan if rp["layer_range"][0] <= ly < rp["layer_range"][1]]
        kind = "dense" if ly < geo["num_dense_layers"] else "moe"
        indexer = "full" if (geo["model_type"] != "glm_moe_dsa" or is_glm53_full_layer(ly)) else "shared"
        layer_map.append({
            "layer": ly,
            "kind": kind,
            "indexer": indexer,
            "pp_stage": owners[0]["pp_rank"] if owners else 0,
            "shards": [
                {
                    "rank": o["rank"],
                    "node_id": o["node_id"],
                    "name": o["name"],
                    "color": o["color"],
                    "pct": o["share_pct"],
                    "heads": f"{o['head_range'][0]}–{o['head_range'][1] - 1}",
                    "moe_ch": f"{o['moe_half_range'][0]}–{o['moe_half_range'][1] - 1}",
                }
                for o in owners
            ],
        })
    if geo["has_mtp"]:
        last_stage = [rp for rp in ranks_plan if rp["pp_rank"] == pp - 1]
        layer_map.append({
            "layer": n_layers,
            "kind": "mtp",
            "indexer": "full",
            "pp_stage": pp - 1,
            "shards": [
                {
                    "rank": o["rank"],
                    "node_id": o["node_id"],
                    "name": o["name"],
                    "color": o["color"],
                    "pct": o["share_pct"],
                    "heads": f"{o['head_range'][0]}–{o['head_range'][1] - 1}",
                    "moe_ch": f"{o['moe_half_range'][0]}–{o['moe_half_range'][1] - 1}",
                }
                for o in last_stage
            ],
        })

    # Effective aggregate memory bandwidth (TP adds bandwidth across nodes in a stage; PP pipelines across stages)
    stage_bws = [sum(rp["bandwidth_gbs"] for rp in ranks_plan if rp["pp_rank"] == s) for s in range(pp)]
    effective_bw_gbs = round(min(stage_bws) if stage_bws else 0.0, 1)
    all_reduce_per_token = (n_layers * 2) if tp > 1 else 0
    pp_handoffs_per_token = (pp - 1) if pp > 1 else 0
    all_fit = all(rp["fits"] for rp in ranks_plan)
    is_hetero = len({rp["share_pct"] for rp in ranks_plan}) > 1 or len({rp["ram_total_gib"] for rp in ranks_plan}) > 1

    return {
        "model": geo,
        "mode": mode,
        "resolved_mode": "single" if world == 1 else ("hybrid" if tp > 1 and pp > 1 else ("tp" if tp > 1 else "pp")),
        "world": world,
        "tp": tp,
        "pp": pp,
        "heterogeneous": is_hetero,
        "context": context,
        "parallel": parallel,
        "all_fit": all_fit,
        "effective_bandwidth_gbs": effective_bw_gbs,
        "total_cluster_ram_gib": round(sum(rp["ram_total_gib"] for rp in ranks_plan), 1),
        "total_weights_gib": round(total_w_gib, 2),
        "all_reduce_per_token": all_reduce_per_token,
        "pp_handoffs_per_token": pp_handoffs_per_token,
        "ranks": ranks_plan,
        "layers": layer_map,
    }
