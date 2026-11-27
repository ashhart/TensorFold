"""TensorFold Cluster Studio coordinator for SSH bootstrap, TB5 RDMA sharding, and orchestration."""

from __future__ import annotations

import json
import os
import secrets
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from . import cluster_ops, cluster_plan, cluster_ssh
from . import settings as settings_mod
from .cluster_plan import GLM53_FULL_INDEXER_LAYERS, is_glm53_full_layer as _is_glm53_full_layer
from .cluster_telemetry import RankManager, collect_local_telemetry
from .cluster_tls import ensure_cluster_ca, is_loopback_url, normalize_and_validate_slave_url

NODE_PALETTE = [
    "#3d6bff",
    "#8b4dff",
    "#ff7a5c",
    "#22d3ee",
    "#3ccf91",
    "#f2b234",
    "#e0507a",
    "#a855f7",
]

__all__ = ["ClusterStudio", "GLM53_FULL_INDEXER_LAYERS", "NODE_PALETTE", "_is_glm53_full_layer"]


class ClusterStudio:
    """Master-side coordinator for heterogeneous Apple Silicon / GPU clusters."""

    def __init__(self, state_dir: Path | None = None, on_engine_url: Any = None, get_api_key: Any = None) -> None:
        self.state_dir = state_dir or settings_mod.config_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.state_dir / "cluster_studio.json"
        self.bench_path = self.state_dir / "cluster_benchmarks.json"
        self.tls_dir = self.state_dir / "cluster_tls"
        self.ca_cert_path = self.tls_dir / "ca.crt"
        self.on_engine_url = on_engine_url
        self.get_api_key = get_api_key
        self._lock = threading.Lock()
        self.master_rank = RankManager(self.state_dir / "cluster")
        self._data = self._load_config()
        self._benchmarks: list[dict[str, Any]] = self._load_benchmarks()
        self._active_deployment: dict[str, Any] | None = None
        self._sync_jobs: dict[str, dict[str, Any]] = {}
        self._bench_job: dict[str, Any] | None = None
        self._probe_running = False
        self._history: list[dict[str, Any]] = []

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.is_file():
            try:
                os.chmod(self.config_path, 0o600)
            except OSError:
                pass
            try:
                doc = json.loads(self.config_path.read_text())
                if isinstance(doc, dict) and doc.get("cluster_token"):
                    doc.setdefault("nodes", [])
                    doc.setdefault("link_probes", {})
                    return doc
            except (OSError, ValueError):
                pass
        doc = {
            "cluster_token": secrets.token_hex(16),
            "nodes": [],
            "link_probes": {},
        }
        self._save_config_raw(doc)
        return doc

    def _save_config_raw(self, doc: dict[str, Any]) -> None:
        tmp = self.config_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(doc, indent=2))
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        os.replace(tmp, self.config_path)
        os.chmod(self.config_path, 0o600)

    def _save_config(self) -> None:
        with self._lock:
            self._save_config_raw(self._data)

    def _load_benchmarks(self) -> list[dict[str, Any]]:
        if self.bench_path.is_file():
            try:
                data = json.loads(self.bench_path.read_text())
                if isinstance(data, list):
                    return data
            except (OSError, ValueError):
                pass
        return []

    def _save_benchmarks(self) -> None:
        with self._lock:
            tmp = self.bench_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._benchmarks, indent=2))
            tmp.replace(self.bench_path)

    @property
    def cluster_token(self) -> str:
        return str(self._data["cluster_token"])

    def _configured_api_key(self) -> str | None:
        if not self.get_api_key:
            return None
        try:
            val = self.get_api_key()
            return str(val).strip() or None if val else None
        except Exception:  # noqa: BLE001
            return None

    def ensure_tls_ca(self) -> Path:
        ca_crt, _ = ensure_cluster_ca(self.tls_dir)
        return ca_crt

    def _tls_verify(self) -> ssl.SSLContext | bool:
        if self.ca_cert_path.is_file():
            ctx = ssl.create_default_context(cafile=str(self.ca_cert_path))
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            return ctx
        return True

    def _slave_headers(self, token: str | None = None) -> dict[str, str]:
        tok = (token or "").strip()
        return {"X-TensorFold-Token": tok} if tok else {}

    def _slave_request(
        self,
        method: str,
        slave_url: str,
        path: str,
        *,
        token: str | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> httpx.Response:
        """Send an authenticated worker control request, refusing plaintext HTTP on non-loopback hosts."""
        base = normalize_and_validate_slave_url(slave_url)
        tok = (token or "").strip() or None
        if not is_loopback_url(base) and not tok:
            raise ValueError("a per-node worker token is required for non-loopback slave requests")
        verify: ssl.SSLContext | bool = self._tls_verify() if base.startswith("https://") else True
        return httpx.request(
            method,
            f"{base}{path}",
            json=json_body,
            headers=self._slave_headers(tok),
            timeout=timeout,
            verify=verify,
            trust_env=False,
        )

    @staticmethod
    def _public_node(rec: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in rec.items() if k != "token"}
        out["token_set"] = bool(rec.get("token"))
        return out

    def register_node(
        self,
        *,
        name: str,
        slave_url: str,
        ssh_host: str | None = None,
        ssh_user: str | None = None,
        ssh_port: int = 22,
        ssh_key: str | None = None,
        slave_port: int = 8091,
        token: str | None = None,
    ) -> dict[str, Any]:
        slave_url = normalize_and_validate_slave_url(slave_url)
        tok = (token or "").strip() or None
        with self._lock:
            existing = next((n for n in self._data["nodes"] if n["slave_url"] == slave_url or n["name"] == name), None)
            effective_tok = tok or (existing.get("token") if existing is not None else None)
            if not is_loopback_url(slave_url) and not effective_tok:
                raise ValueError("token is required when registering a non-loopback slave node")
            if existing is not None:
                existing.update(
                    name=name,
                    slave_url=slave_url,
                    ssh_host=ssh_host or existing.get("ssh_host"),
                    ssh_user=ssh_user if ssh_user is not None else existing.get("ssh_user"),
                    ssh_port=ssh_port,
                    ssh_key=ssh_key if ssh_key is not None else existing.get("ssh_key"),
                    slave_port=slave_port,
                    token=effective_tok,
                )
                rec = dict(existing)
            else:
                rec = {
                    "id": f"node-{secrets.token_hex(4)}",
                    "name": name,
                    "slave_url": slave_url,
                    "ssh_host": ssh_host,
                    "ssh_user": ssh_user,
                    "ssh_port": ssh_port,
                    "ssh_key": ssh_key,
                    "slave_port": slave_port,
                    "token": effective_tok,
                    "enabled": True,
                    "weight": 1.0,
                    "added_at": time.time(),
                }
                self._data["nodes"].append(rec)
        self._save_config()
        return self._public_node(rec)

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            before = len(self._data["nodes"])
            self._data["nodes"] = [n for n in self._data["nodes"] if n["id"] != node_id]
            if len(self._data["nodes"]) == before:
                raise KeyError(node_id)
        self._save_config()

    def update_node(self, node_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node = next((n for n in self._data["nodes"] if n["id"] == node_id), None)
            if node is None:
                raise KeyError(node_id)
            next_url = node["slave_url"]
            if "slave_url" in patch and patch["slave_url"] is not None:
                next_url = normalize_and_validate_slave_url(str(patch["slave_url"]))
            next_tok = node.get("token")
            if "token" in patch and patch["token"] is not None:
                next_tok = str(patch["token"]).strip() or None
            if not is_loopback_url(next_url) and not next_tok:
                raise ValueError("token is required for non-loopback slave nodes")
            for k in ("name", "enabled", "weight", "ssh_host", "ssh_user", "ssh_key"):
                if k in patch and patch[k] is not None:
                    node[k] = patch[k]
            node["slave_url"] = next_url
            node["token"] = next_tok
            out = dict(node)
        self._save_config()
        return self._public_node(out)

    def _poll_slave(self, node: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        url = node["slave_url"].rstrip("/")
        tok = node.get("token")
        try:
            r = self._slave_request("GET", url, "/api/slave/telemetry", token=tok, timeout=2.2)
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if r.status_code == 200:
                tele = r.json()
                return {
                    "id": node["id"],
                    "name": node["name"],
                    "role": "slave",
                    "online": True,
                    "ping_ms": ms,
                    "error": None,
                    "enabled": node.get("enabled", True),
                    "weight": float(node.get("weight", 1.0)),
                    "ssh_host": node.get("ssh_host"),
                    "slave_url": url,
                    **tele,
                }
            err = f"HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            err = str(exc)[:160]
        return {
            "id": node["id"],
            "name": node["name"],
            "role": "slave",
            "online": False,
            "ping_ms": ms,
            "error": err,
            "enabled": node.get("enabled", True),
            "weight": float(node.get("weight", 1.0)),
            "ssh_host": node.get("ssh_host"),
            "slave_url": url,
            "hostname": node["name"],
            "hardware": {"chip": "Offline", "ram_total_bytes": 0, "gpu_cores": 0, "memory_bandwidth_gbs": 0},
            "memory": {"total_bytes": 0, "used_bytes": 0, "available_bytes": 0, "pressure_pct": 0},
            "gpu": {"util_pct": 0, "alloc_bytes": 0, "in_use_bytes": 0},
            "cpu": {"util_pct": 0, "load_1m": 0},
            "rdma": [],
            "models": [],
            "rank": {"state": "offline"},
        }


    def state(self) -> dict[str, Any]:
        master_tele = collect_local_telemetry(self.master_rank, role="master")
        master_node = {
            "id": "master",
            "name": master_tele["hostname"],
            "role": "master",
            "online": True,
            "ping_ms": 0.0,
            "error": None,
            "enabled": True,
            "weight": 1.0,
            "ssh_host": "localhost",
            "slave_url": None,
            **master_tele,
        }
        with self._lock:
            reg_nodes = [dict(n) for n in self._data["nodes"]]

        slave_nodes: list[dict[str, Any]] = []
        if reg_nodes:
            with ThreadPoolExecutor(max_workers=min(8, len(reg_nodes))) as ex:
                slave_nodes = list(ex.map(self._poll_slave, reg_nodes))

        all_nodes = [master_node, *slave_nodes]
        for idx, n in enumerate(all_nodes):
            n["color"] = NODE_PALETTE[idx % len(NODE_PALETTE)]

        links = self._build_mesh_links(all_nodes)
        online_nodes = [n for n in all_nodes if n["online"]]

        # Aggregate cluster-wide totals
        total_ram = sum(n["hardware"].get("ram_total_bytes", 0) for n in online_nodes)
        used_ram = sum(n["memory"].get("used_bytes", 0) for n in online_nodes)
        avail_ram = sum(n["memory"].get("available_bytes", 0) for n in online_nodes)
        total_gpu_cores = sum(n["hardware"].get("gpu_cores", 0) for n in online_nodes)
        total_bw_gbs = round(sum(n["hardware"].get("memory_bandwidth_gbs", 0.0) for n in online_nodes), 1)
        total_tflops = round(sum(n["hardware"].get("peak_tflops_fp16", 0.0) for n in online_nodes), 1)
        avg_gpu_util = round(
            sum(n["gpu"].get("util_pct", 0.0) for n in online_nodes) / max(1, len(online_nodes)), 1
        )
        total_gpu_alloc = sum(n["gpu"].get("alloc_bytes", 0) for n in online_nodes)
        total_rdma_mbps = round(sum(lnk.get("traffic_mbps", 0.0) for lnk in links), 2)

        # Pull live tok/s & engine stats from rank 0 (Master) if serving
        master_stats = (master_node.get("rank") or {}).get("stats") or {}
        decode_tps = float(master_stats.get("decode_tokens_per_second") or 0.0)
        prefill_tps = float(master_stats.get("prefill_tokens_per_second") or 0.0)
        drafted = int(master_stats.get("mtp_drafted_total") or 0)
        accepted = int(master_stats.get("mtp_accepted_total") or 0)
        mtp_accept_rate = round((accepted / drafted) * 100.0, 1) if drafted > 0 else None

        # Record live telemetry sparkline point (120 samples)
        pt = {
            "t": round(time.time(), 1),
            "decode_tps": round(decode_tps, 2),
            "prefill_tps": round(prefill_tps, 2),
            "gpu_util": avg_gpu_util,
            "rdma_mbps": total_rdma_mbps,
        }
        with self._lock:
            if not self._history or (pt["t"] - self._history[-1]["t"]) >= 0.8:
                self._history.append(pt)
                if len(self._history) > 90:
                    self._history = self._history[-90:]
            history_copy = list(self._history)

        # Collect distinct models known on Master (plus per-node availability matrix)
        master_models = master_node.get("models") or []
        for m in master_models:
            per_node = []
            for n in all_nodes:
                nm = next((x for x in (n.get("models") or []) if x["name"] == m["name"]), None)
                per_node.append({
                    "node_id": n["id"],
                    "name": n["name"],
                    "present": nm is not None,
                    "complete": bool(nm and nm.get("complete")),
                    "shards_present": nm.get("shards_present", 0) if nm else 0,
                    "shards_expected": nm.get("shards_expected", m["shards_expected"]) if nm else m["shards_expected"],
                    "bytes_on_disk": nm.get("bytes_on_disk", 0) if nm else 0,
                })
            m["cluster_presence"] = per_node

        # Build secure manual join command (loopback + token file; use SSH bootstrap for automatic TLS)
        lan_ip = next((x["ip"] for x in master_tele.get("network") or [] if x["kind"] == "lan"), master_tele["hostname"])
        join_cmd = (
            "tensorfold-web --slave --host 127.0.0.1 --port 8091 "
            "--token-file ~/.config/tensorfold/slave.token"
        )

        return {
            "timestamp": time.time(),
            "master_id": "master",
            "master_lan_ip": lan_ip,
            "tls_required": True,
            "slave_join_command": join_cmd,
            "summary": {
                "nodes_total": len(all_nodes),
                "nodes_online": len(online_nodes),
                "links_total": len(links),
                "links_active": sum(1 for lnk in links if lnk["state"] == "active"),
                "total_ram_bytes": total_ram,
                "used_ram_bytes": used_ram,
                "available_ram_bytes": avail_ram,
                "total_gpu_cores": total_gpu_cores,
                "total_gpu_alloc_bytes": total_gpu_alloc,
                "total_bandwidth_gbs": total_bw_gbs,
                "total_tflops_fp16": total_tflops,
                "avg_gpu_util_pct": avg_gpu_util,
                "total_rdma_mbps": total_rdma_mbps,
                "decode_tps": round(decode_tps, 2),
                "prefill_tps": round(prefill_tps, 2),
                "mtp_accept_rate": mtp_accept_rate,
                "running_requests": int(master_stats.get("running") or 0),
                "waiting_requests": int(master_stats.get("waiting") or 0),
                "prompt_tokens_total": int(master_stats.get("prompt_tokens_total") or 0),
                "generation_tokens_total": int(master_stats.get("generation_tokens_total") or 0),
                "ttft_p50_ms": master_stats.get("ttft_p50_ms"),
            },
            "nodes": all_nodes,
            "links": links,
            "models": master_models,
            "deployment": self._active_deployment,
            "sync_jobs": dict(self._sync_jobs),
            "benchmark_job": self._bench_job,
            "benchmarks": list(self._benchmarks),
            "probe_running": self._probe_running,
            "history": history_copy,
        }


    def discover_peers(self) -> list[dict[str, Any]]:
        return cluster_ssh.discover_peers(self)

    def test_ssh(self, host: str, user: str | None = None, port: int = 22, key_path: str | None = None) -> dict[str, Any]:
        return cluster_ssh.test_ssh(self, host, user=user, port=port, key_path=key_path)

    def bootstrap_slave(
        self,
        *,
        host: str,
        user: str | None = None,
        port: int = 22,
        key_path: str | None = None,
        slave_port: int = 8091,
        slave_host: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        return cluster_ssh.bootstrap_slave(
            self,
            host=host,
            user=user,
            port=port,
            key_path=key_path,
            slave_port=slave_port,
            slave_host=slave_host,
            name=name,
        )

    def _build_mesh_links(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return cluster_plan.build_mesh_links(self, nodes)

    def _model_geometry(self, model_path: str) -> dict[str, Any]:
        return cluster_plan.model_geometry(self, model_path)

    def _link_between(self, a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return cluster_plan.link_between(self, a, b)

    def plan_cluster(
        self,
        *,
        node_ids: list[str] | None = None,
        model_path: str | None = None,
        mode: str = "auto",
        context: int = 8192,
        parallel: int = 1,
        tp_override: int | None = None,
        pp_override: int | None = None,
        custom_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return cluster_plan.plan_cluster(
            self,
            node_ids=node_ids,
            model_path=model_path,
            mode=mode,
            context=context,
            parallel=parallel,
            tp_override=tp_override,
            pp_override=pp_override,
            custom_weights=custom_weights,
        )

    def deploy_cluster(
        self,
        *,
        node_ids: list[str] | None = None,
        model_path: str | None = None,
        mode: str = "auto",
        context: int = 8192,
        parallel: int = 1,
        prompt_cache_gib: float = 0.0,
        tp_override: int | None = None,
        pp_override: int | None = None,
        custom_weights: dict[str, float] | None = None,
        serve_port: int = 8080,
        serve_host: str = "127.0.0.1",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        return cluster_ops.deploy_cluster(
            self,
            node_ids=node_ids,
            model_path=model_path,
            mode=mode,
            context=context,
            parallel=parallel,
            prompt_cache_gib=prompt_cache_gib,
            tp_override=tp_override,
            pp_override=pp_override,
            custom_weights=custom_weights,
            serve_port=serve_port,
            serve_host=serve_host,
            api_key=api_key,
        )

    def stop_cluster(self) -> dict[str, Any]:
        return cluster_ops.stop_cluster(self)

    def probe_all_links(self) -> dict[str, Any]:
        return cluster_ops.probe_all_links(self)

    def sync_model_to_slaves(self, model_path: str, node_ids: list[str] | None = None) -> dict[str, Any]:
        return cluster_ssh.sync_model_to_slaves(self, model_path, node_ids=node_ids)

    def run_benchmark(
        self,
        *,
        prompt_tokens: list[int] | None = None,
        decode_tokens: int = 128,
        draft: bool = True,
        label: str | None = None,
        engine_url: str | None = None,
    ) -> dict[str, Any]:
        return cluster_ops.run_benchmark(
            self,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            draft=draft,
            label=label,
            engine_url=engine_url,
        )

    def record_external_benchmark(self, record: dict[str, Any]) -> dict[str, Any]:
        return cluster_ops.record_external_benchmark(self, record)
