"""HTTP routes for TensorFold Cluster Studio (Master control plane)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException

from .cluster_studio import ClusterStudio
from .cluster_tls import normalize_and_validate_slave_url


def install(app, studio: ClusterStudio) -> None:
    app.state.cluster = studio

    @app.get("/api/cluster/state")
    def cluster_state():
        return studio.state()

    @app.post("/api/cluster/discover")
    def cluster_discover():
        return {"candidates": studio.discover_peers()}

    @app.post("/api/cluster/ssh/test")
    def cluster_ssh_test(body: dict[str, Any]):
        host = str(body.get("host") or "").strip()
        if not host:
            raise HTTPException(400, "host is required")
        return studio.test_ssh(
            host,
            user=str(body["user"]).strip() if body.get("user") else None,
            port=int(body.get("port") or 22),
            key_path=str(body["key_path"]).strip() if body.get("key_path") else None,
        )

    @app.post("/api/cluster/ssh/bootstrap")
    def cluster_ssh_bootstrap(body: dict[str, Any]):
        host = str(body.get("host") or "").strip()
        if not host:
            raise HTTPException(400, "host is required")
        try:
            return studio.bootstrap_slave(
                host=host,
                user=str(body["user"]).strip() if body.get("user") else None,
                port=int(body.get("port") or 22),
                key_path=str(body["key_path"]).strip() if body.get("key_path") else None,
                name=str(body["name"]).strip() if body.get("name") else None,
                slave_port=int(body.get("slave_port") or 8091),
                slave_host=str(body["slave_host"]).strip() if body.get("slave_host") else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cluster/nodes")
    def cluster_register_node(body: dict[str, Any]):
        raw_url = str(body.get("slave_url") or "").strip()
        name = str(body.get("name") or "").strip()
        try:
            slave_url = normalize_and_validate_slave_url(raw_url, allow_schemeless=True)
            if not name:
                name = urlsplit(slave_url).hostname or slave_url
            return studio.register_node(
                name=name,
                slave_url=slave_url,
                ssh_host=str(body["ssh_host"]).strip() if body.get("ssh_host") else None,
                ssh_user=str(body["ssh_user"]).strip() if body.get("ssh_user") else None,
                ssh_port=int(body.get("ssh_port") or 22),
                ssh_key=str(body["ssh_key"]).strip() if body.get("ssh_key") else None,
                slave_port=int(body.get("slave_port") or 8091),
                token=str(body["token"]).strip() if body.get("token") else None,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/cluster/nodes/{node_id}")
    def cluster_update_node(node_id: str, body: dict[str, Any]):
        try:
            return studio.update_node(node_id, body)
        except KeyError as exc:
            raise HTTPException(404, "unknown node") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/cluster/nodes/{node_id}")
    def cluster_delete_node(node_id: str):
        try:
            studio.remove_node(node_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown node") from exc
        return {"removed": node_id}

    @app.post("/api/cluster/plan")
    def cluster_plan(body: dict[str, Any]):
        try:
            return studio.plan_cluster(
                node_ids=body.get("node_ids"),
                model_path=body.get("model_path"),
                mode=str(body.get("mode") or "auto"),
                context=int(body.get("context") or 8192),
                parallel=int(body.get("parallel") or 1),
                tp_override=int(body["tp"]) if body.get("tp") else None,
                pp_override=int(body["pp"]) if body.get("pp") else None,
                custom_weights=body.get("custom_weights"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cluster/deploy")
    def cluster_deploy(body: dict[str, Any]):
        try:
            return studio.deploy_cluster(
                node_ids=body.get("node_ids"),
                model_path=body.get("model_path"),
                mode=str(body.get("mode") or "auto"),
                context=int(body.get("context") or 8192),
                parallel=int(body.get("parallel") or 1),
                prompt_cache_gib=float(body.get("prompt_cache_gib") or 0.0),
                tp_override=int(body["tp"]) if body.get("tp") else None,
                pp_override=int(body["pp"]) if body.get("pp") else None,
                custom_weights=body.get("custom_weights"),
                serve_port=int(body.get("serve_port") or 8080),
                serve_host=str(body.get("serve_host") or "127.0.0.1").strip(),
                api_key=str(body["api_key"]).strip() if body.get("api_key") else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cluster/stop")
    def cluster_stop():
        return studio.stop_cluster()

    @app.post("/api/cluster/probe-links")
    def cluster_probe_links():
        return studio.probe_all_links()

    @app.post("/api/cluster/sync-model")
    def cluster_sync_model(body: dict[str, Any]):
        model_path = str(body.get("model_path") or "").strip()
        if not model_path:
            raise HTTPException(400, "model_path is required")
        try:
            return studio.sync_model_to_slaves(model_path, body.get("node_ids"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cluster/benchmark")
    def cluster_benchmark(body: dict[str, Any]):
        try:
            return studio.run_benchmark(
                prompt_tokens=body.get("prompt_tokens"),
                decode_tokens=int(body.get("decode_tokens") or 128),
                draft=bool(body.get("draft", True)),
                label=body.get("label"),
                engine_url=body.get("engine_url"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cluster/benchmarks/record")
    def cluster_record_benchmark(body: dict[str, Any]):
        return studio.record_external_benchmark(body)
