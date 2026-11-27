"""Cluster rank deployment, RDMA mesh probing, and benchmark orchestration."""

from __future__ import annotations

import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from .cluster_rank import is_loopback_host
from .cluster_telemetry import probe_rdma_link


def deploy_cluster(studio,
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
    serve_host = (serve_host or "127.0.0.1").strip()
    if not serve_host or serve_host.startswith("-") or any(c.isspace() for c in serve_host):
        raise ValueError(f"invalid serve_host: {serve_host!r}")
    effective_api_key = (api_key or "").strip() or studio._configured_api_key()
    if effective_api_key and any(c.isspace() for c in effective_api_key):
        raise ValueError("api_key must not contain whitespace")
    if not is_loopback_host(serve_host) and not effective_api_key:
        raise ValueError(
            "binding the inference API on a non-loopback address requires an API key; "
            "pass api_key, configure one in Settings, or keep serve_host='127.0.0.1'"
        )

    plan = studio.plan_cluster(
        node_ids=node_ids,
        model_path=model_path,
        mode=mode,
        context=context,
        parallel=parallel,
        tp_override=tp_override,
        pp_override=pp_override,
        custom_weights=custom_weights,
    )
    if not plan["all_fit"]:
        bad = [f"{r['name']} needs {r['total_needed_gib']} GiB > {r['load_limit_gib']} GiB limit" for r in plan["ranks"] if not r["fits"]]
        raise RuntimeError("Cluster plan exceeds memory budget: " + "; ".join(bad))

    studio.stop_cluster()
    with studio._lock:
        nodes_by_id = {n["id"]: n for n in studio._data["nodes"]}

    # Start follower ranks (rank N-1 .. 1) on slaves concurrently so RDMA handshakes connect cleanly
    errors: list[str] = []

    def start_one(rp: dict[str, Any]) -> None:
        spec = {
            "rank": rp["rank"],
            "world": plan["world"],
            "tp": plan["tp"],
            "pp": plan["pp"],
            "tp_rank": rp["tp_rank"],
            "pp_rank": rp["pp_rank"],
            "model_path": plan["model"]["path"],
            "port": serve_port if rp["rank"] == 0 else serve_port + rp["rank"],
            "host": serve_host if rp["rank"] == 0 else "127.0.0.1",
            "context": context,
            "parallel": parallel,
            "prompt_cache_gib": prompt_cache_gib,
            "temperature": 0 if plan["model"].get("model_type") == "glm_moe_dsa" else None,
            "layer_total": rp["layer_count"],
            "load_limit_gb": rp["load_limit_gib"],
            "rank_settings": rp["rank_json"],
        }
        if rp["rank"] == 0 and effective_api_key:
            spec["api_key"] = effective_api_key
        if rp["node_id"] == "master":
            studio.master_rank.start(spec)
        else:
            node = nodes_by_id[rp["node_id"]]
            r = studio._slave_request(
                "POST",
                node["slave_url"],
                "/api/slave/rank/start",
                token=node.get("token"),
                json_body=spec,
                timeout=10.0,
            )
            if r.status_code != 200:
                raise RuntimeError(f"{rp['name']} (rank {rp['rank']}): {r.text[:200]}")

    followers = [rp for rp in plan["ranks"] if rp["rank"] > 0]
    master_rp = next((rp for rp in plan["ranks"] if rp["rank"] == 0), None)

    if followers:
        with ThreadPoolExecutor(max_workers=len(followers)) as ex:
            futs = [(rp, ex.submit(start_one, rp)) for rp in followers]
            for rp, fut in futs:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
    if errors:
        studio.stop_cluster()
        raise RuntimeError("Failed to start follower ranks: " + " | ".join(errors))

    if followers:
        time.sleep(0.4)
    if master_rp is not None:
        try:
            start_one(master_rp)
        except Exception as exc:  # noqa: BLE001
            studio.stop_cluster()
            raise RuntimeError(f"Failed to start Master rank 0: {exc}") from exc

    studio._active_deployment = {
        "id": f"dep-{secrets.token_hex(4)}",
        "started_at": time.time(),
        "serve_host": serve_host,
        "serve_port": serve_port,
        "api_auth_enabled": bool(effective_api_key),
        "model_name": plan["model"]["name"],
        "model_path": plan["model"]["path"],
        "mode": plan["resolved_mode"],
        "world": plan["world"],
        "tp": plan["tp"],
        "pp": plan["pp"],
        "context": context,
        "parallel": parallel,
        "node_ids": [rp["node_id"] for rp in plan["ranks"]],
        "plan": plan,
    }
    if studio.on_engine_url:
        try:
            studio.on_engine_url(f"http://127.0.0.1:{serve_port}", effective_api_key)
        except TypeError:
            try:
                studio.on_engine_url(f"http://127.0.0.1:{serve_port}")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    return studio._active_deployment


def stop_cluster(studio) -> dict[str, Any]:
    studio.master_rank.stop()
    with studio._lock:
        reg_nodes = [dict(n) for n in studio._data["nodes"]]

    def stop_slave(n: dict[str, Any]) -> None:
        try:
            studio._slave_request(
                "POST",
                n["slave_url"],
                "/api/slave/rank/stop",
                token=n.get("token"),
                timeout=4.0,
            )
        except (httpx.HTTPError, ValueError):
            pass

    if reg_nodes:
        with ThreadPoolExecutor(max_workers=len(reg_nodes)) as ex:
            list(ex.map(stop_slave, reg_nodes))
    studio._active_deployment = None
    return {"stopped": True}


def probe_all_links(studio) -> dict[str, Any]:
    """Sequentially run live fabric-check RDMA probes on every active TB5 link in the mesh."""
    if studio._probe_running:
        return {"status": "already_running"}
    studio._probe_running = True

    def worker() -> None:
        try:
            st = studio.state()
            nodes = {n["id"]: n for n in st["nodes"]}
            reg_nodes = {n["id"]: n for n in studio._data["nodes"]}
            port = 19780
            for link in st["links"]:
                if link["state"] != "active":
                    continue
                port += 20
                a_id, b_id = link["a_id"], link["b_id"]
                na, nb = nodes.get(a_id), nodes.get(b_id)
                if not na or not nb:
                    continue
                via_a = f"{link['a_iface']}/{link['b_ip']}"
                via_b = f"{link['b_iface']}/{link['a_ip']}"

                def run_side(nid: str, dev: str, via: str, rank: int) -> dict[str, Any]:
                    if nid == "master":
                        return probe_rdma_link(dev, via, port=port, rank=rank, rounds=16, seconds=1, peer_port=port)
                    rn = reg_nodes.get(nid)
                    if not rn:
                        return {"ok": False, "error": "node missing"}
                    try:
                        r = studio._slave_request(
                            "POST",
                            rn["slave_url"],
                            "/api/slave/rdma/probe",
                            token=rn.get("token"),
                            json_body={
                                "device": dev,
                                "via": via,
                                "port": port,
                                "rank": rank,
                                "rounds": 16,
                                "seconds": 1,
                                "peer_port": port,
                            },
                            timeout=25.0,
                        )
                    except (httpx.HTTPError, ValueError) as exc:
                        return {"ok": False, "error": str(exc)[:160]}
                    return r.json() if r.status_code == 200 else {"ok": False, "error": r.text[:160]}

                with ThreadPoolExecutor(max_workers=2) as ex:
                    fa = ex.submit(run_side, a_id, link["a_device"], via_a, 0)
                    fb = ex.submit(run_side, b_id, link["b_device"], via_b, 1)
                    ra, rb = fa.result(), fb.result()
                if ra.get("ok") or rb.get("ok"):
                    lat = ra.get("latency_us") or rb.get("latency_us")
                    bw = max(ra.get("bandwidth_gbs") or 0.0, rb.get("bandwidth_gbs") or 0.0) or None
                    with studio._lock:
                        studio._data.setdefault("link_probes", {})[link["id"]] = {
                            "latency_us": lat,
                            "bandwidth_gbs": bw,
                            "probed_at": time.time(),
                            "ok": True,
                        }
                    studio._save_config()
        finally:
            studio._probe_running = False

    threading.Thread(target=worker, name="rdma-mesh-probe", daemon=True).start()
    return {"started": True}


def run_benchmark(studio,
    *,
    prompt_tokens: list[int] | None = None,
    decode_tokens: int = 128,
    draft: bool = True,
    label: str | None = None,
    engine_url: str | None = None,
) -> dict[str, Any]:
    if studio._bench_job and studio._bench_job.get("state") == "running":
        raise RuntimeError("A cluster benchmark is already running")

    dep = studio._active_deployment
    url = (engine_url or f"http://127.0.0.1:{(dep or {}).get('serve_port', 8080)}").rstrip("/")
    lens = prompt_tokens or [512, 2048]
    job_id = f"bench-{secrets.token_hex(4)}"
    topo_label = label or (
        f"{dep['world']}-node {dep['mode'].upper()} (TP={dep['tp']}×PP={dep['pp']})" if dep else "Active Cluster"
    )
    job: dict[str, Any] = {
        "id": job_id,
        "label": topo_label,
        "state": "running",
        "started_at": time.time(),
        "model": (dep or {}).get("model_name") or "unknown",
        "world": (dep or {}).get("world") or 1,
        "tp": (dep or {}).get("tp") or 1,
        "pp": (dep or {}).get("pp") or 1,
        "mode": (dep or {}).get("mode") or "single",
        "draft": draft,
        "decode_tokens": decode_tokens,
        "steps_total": len(lens),
        "steps_done": 0,
        "points": [],
        "error": None,
    }
    studio._bench_job = job

    def worker() -> None:
        try:
            bench_api_key = studio.master_rank.active_api_key or studio._configured_api_key()
            auth_headers = {"Authorization": f"Bearer {bench_api_key}"} if bench_api_key else {}
            # Identify served model id
            rm = httpx.get(f"{url}/v1/models", headers=auth_headers, timeout=5.0, trust_env=False)
            rm.raise_for_status()
            models_data = rm.json().get("data") or []
            model_id = models_data[0]["id"] if models_data else job["model"]
            job["model"] = model_id

            filler = "The Thunderbolt 5 RDMA fabric distributes tensor parallel matrix multiplications across unified Apple Silicon memory. "
            for idx, target_toks in enumerate(lens):
                # Unique nonce ensures cold prefill (zero prompt-cache hit!)
                nonce = f"Benchmark run {job_id} step {idx} nonce {secrets.token_hex(6)}. "
                words_needed = max(32, int(target_toks * 0.72))
                prompt_text = nonce + (filler * (words_needed // 14 + 2))
                body: dict[str, Any] = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt_text + "\nSummarize the architecture in detail."}],
                    "max_tokens": decode_tokens,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if not draft:
                    body["draft"] = False
                t_start = time.perf_counter()
                t_first: float | None = None
                prompt_tok_actual = target_toks
                comp_tok_actual = 0
                with httpx.stream(
                    "POST",
                    f"{url}/v1/chat/completions",
                    json=body,
                    headers=auth_headers,
                    timeout=300.0,
                    trust_env=False,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            ev = json.loads(payload)
                        except ValueError:
                            continue
                        if ev.get("usage"):
                            prompt_tok_actual = int(ev["usage"].get("prompt_tokens") or prompt_tok_actual)
                            comp_tok_actual = int(ev["usage"].get("completion_tokens") or comp_tok_actual)
                        choices = ev.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            if (delta.get("content") or delta.get("reasoning_content")) and t_first is None:
                                t_first = time.perf_counter()
                            if delta.get("content") or delta.get("reasoning_content"):
                                comp_tok_actual = max(comp_tok_actual, 1)
                t_end = time.perf_counter()
                ttft_s = max(0.001, (t_first or t_end) - t_start)
                decode_s = max(0.001, t_end - (t_first or t_start))
                prefill_tps = round(prompt_tok_actual / ttft_s, 1)
                decode_tps = round(max(1, comp_tok_actual - 1) / decode_s, 2) if comp_tok_actual > 1 else 0.0
                job["points"].append({
                    "target_prompt_tokens": target_toks,
                    "prompt_tokens": prompt_tok_actual,
                    "completion_tokens": comp_tok_actual,
                    "ttft_ms": round(ttft_s * 1000.0, 1),
                    "prefill_tps": prefill_tps,
                    "decode_tps": decode_tps,
                })
                job["steps_done"] = idx + 1

            if job["points"]:
                job["peak_prefill_tps"] = max(p["prefill_tps"] for p in job["points"])
                job["avg_decode_tps"] = round(sum(p["decode_tps"] for p in job["points"]) / len(job["points"]), 2)
            job["state"] = "done"
            job["finished_at"] = time.time()
            with studio._lock:
                studio._benchmarks.append(dict(job))
                studio._benchmarks = studio._benchmarks[-40:]
            studio._save_benchmarks()
        except Exception as exc:  # noqa: BLE001
            job["state"] = "failed"
            job["error"] = str(exc)[:260]
            job["finished_at"] = time.time()

    threading.Thread(target=worker, name="cluster-bench", daemon=True).start()
    return job


def record_external_benchmark(studio, record: dict[str, Any]) -> dict[str, Any]:
    """Store an externally measured CLI benchmark run so it renders in the Cluster Studio charts."""
    points = record.get("points") or []
    peak_pf = float(record.get("peak_prefill_tps") or 0.0)
    avg_dec = float(record.get("avg_decode_tps") or 0.0)
    if points and peak_pf <= 0.0:
        peak_pf = round(max(float(p.get("prefill_tps") or 0.0) for p in points), 1)
    if points and avg_dec <= 0.0:
        decs = [float(p.get("decode_tps") or 0.0) for p in points if float(p.get("decode_tps") or 0.0) > 0.0]
        if decs:
            avg_dec = round(sum(decs) / len(decs), 2)
    rec = {
        "id": record.get("id") or f"bench-{secrets.token_hex(4)}",
        "label": str(record.get("label") or "Benchmark"),
        "state": "done",
        "started_at": record.get("started_at") or time.time(),
        "finished_at": record.get("finished_at") or time.time(),
        "model": str(record.get("model") or "GLM-5.3"),
        "world": int(record.get("world") or 2),
        "tp": int(record.get("tp") or 2),
        "pp": int(record.get("pp") or 1),
        "mode": str(record.get("mode") or "tp"),
        "draft": bool(record.get("draft", True)),
        "points": points,
        "peak_prefill_tps": peak_pf,
        "avg_decode_tps": avg_dec,
    }
    with studio._lock:
        studio._benchmarks = [b for b in studio._benchmarks if b.get("id") != rec["id"]]
        studio._benchmarks.append(rec)
        studio._benchmarks = studio._benchmarks[-40:]
    studio._save_benchmarks()
    return rec
