"""Benchmark mode execution."""

from __future__ import annotations

import itertools
import math
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import bench, kl, quality, tooleval
from .bench_sampling import (
    summarize,
)
from .bench_stream import (
    Reply,
    drafts_off_confirmed,
    drafts_on_confirmed,
    measure,
    stream_once,
    text_digest,
    thought,
)


class BenchModes:
    def _prefill(self, run: dict, sizes: list[int], client: httpx.Client, base: str, headers: dict,
                 results: list[dict], off: dict) -> None:
        """One prompt per size, each opening with its own nonce so the prompt cache cannot serve the next one."""

        url = base + "/v1/chat/completions"
        filler = self.suites["passages"]["water"] + "\n\n"
        per_filler, window = None, None
        try:
            r = client.post(base + "/tokenize", json={"model": run["model"], "prompt": filler}, headers=headers)
            if r.status_code == 200:
                body = r.json()
                per_filler = body.get("count") or None
                window = body.get("max_model_len") if isinstance(body.get("max_model_len"), int) else None
        except (httpx.HTTPError, ValueError):
            pass
        if not per_filler:
            per_filler = max(1, len(filler) // 4)
            run["notes"].append("the engine has no /tokenize; prompt sizes estimated at four characters a token")
        if window:
            run["window"] = window
            kept = [n for n in sizes if n <= window - 256]
            if len(kept) < len(sizes):
                run["notes"].append(f"sizes above the served window ({window:,} tokens) left out")
            sizes = kept
        run["progress"]["total"] = len(sizes)
        question = "\n\nIn one word: which river does the town draw its water from?"
        spent = 0.0
        overhead: float | None = None
        for size in sizes:
            if self._cancel.is_set():
                raise InterruptedError
            if results and spent + bench.GROWTH * results[-1]["m"]["wall_s"] > bench.SWEEP_LIMIT_S:
                run["notes"].append(f"stopped before {size:,} tokens: the sweep would pass {int(bench.SWEEP_LIMIT_S)} s")
                break
            run["progress"]["current"] = f"{size:,} tokens"
            self.store.save_run(run)

            def once(size: int = size) -> tuple[Reply, dict]:
                nonce = uuid.uuid4().hex             # a new opening each time, so no repeat is served from the cache
                reps = max(1, (size - 64) // per_filler)
                content = (f"Reference {nonce}. Read this report, then answer the question after it.\n\n"
                           + filler * reps + question)
                body = {"model": run["model"], "messages": [{"role": "user", "content": content}], "max_tokens": 8,
                        "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
                        **off, **({} if run.get("drafts", True) else {"draft": False})}
                got = stream_once(client, url, headers, body, self._cancel)
                if got.error == "cancelled":
                    raise InterruptedError
                return got, measure(got)

            rep, m = once()
            if not rep.error and m["prefill_s"] is None:
                # no prefill time from this engine: the first token less a measured overhead, median of a few tries
                if overhead is None:
                    overhead = self._overhead(run, client, url, headers, off)
                tries = [(rep, m)] + [once() for _ in range(bench.prefill_repeats(size) - 1)]
                extra_wall = sum(t[1]["wall_s"] for t in tries)
                tries = [t for t in tries if not t[0].error and t[1]["ttft"]] or tries
                rep, m = sorted(tries, key=lambda t: t[1]["ttft"] or 0)[len(tries) // 2]
                m["repeats"] = len(tries)
                m["tries_wall_s"] = extra_wall                  # what the size really took, every try counted
                run["prefill_repeated"] = True       # energy then went to repeats and probes the sweep does not count
            if m["prefill_s"]:
                took = m["prefill_s"]
            elif m["ttft"] and overhead is not None:
                took = max(m["ttft"] - overhead, 1e-3)
                m["prefill_estimated"] = True
                m["prefill_weak"] = overhead > 0.3 * m["ttft"]       # the overhead is most of what was timed
            else:
                took = m["ttft"]
            m["prefill_tps"] = (m["prompt_tokens"] - (m["cached_tokens"] or 0)) / took if m["prompt_tokens"] and took else None
            if m["prompt_tokens"] and m["prompt_tokens"] < 0.8 * (size - 64) and not rep.error:
                run["notes"].append(f"{size:,} tokens: the engine read only {m['prompt_tokens']:,}; its context is "
                                    "smaller than the prompt and it cut it (Ollama: OLLAMA_CONTEXT_LENGTH)")
            r = {"suite": "prefill", "prompt": str(size), "size": size, "draft": True, "m": m, "error": rep.error,
                 "text": rep.text[:200]}
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] = len(results)
            run["summary"] = summarize(results)
            took_all = m.get("tries_wall_s") or m["wall_s"]
            spent += took_all
            run["progress"]["eta_s"] = bench.sweep_eta(took_all, len(sizes) - len(results), spent)
            run["progress"]["eta_at"] = time.time()
            if rep.error:
                run["notes"].append(f"stopped at {size:,} tokens: {rep.error}")
                break
            if m["cached_tokens"]:
                run["notes"].append(f"{size:,} tokens: {m['cached_tokens']:,} came from the prompt cache")

    def _tools(self, run: dict, base: str, key: str | None, results: list[dict]) -> None:
        """tool-eval-bench on the same engine; its progress events drive the run's, its scenarios become results."""

        def on_event(e: dict) -> None:
            if e.get("event") == "scenario_start":
                run["progress"]["current"] = f"{e.get('scenario_id')}: {e.get('title', '')}"
                if isinstance(e.get("total"), int):
                    run["progress"]["total"] = e["total"]
            elif e.get("event") == "scenario_result":
                # Growing multi-turn scenarios make early timings unsuitable for a remaining-time estimate.
                run["progress"]["done"] = (e.get("index") or 0) + 1

        t = tooleval.Run(self._tools_exe, base, key, run["model"], full=run.get("tools_full", False),
                         on_event=on_event)
        try:
            doc = t.wait(self._cancel)
        finally:
            t.close()
        run["tools"] = tooleval.summary(doc)
        run["thinking"] = {"sent": "--no-think (tool-eval-bench)", "off": True, "refused": []}
        for r in tooleval.results(doc, t.categories):
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
        run["progress"]["done"] = len(results)
        run["summary"] = summarize(results)

    def _kl(self, run: dict, work: list[tuple[dict, dict]], base: str, headers: dict, results: list[dict],
            off: dict) -> None:
        """The reference's replies first, then the candidate's (under load if asked), compared prompt by prompt."""

        url = base + "/v1/chat/completions"
        cand = kl.Target(url, headers, run["model"], dict(off))
        ref_at = getattr(self, "_kl_reference", None)
        if ref_at:
            # a fourth part: headers, such as another TensorFold Web's token
            r_base, r_key, r_model, *more = ref_at
            ref = kl.Target(r_base.rstrip("/") + "/v1/chat/completions",
                            {**({"Authorization": f"Bearer {r_key}"} if r_key else {}), **(more[0] if more else {})},
                            r_model, dict(off))
        else:
            ref = kl.Target(url, headers, run["model"], {**off, "draft": False})
        run["progress"]["total"] = 2 * len(work)
        replies: list[list[kl.Step]] = []
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(600.0, connect=5.0)) as c:
            for side, target in (("reference", ref), ("candidate", cand)):
                load = (kl.Load(cand, run.get("kl_load") or 0, [{"role": "user", "content": "Write a long story about "
                        "a lighthouse keeper."}]) if side == "candidate" and run.get("kl_load") else None)
                if load:
                    load.__enter__()
                try:
                    for i, (suite, prompt) in enumerate(work):
                        if self._cancel.is_set():
                            raise InterruptedError
                        run["progress"]["current"] = f"{side}: {suite['name']}: {prompt['id']}"
                        self.store.save_run(run)
                        try:
                            steps = (kl.score(c, target, prompt["text"], cancel=self._cancel) if "text" in prompt else
                                     kl.ask(c, target, prompt["messages"], min(prompt.get("max_tokens", 300), kl.MAX_TOKENS),
                                            cancel=self._cancel))
                        except RuntimeError as exc:
                            raise RuntimeError(f"the {side} engine: {exc}") from exc
                        if side == "reference":
                            replies.append(steps)
                            if ref_at is None:            # the same engine asked for drafts off: did it say so?
                                confirmed = drafts_off_confirmed(getattr(steps, "runtime", {}) or {})
                                ref = run["kl_reference"]
                                ref["drafts_off"] = confirmed if ref.get("drafts_off") is None else \
                                    ref["drafts_off"] and confirmed
                        else:
                            p = kl.pair_text(replies[i], steps) if "text" in prompt else kl.pair(replies[i], steps)
                            p["prompt"] = f"{suite['id']}/{prompt['id']}"
                            r = {"suite": suite["id"], "prompt": prompt["id"], "draft": True, "error": None,
                                 "m": {"tokens": p["tokens"], "tps": None, "ttft": None, "ttft_answer": None,
                                       "prefill_s": None, "gap_p50": None, "gap_p95": None, "runtime": {}},
                                 "kl": {k: p[k] for k in ("positions", "agree", "equal", "differs_at", "identical", "worst",
                                                          "width", "unscored")},
                                 "kl_values": p["kl"]}
                            results.append(r)
                            self.store.add_result(run["id"], len(results) - 1, r)
                        run["progress"]["done"] += 1
                finally:
                    if load:
                        load.stop(wait=2.0 if self._cancel.is_set() else 320.0)
                        run["kl_load_done"] = {"ok": load.ok, "failed": load.failed, "late": load.late,
                                               "full_share": load.full_share, "last_error": load.last_error}
                if load and not (load.ok or load.late):         # refused throughout
                    raise RuntimeError(f"the background load never got through ({load.failed} refused: "
                                       f"{load.last_error}); the run would not say what it claims")
                if load:
                    # Load measurements need completed replies and requests in flight for most of the interval.
                    share = load.full_share or 0.0
                    run["kl_load_verified"] = bool(load.ok) and share >= bench.LOAD_SHARE
                    if not run["kl_load_verified"]:
                        run["notes"].append(f"the background load is not established: {load.ok} replies during the "
                                            f"measurement, all {run['kl_load']} requests in flight {share * 100:.0f}% "
                                            "of the time; this run does not show the candidate under load")
        run["kl"] = kl.summary([{**r["kl"], "kl": r["kl_values"], "prompt": f"{r['suite']}/{r['prompt']}",
                                 "tokens": r["m"]["tokens"], "requests": 2} for r in results])

    def _quality(self, run: dict, work: list[dict], client: httpx.Client, url: str, headers: dict,
                 results: list[dict], off: dict) -> None:
        """Each item once, scored strictly; the reply and what was read from it kept with the result."""

        for it in work:
            if self._cancel.is_set():
                raise InterruptedError
            run["progress"]["current"] = f"{run['suite_names'].get(it['category'], it['category'])}: {it['id']}"
            self.store.save_run(run)
            body = {"model": run["model"], "messages": it["messages"], "max_tokens": it["max_tokens"],
                    "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True}, **off,
                    **({} if run.get("drafts", True) else {"draft": False})}
            rep = stream_once(client, url, headers, body, self._cancel)
            if rep.error == "cancelled":
                raise InterruptedError
            # a request that failed got no answer to grade: unscored, kept out of the share right, counted apart
            passed, read = (None, rep.error) if rep.error else quality.score(it["category"], it["item"], rep.text)
            if not passed and rep.finish == "length":       # the reply did not end: said apart from a wrong answer
                read = f"cut at {it['max_tokens']} tokens ({read})"
            r = {"suite": it["category"], "prompt": it["id"], "draft": True, "m": measure(rep), "error": rep.error,
                 "text": rep.text[:4000], "text_sha": text_digest(rep.text), "passed": passed, "read": read,
                 "expected": it["item"].get("answer")}
            results.append(r)
            self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] = len(results)
            run["summary"] = summarize(results)
            run["progress"]["eta_s"] = statistics.mean(x["m"]["wall_s"] for x in results) * (len(work) - len(results))
            run["progress"]["eta_at"] = time.time()

    def _load(self, run: dict, levels: list[int], work: list[tuple[dict, dict]], client: httpx.Client, url: str,
              headers: dict, results: list[dict], off: dict) -> None:

        turn = itertools.cycle(work)
        spent = 0.0
        for c in levels:
            if self._cancel.is_set():
                raise InterruptedError
            last = [r for r in results if r.get("concurrency") == c // 2]
            if last and spent + 2 * max(r["wave_s"] for r in last) * math.ceil(bench.LOAD_MIN_REQUESTS / c) > bench.LOAD_LIMIT_S:
                run["notes"].append(f"stopped before {c} at once: the sweep would pass {int(bench.LOAD_LIMIT_S)} s")
                break
            run["progress"]["current"] = f"{c} at once"
            self.store.save_run(run)
            level: list[dict] = []
            for wave in range(math.ceil(bench.LOAD_MIN_REQUESTS / c)):
                if self._cancel.is_set():
                    raise InterruptedError
                bodies, asked = [], []
                for _ in range(c):
                    suite, prompt = next(turn)
                    asked.append(f"{suite['id']}/{prompt['id']}")
                    bodies.append({"model": run["model"], "messages": prompt["messages"], "max_tokens": bench.LOAD_TOKENS,
                                   "temperature": 0.0, "top_p": 1.0, "stream": True,
                                   "stream_options": {"include_usage": True}, **off,
                                   **({} if run.get("drafts", True) else {"draft": False})})
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    replies = list(pool.map(lambda b: stream_once(client, url, headers, b, self._cancel), bodies))
                wall = time.perf_counter() - t0
                spent += wall
                if any(r.error == "cancelled" for r in replies):
                    raise InterruptedError
                for i, rep in enumerate(replies):
                    r = {"suite": "load", "prompt": f"{c}x{wave}.{i}", "asked": asked[i], "concurrency": c, "wave": wave,
                         "wave_s": wall,
                         "draft": True, "m": measure(rep), "error": rep.error, "text": rep.text[:200]}
                    results.append(r)
                    level.append(r)
                    self.store.add_result(run["id"], len(results) - 1, r)
            run["progress"]["done"] += 1
            run["summary"] = summarize(results)
            failed = sum(1 for r in level if r["error"])
            if failed * 2 >= len(level):
                run["notes"].append(f"stopped at {c} at once: {failed} of {len(level)} requests failed "
                                    f"({next(r['error'] for r in level if r['error'])})")
                break
            waits = [r["m"]["ttft_client"] for r in level if r["m"].get("ttft_client")]
            if waits and max(waits) > bench.LOAD_TTFT_LIMIT_S:
                run["notes"].append(f"stopped at {c} at once: the slowest request waited {max(waits):.0f} s for its "
                                    "first token")
                break

    def _overhead(self, run: dict, client: httpx.Client, url: str, headers: dict, off: dict) -> float:
        """The first-token time of a nearly empty prompt: what an engine spends whatever the prompt, median of three."""

        times = []
        for _ in range(3):
            body = {"model": run["model"], "messages": [{"role": "user", "content": f"Ref {uuid.uuid4().hex[:8]}. Say ok."}],
                    "max_tokens": 1, "temperature": 0.0, "stream": True, **off}
            got = stream_once(client, url, headers, body, self._cancel)
            if got.error == "cancelled":
                raise InterruptedError
            if got.first_at:
                times.append(got.first_at - got.sent_at)
        value = statistics.median(times) if times else 0.0
        run["prefill_overhead_s"] = value
        run["notes"].append(f"the engine gives no prefill time: prefill is estimated as the first token less "
                            f"{value * 1000:.0f} ms of overhead (three near-empty prompts), median of a few tries per size")
        return value

    def _warm(self, run: dict, client: httpx.Client, url: str, headers: dict) -> dict:

        run["progress"]["current"], run["progress"]["phase"] = "warm-up", "warm-up"
        warm = {"model": run["model"], "messages": [{"role": "user", "content": "Count from 1 to 40."}],
                "max_tokens": bench.PROTOCOL["warmup_tokens"], "temperature": 0.0, "top_p": 1.0, "stream": True}
        accepted: tuple[str, dict] | None = None
        refused: list[str] = []
        for name, fields in bench.THINKING_OFF:
            rep = stream_once(client, url, headers, {**warm, **fields}, self._cancel)
            if rep.error == "cancelled":
                raise InterruptedError
            if rep.error and rep.error.startswith("400"):
                refused.append(name)
                continue
            if rep.error:
                raise RuntimeError(f"warm-up failed: {rep.error}")
            if not thought(rep):
                run["thinking"] = {"sent": name, "off": True, "refused": refused}
                return fields
            accepted = accepted or (name, fields)
        if accepted is not None:
            run["thinking"] = {"sent": accepted[0], "off": False, "refused": refused}
            run["notes"].append("thinking stayed on whichever way it was turned off; the figures include reasoning tokens")
            return accepted[1]
        # every way refused: the request goes without the fields, and the receipt says thinking was not controlled
        rep = stream_once(client, url, headers, warm, self._cancel)
        if rep.error == "cancelled":
            raise InterruptedError
        if rep.error:
            raise RuntimeError(f"warm-up failed: {rep.error}")
        run["thinking"] = {"sent": None, "off": not thought(rep), "refused": refused}
        run["notes"].append("the engine refused every way of turning thinking off; requests go without them")
        return {}

    def _sweep(self, run: dict, work: list[tuple[dict, dict]], client: httpx.Client, url: str, headers: dict,
               results: list[dict], off: dict) -> None:
        for suite, prompt in work:
            first: Reply | None = None
            for draft in ((True, False) if run["exactness"] else (True,)):
                if self._cancel.is_set():
                    raise InterruptedError
                run["progress"]["current"] = f"{suite['name']}: {prompt['id']}" + ("" if draft else ", drafts off")
                self.store.save_run(run)
                body = {"model": run["model"], "messages": prompt["messages"], "max_tokens": prompt.get("max_tokens", 300),
                        "temperature": 0.0, "top_p": 1.0, "stream": True, "stream_options": {"include_usage": True},
                        **off}
                if not draft or not run.get("drafts", True):
                    body["draft"] = False
                rep = stream_once(client, url, headers, body, self._cancel)
                if rep.error == "cancelled":
                    raise InterruptedError
                r = {"suite": suite["id"], "prompt": prompt["id"], "draft": draft, "m": measure(rep),
                     "error": rep.error, "text": rep.text[:4000], "text_sha": text_digest(rep.text)}
                if thought(rep) and (run.get("thinking") or {}).get("off"):
                    run["notes"].append(f"{prompt['id']}: the model reasoned although thinking was turned off")
                if draft:
                    first = rep
                cached = (rep.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                note = ("some drafts-off replies read their prompt from the cache the drafts-on request left: their "
                        "first token and prefill are not comparable with drafts on (speeds still are)")
                if not draft and cached > 0 and note not in run["notes"]:
                    run["notes"].append(note)
                if not draft and first is not None and not first.error and not rep.error:
                    # Exactness needs an active drafted arm and a confirmed plain arm, not merely equal replies.
                    off_known = drafts_off_confirmed(rep.runtime) and drafts_on_confirmed(first.runtime)
                    if rep.runtime.get("drafts") is True:
                        run["notes"].append(f"{prompt['id']}: the engine reported drafts on although draft=false was sent")
                    if not drafts_on_confirmed(first.runtime):
                        note = "the engine does not say drafts ran on the first reply: the exactness check is not counted"
                        if note not in run["notes"]:
                            run["notes"].append(note)
                    if off_known:
                        sha_a, sha_b = first.runtime.get("token_sha"), rep.runtime.get("token_sha")
                        r["same"] = (sha_a == sha_b) if sha_a and sha_b else (first.text == rep.text and bool(rep.text))
                        r["compared_by"] = "token_sha" if sha_a and sha_b else "text"
                    else:
                        r["same"], r["compared_by"] = None, "unverified"
                        note = "the engine does not say whether drafts were off: the exactness check is not counted"
                        if note not in run["notes"]:
                            run["notes"].append(note)
                results.append(r)
                self.store.add_result(run["id"], len(results) - 1, r)
                run["progress"]["done"] = len(results)
                run["summary"] = summarize(results)
                run["progress"]["eta_s"] = bench.suites_eta(results, work, run["exactness"])
                run["progress"]["eta_at"] = time.time()
