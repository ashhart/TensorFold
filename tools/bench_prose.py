#!/usr/bin/env python3
"""Model-agnostic natural-prose benchmark client with verified cold-cache, equality, logprob and long-context checks."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_prose_loss
import bench_stream
from bench_evidence import count as valid_count, token_equal
from bench_prose_loss import CLEAN_FINISH, LOSS_CASES, compare_prose_loss, sha256_token_ids, valid_token_ids

DISTINCT_PROSE_8 = [
    ("war_and_peace", 12000, 9500, "Analyze how Tolstoy contrasts salon artifice with genuine human emotion in this opening scene, then continue the narrative tone for two paragraphs."),
    ("monte_cristo", 15000, 9500, "Examine Edmond Dantes' character and relationship with shipboard authority in this passage, and write a detailed critical commentary."),
    ("les_miserables", 18000, 9500, "Discuss Victor Hugo's portrait of Bishop Myriel's moral philosophy in this excerpt and trace its narrative consequences."),
    ("don_quixote", 14000, 9500, "Analyze Cervantes' use of irony and chivalric obsession in this passage and explain how the prose rhythm builds comedic tension."),
    ("moby_dick", 22000, 9500, "Examine Ishmael's reflections on the sea and solitude in this excerpt, focusing on Melville's imagery and philosophical undertones."),
    ("anna_karenina", 16000, 9500, "Analyze how Tolstoy establishes domestic conflict and psychological realism in the Oblonsky household in this scene."),
    ("crime_and_punishment", 14000, 9500, "Examine how Dostoevsky conveys Raskolnikov's feverish alienation through sensory detail in St. Petersburg."),
    ("ulysses", 14000, 9500, "Analyze Joyce's shifts between external action and interior monologue on the Martello tower in this opening passage."),
]

STREAM_SOURCES_500K = [
    ("Stream0_WarAndPeace", ["war_and_peace"], ("FOLIO-WP-1812", "FOLIO-WP-4409", "FOLIO-WP-8831"), "Summarize the closing philosophical reflections in the manuscript"),
    ("Stream1_LesMiserables", ["les_miserables"], ("FOLIO-LM-1832", "FOLIO-LM-5510", "FOLIO-LM-9047"), "Describe how the narrative resolves Jean Valjean's journey"),
    ("Stream2_MonteCristo", ["monte_cristo"], ("FOLIO-MC-1815", "FOLIO-MC-6128", "FOLIO-MC-9723"), "Explain how Edmond Dantes' final philosophy is presented"),
    ("Stream3_DonQuixote", ["don_quixote", "vanity_fair"], ("FOLIO-DQ-1605", "FOLIO-DQ-3384", "FOLIO-DQ-7762"), "Describe the contrast in prose style near the end of this volume"),
    ("Stream4_Karenina_Crime", ["anna_karenina", "crime_and_punishment"], ("FOLIO-AK-1877", "FOLIO-AK-4920", "FOLIO-AK-8156"), "Identify the central moral dilemma in the final chapters above"),
    ("Stream5_Copperfield_Moby", ["david_copperfield", "moby_dick"], ("FOLIO-DC-1850", "FOLIO-DC-5291", "FOLIO-DC-9408"), "Characterize the narrator's voice in the concluding section"),
    ("Stream6_Dorrit_Ulysses", ["little_dorrit", "ulysses"], ("FOLIO-LD-1857", "FOLIO-LD-6634", "FOLIO-LD-8519"), "Analyze the stylistic shift in the final portion of this manuscript"),
    ("Stream7_Dickens_Trilogy", ["our_mutual_friend", "nicholas_nickleby", "martin_chuzzlewit"], ("FOLIO-MF-1865", "FOLIO-MF-4073", "FOLIO-MF-7295"), "Summarize the social critique running through the closing chapters"),
]


def strip_gutenberg(raw: str) -> str:
    """Strip Project Gutenberg header and footer so books share no boilerplate prefix."""
    m_start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, flags=re.I | re.S)
    if m_start:
        raw = raw[m_start.end():]
    elif "PROJECT GUTENBERG" in raw[:4000].upper():
        raw = raw[4000:]
    m_end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", raw, flags=re.I)
    if m_end:
        raw = raw[:m_end.start()]
    return raw.strip()


def load_book(prose_dir: str, name: str) -> str:
    """Read one plain-text book from prose_dir and strip any Gutenberg wrapper."""
    with open(os.path.join(prose_dir, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return strip_gutenberg(f.read())


def post_json(url: str, payload: dict, timeout: int = 3600) -> dict:
    """POST JSON payload and return the decoded JSON object."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sha256_text(text: str) -> str:
    """Return lowercase 64-hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_usage(usage, complete: bool) -> dict:
    """Validate usage counters; only explicit cached_tokens == 0 <= prompt_tokens qualifies as cold."""
    u = usage if isinstance(usage, dict) else {}
    details = u.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    prompt = valid_count(u.get("prompt_tokens"))
    cached = valid_count(details.get("cached_tokens"))
    tokens = valid_count(u.get("completion_tokens"))
    valid_cache = prompt is not None and cached is not None and cached <= prompt
    if not complete or not valid_cache:
        state = "unverified"
    elif cached == 0:
        state = "cold"
    else:
        state = "reused"
    return {"prompt_tokens": prompt, "cached_tokens": cached, "tokens": tokens, "cache_state": state}


def pair_token_equal(left: dict, right: dict) -> bool | None:
    """Compare complete token IDs when both sides have them, else full 64-hex token_sha256."""
    if not left.get("accepted") or not right.get("accepted"):
        return None
    l_ids, r_ids = left.get("token_ids"), right.get("token_ids")
    if isinstance(l_ids, list) and isinstance(r_ids, list):
        return l_ids == r_ids
    return token_equal(left, right)


def pair_output_equal(left: dict, right: dict) -> bool | None:
    """Compare full 64-hex output_sha256 across two accepted replies."""
    if not left.get("accepted") or not right.get("accepted"):
        return None
    a, b = left.get("output_sha256"), right.get("output_sha256")
    return None if not a or not b else (a == b)


def stream_chat(url: str, body: dict, timeout: int = 600) -> dict:
    """Stream one SSE chat completion using monotonic timing and strict completion/output validation."""
    payload = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    sent = time.perf_counter()
    reply = bench_stream.Collector(sent)
    chunk_token_ids: list[int] = []
    error = None

    def tap_lines(resp):
        for raw in resp:
            if b"token_ids" in raw:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:"):
                    try:
                        chunk = json.loads(line[5:])
                    except ValueError:
                        chunk = None
                    if isinstance(chunk, dict):
                        for ch in chunk.get("choices") or []:
                            ids = ch.get("token_ids") or (ch.get("delta") or {}).get("token_ids")
                            if isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids):
                                chunk_token_ids.extend(ids)
            yield raw

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply.read(tap_lines(resp))
    except Exception as exc:  # noqa: BLE001 - stream failures are reported in the receipt, not hidden
        error = type(exc).__name__

    if error is None and reply.end == "error":
        error = "server_stream_error"
    complete = error is None and reply.end == "done"
    if error is None and not complete:
        error = "incomplete_stream"
    finish_reasons = [reply.finish[i] for i in sorted(reply.finish)]
    clean_finish = bool(finish_reasons) and all(r in CLEAN_FINISH for r in finish_reasons)
    observed = reply.evidence.finish(reply.usage, reply.runtime, complete and clean_finish)
    has_output = reply.evidence.observed_output and bool(reply.pieces)
    tokens, prompt_tokens = observed["tokens"], observed["prompt_tokens"]
    accepted = complete and clean_finish and has_output and prompt_tokens is not None and tokens is not None and tokens > 0
    if error is None and not clean_finish:
        error = "unclean_finish_reason"
    elif error is None and not has_output:
        error = "no_output"
    elif error is None and not accepted:
        error = "invalid_usage"

    raw_ids = reply.runtime.get("token_ids") if isinstance(reply.runtime, dict) else None
    if raw_ids is None and chunk_token_ids:
        raw_ids = chunk_token_ids
    token_ids = valid_token_ids(raw_ids, tokens) if accepted else None
    token_sha256 = observed["token_sha256"] or (sha256_token_ids(token_ids) if token_ids is not None else None)
    first = reply.pieces[0][0] if (accepted and reply.pieces) else None
    last = reply.pieces[-1][0] if (accepted and reply.pieces) else None
    ttft_s = (first - sent) if first is not None else None
    decode_tps = ((tokens - 1) / (last - first)) if (accepted and tokens > 1 and last > first) else None

    return {
        "accepted": accepted, "complete": complete and clean_finish, "completion": reply.completion(),
        "finish_reasons": finish_reasons, "error": error,
        "prompt_tokens": prompt_tokens, "cached_tokens": observed["cached_tokens"],
        "cache_state": observed["cache_state"], "tokens": tokens, "sent": sent, "first": first, "last": last,
        "ttft_s": ttft_s, "decode_tps": decode_tps, "token_ids": token_ids,
        "token_sha256": token_sha256 if accepted else None,
        "output_sha256": observed["output_sha256"] if accepted else None,
    }


def _validate_decode_args(levels, gen_tokens: int, pool_len: int) -> list[int]:
    """Validate prose-decode counts before issuing any HTTP requests."""
    if type(gen_tokens) is not int or gen_tokens < 1:
        raise ValueError("gen_tokens must be a positive integer")
    vals = list(levels)
    if not vals or any(type(n) is not int or n < 1 or n > pool_len for n in vals):
        raise ValueError(f"levels must be integers between 1 and {pool_len}")
    return vals


def _decode_overlap(rows: list[dict]) -> tuple[bool | None, int]:
    """Return (overlapped, max_concurrent) across accepted streams' [first, last] decode intervals."""
    spans = sorted((r["first"], r["last"]) for r in rows if r.get("first") is not None and r.get("last") is not None)
    if len(spans) < 2:
        return None, len(spans)
    overlapped = any(b[0] < a[1] for a, b in zip(spans, spans[1:]))
    events = [(lo, 1) for lo, _ in spans] + [(hi, -1) for _, hi in spans]
    # Process end (-1) before start (+1) at equal timestamps so abutting intervals do not count as concurrent.
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return overlapped, peak


def run_prose_decode(
    base_url: str,
    prose_dir: str,
    model: str = "model",
    levels=(1, 4, 8),
    gen_tokens: int = 256,
    seed: int = 1234,
    excerpt_specs=(("350tok_prose", 1400), ("2.5Ktok_chapter", 9500)),
    sources=None,
) -> dict:
    """Run paired drafted/plain distinct-prose decode with identical prompts, seeds, and cold-cache verification."""
    pool = sources if sources is not None else DISTINCT_PROSE_8
    checked_levels = _validate_decode_args(levels, gen_tokens, len(pool))
    print("\n=== REALISTIC DISTINCT-PROSE MULTI-STREAM DECODE BENCHMARK (SSE TTFT + PURE DECODE) ===", flush=True)
    report = {"mode": "prose-decode", "seed": seed, "gen_tokens": gen_tokens, "pairs": [], "qualified": True}

    for tag, excerpt_chars in excerpt_specs:
        prompts = []
        for i, (book, off, _, question) in enumerate(pool):
            raw_book = load_book(prose_dir, book)
            if not raw_book.strip():
                raise ValueError(f"empty source book for {book}")
            txt = raw_book[off : off + excerpt_chars] if len(raw_book) > off else raw_book[:excerpt_chars]
            prompt_text = f"[Excerpt {i + 1} — {book}]\n\n{txt}\n\nTask: {question}"
            prompts.append((book, prompt_text, seed + i, sha256_text(prompt_text)))
        print(f"  --- Workload: {tag} ({len(prompts)} distinct novels) ---", flush=True)

        for n in checked_levels:
            arms = {}
            for draft_on in (True, False):
                def one(idx, _draft=draft_on):
                    book, prompt_text, req_seed, prompt_sha = prompts[idx]
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "chat_template_kwargs": {"enable_thinking": False},
                        "temperature": 0,
                        "seed": req_seed,
                        "max_tokens": gen_tokens,
                    }
                    if not _draft:
                        body["draft"] = False
                    row = stream_chat(f"{base_url}/v1/chat/completions", body)
                    row.update(book=book, seed=req_seed, prompt_sha256=prompt_sha)
                    return row

                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                    rows = [f.result() for f in [ex.submit(one, i) for i in range(n)]]

                all_accepted = all(r["accepted"] for r in rows)
                all_cold = all(r["cache_state"] == "cold" for r in rows)
                all_timed = all_accepted and all(r["decode_tps"] is not None for r in rows)
                overlapped, max_concurrent = _decode_overlap(rows) if all_accepted else (None, 0)
                serialized = (n > 1) and (overlapped is False)
                timing_row = None
                if all_accepted and all_cold:
                    tot_prompt = sum(r["prompt_tokens"] for r in rows)
                    tot_gen = sum(r["tokens"] for r in rows)
                    start_t = min(r["sent"] for r in rows)
                    first_tok_t = min(r["first"] for r in rows)
                    last_tok_t = max(r["last"] for r in rows)
                    shared_span = last_tok_t - first_tok_t
                    agg_dec_tps = (sum(r["tokens"] - 1 for r in rows) / shared_span) if (all_timed and shared_span > 0) else None
                    per_med_tps = statistics.median(r["decode_tps"] for r in rows) if all_timed else None
                    timing_row = {
                        "prompt_tokens": tot_prompt, "cached_tokens": 0, "completion_tokens": tot_gen,
                        "decode_tps": agg_dec_tps, "per_stream_median_tps": per_med_tps,
                        "shared_decode_s": shared_span if shared_span > 0 else None,
                        "overlapped": overlapped, "max_concurrent_decoding": max_concurrent,
                        "serialized": serialized,
                        "e2e_tps": (tot_gen / (last_tok_t - start_t)) if last_tok_t > start_t else None,
                        "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
                    }
                mode_key, mode_s = ("draft", "DFly2") if draft_on else ("plain", "Plain")
                arms[mode_key] = {"rows": rows, "all_accepted": all_accepted,
                                  "cold_verified": all_accepted and all_cold,
                                  "concurrent_verified": (n == 1) or (overlapped is True),
                                  "overlapped": overlapped, "serialized": serialized, "timing": timing_row}
                if timing_row is not None:
                    dec_s = f"{timing_row['decode_tps']:6.2f}" if timing_row["decode_tps"] is not None else "  n/a "
                    e2e_s = f"{timing_row['e2e_tps']:6.2f}" if timing_row["e2e_tps"] is not None else "  n/a "
                    queue_note = " [SERIALIZED/QUEUED]" if serialized else ""
                    print(f"    [{n}s {mode_s:5s}] prompt={timing_row['prompt_tokens']:,} (cached=0, cold) | "
                          f"decode={dec_s} tok/s | e2e={e2e_s} tok/s | median_TTFT={timing_row['median_ttft_s']:.2f}s{queue_note}", flush=True)
                else:
                    reasons = sorted({r["error"] or f"cache_{r['cache_state']}" for r in rows if not r["accepted"] or r["cache_state"] != "cold"})
                    print(f"    [{n}s {mode_s:5s}] REJECTED (no timing row): {', '.join(reasons)}", flush=True)

            d_rows, p_rows = arms["draft"]["rows"], arms["plain"]["rows"]
            prompts_identical = all(d["prompt_sha256"] == p["prompt_sha256"] for d, p in zip(d_rows, p_rows))
            seeds_identical = all(d["seed"] == p["seed"] for d, p in zip(d_rows, p_rows))
            tok_eq = [pair_token_equal(d, p) for d, p in zip(d_rows, p_rows)]
            out_eq = [pair_output_equal(d, p) for d, p in zip(d_rows, p_rows)]
            pair_qualified = (arms["draft"]["cold_verified"] and arms["plain"]["cold_verified"]
                              and arms["draft"]["concurrent_verified"] and arms["plain"]["concurrent_verified"]
                              and prompts_identical and seeds_identical
                              and all(v is True for v in tok_eq) and all(v is True for v in out_eq))
            if not pair_qualified:
                report["qualified"] = False
            report["pairs"].append({
                "workload": tag, "streams": n, "prompts_identical": prompts_identical,
                "seeds_identical": seeds_identical, "token_equal": tok_eq, "output_equal": out_eq,
                "draft": arms["draft"], "plain": arms["plain"], "qualified": pair_qualified,
            })
    return report


def dump_prose_loss(base_url: str, prose_dir: str, out_json: str, model: str = "model",
                    cases=None, tf_tokens: int = 48, tf_file: str = "") -> dict:
    """Delegate prose KV loss dumping to bench_prose_loss."""
    return bench_prose_loss.dump_prose_loss(
        post_json, load_book, parse_usage, base_url, prose_dir, out_json,
        model=model, cases=cases, tf_tokens=tf_tokens, tf_file=tf_file,
    )


def build_stream_manuscript(prose_dir: str, idx: int, target_tokens: int, chars_per_tok: float, sources=None) -> tuple[str, tuple[str, str, str], str, bool]:
    """Assemble one distinct archival manuscript and flag whether source text fell short of target_chars."""
    pool = sources if sources is not None else STREAM_SOURCES_500K
    name, books, codes, lit_q = pool[idx]
    joined = "\n\n".join(load_book(prose_dir, b) for b in books)
    target_chars = int(target_tokens * chars_per_tok)
    source_undersized = len(joined) < target_chars
    body = joined[:target_chars]
    p10, p50, p90 = int(len(body) * 0.10), int(len(body) * 0.50), int(len(body) * 0.90)
    manuscript = (
        f"Archival Volume #{idx + 1} ({name}):\n\n" + body[:p10]
        + f"\n\n[Archival Margin Note Alpha for {name}: Early catalog marker is {codes[0]}.]\n\n" + body[p10:p50]
        + f"\n\n[Archival Margin Note Beta for {name}: Mid-volume catalog marker is {codes[1]}.]\n\n" + body[p50:p90]
        + f"\n\n[Archival Margin Note Gamma for {name}: Late catalog marker is {codes[2]}.]\n\n" + body[p90:]
        + f"\n\nFinal Instructions for {name}: First list the three exact catalog markers (Alpha, Beta, Gamma) recorded in this volume, then in two sentences: {lit_q}."
    )
    return name, codes, manuscript, source_undersized


def calibrate_stream_cpts(base_url: str, prose_dir: str, n_streams: int, model: str = "model", sources=None) -> list[float]:
    """Estimate characters per token for each stream's source corpus; reject empty sources cleanly."""
    pool = sources if sources is not None else STREAM_SOURCES_500K
    if type(n_streams) is not int or n_streams < 1 or n_streams > len(pool):
        raise ValueError(f"n_streams must be between 1 and {len(pool)}")
    cpts = []
    for idx in range(n_streams):
        name, books, _, _ = pool[idx]
        joined = "\n\n".join(load_book(prose_dir, b) for b in books)
        if not joined.strip():
            raise ValueError(f"calibration failed for {name}: empty source text")
        n = min(len(joined), 2_300_000)
        if n > 60000:
            sample = joined[n // 6 : n // 6 + 20000] + "\n" + joined[n // 2 : n // 2 + 20000] + "\n" + joined[(5 * n) // 6 : (5 * n) // 6 + 20000]
        else:
            sample = joined[: min(n, 4000)]
        if not sample:
            raise ValueError(f"calibration failed for {name}: empty sample")
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": sample}],
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "max_tokens": 1,
        })
        ptok = valid_count((res.get("usage") or {}).get("prompt_tokens"))
        if not ptok:
            raise ValueError(f"calibration failed for {name}: invalid prompt_tokens")
        cpt = len(sample) / ptok
        if cpt <= 0.0:
            raise ValueError(f"calibration failed for {name}: non-positive chars/tok")
        cpts.append(cpt)
        print(f"  Calibrated {name:24s}: {cpt:.4f} chars/tok (available={int(len(joined) / cpt):,} tok)", flush=True)
    return cpts


def _observed_kv_residency(health: dict) -> int | None:
    """Return observed engine KV token residency from /health only when explicitly reported."""
    if not isinstance(health, dict):
        return None
    kv = health.get("kv_cache") if isinstance(health.get("kv_cache"), dict) else {}
    for candidate in (health.get("kv_resident_tokens"), kv.get("used_tokens"), kv.get("resident_tokens")):
        val = valid_count(candidate)
        if val is not None:
            return val
    return None


def run_distinct_long(base_url: str, prose_dir: str, target_tokens: int, n_streams: int, model: str = "model",
                      min_prompt_ratio: float = 0.95, sources=None) -> dict:
    """Run N distinct long-context prose streams with separate submitted/completed totals and undersized-cell checks."""
    pool = sources if sources is not None else STREAM_SOURCES_500K
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    if type(n_streams) is not int or n_streams < 1 or n_streams > len(pool):
        raise ValueError(f"n_streams must be between 1 and {len(pool)}")
    if type(min_prompt_ratio) not in (int, float) or not math.isfinite(min_prompt_ratio) or not (0.0 < min_prompt_ratio <= 1.0):
        raise ValueError("min_prompt_ratio must be finite and in (0, 1]")
    print(f"\n=== {n_streams}x TARGET {target_tokens:,}-TOKEN DISTINCT REAL-PROSE CONCURRENT BENCHMARK ===", flush=True)
    cpts = calibrate_stream_cpts(base_url, prose_dir, n_streams, model=model, sources=pool)
    min_prompt_tokens = math.ceil(target_tokens * min_prompt_ratio)
    payloads = [build_stream_manuscript(prose_dir, i, target_tokens, cpts[i], sources=pool) for i in range(n_streams)]
    est_tokens = [int(len(ms) / cpts[i]) for i, (_, _, ms, _) in enumerate(payloads)]

    for i, (name, codes, ms, src_short) in enumerate(payloads):
        tag = " [UNDERSIZED SOURCE]" if src_short else ""
        print(f"  Prepared {name:24s}: {len(ms):,} chars (~{est_tokens[i]:,} est tok), markers={codes}{tag}", flush=True)

    def run_one(i):
        name, codes, ms, src_short = payloads[i]
        t0 = time.perf_counter()
        try:
            res = post_json(f"{base_url}/v1/chat/completions", {
                "model": model, "messages": [{"role": "user", "content": ms}],
                "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "seed": 1234 + i, "max_tokens": 96,
            }, timeout=7200)
        except Exception as exc:  # noqa: BLE001 - capture per-stream failures without losing submitted totals
            return {"name": name, "accepted": False, "error": type(exc).__name__, "est_prompt_tokens": est_tokens[i],
                    "prompt_tokens": None, "cached_tokens": None, "cache_state": "unverified", "tokens": None,
                    "elapsed_s": time.perf_counter() - t0, "hits": 0, "undersized": src_short, "source_undersized": src_short}
        el = time.perf_counter() - t0
        if res.get("error"):
            return {"name": name, "accepted": False, "error": "server_error", "est_prompt_tokens": est_tokens[i],
                    "prompt_tokens": None, "cached_tokens": None, "cache_state": "unverified", "tokens": None,
                    "elapsed_s": el, "hits": 0, "undersized": src_short, "source_undersized": src_short}
        choices = res.get("choices") or []
        ch = choices[0] if choices else {}
        msg = ch.get("message") or {}
        txt, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
        complete = ch.get("finish_reason") in CLEAN_FINISH and bool(txt or reasoning)
        usage = parse_usage(res.get("usage"), complete=complete)
        accepted = complete and usage["prompt_tokens"] is not None and usage["tokens"] is not None and usage["tokens"] > 0
        ptok = usage["prompt_tokens"]
        undersized = src_short or (ptok is not None and ptok < min_prompt_tokens)
        hits = sum(1 for c in codes if c in txt or c in reasoning) if accepted else 0
        status = "OK" if accepted else f"FAILED({ch.get('finish_reason') or 'no_output'})"
        print(f"  [{status:12s} {name:24s}] prompt={ptok} (cached={usage['cached_tokens']}, {usage['cache_state']}), gen={usage['tokens']}, {el:.1f}s, markers={hits}/3", flush=True)
        return {
            "name": name, "accepted": accepted, "error": None if accepted else (ch.get("finish_reason") or "incomplete_or_empty"),
            "est_prompt_tokens": est_tokens[i], "prompt_tokens": ptok, "cached_tokens": usage["cached_tokens"],
            "cache_state": usage["cache_state"], "tokens": usage["tokens"], "elapsed_s": el,
            "hits": hits, "undersized": undersized, "source_undersized": src_short,
        }

    t_all = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_streams) as ex:
        results = [f.result() for f in [ex.submit(run_one, i) for i in range(n_streams)]]
    wall = time.perf_counter() - t_all

    completed = [r for r in results if r["accepted"]]
    cold_completed = [r for r in completed if r["cache_state"] == "cold"]
    failed_names = [r["name"] for r in results if not r["accepted"]]
    undersized_names = [r["name"] for r in results if r["undersized"]]
    submitted_est = sum(r["est_prompt_tokens"] for r in results)
    reported_prompt = sum(r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None)
    completed_prompt = sum(r["prompt_tokens"] for r in completed)
    completed_gen = sum(r["tokens"] for r in completed)
    cold_prompt = sum(r["prompt_tokens"] for r in cold_completed)
    tot_hits = sum(r["hits"] for r in completed)

    observed_kv = None
    try:
        health = json.loads(urllib.request.urlopen(f"{base_url}/health", timeout=10).read().decode("utf-8"))
        observed_kv = _observed_kv_residency(health)
    except Exception:
        pass

    if failed_names and undersized_names:
        cell_status = "incomplete_and_undersized"
    elif failed_names:
        cell_status = "incomplete"
    elif undersized_names:
        cell_status = "undersized"
    elif len(cold_completed) != n_streams:
        cell_status = "cache_not_cold"
    else:
        cell_status = "ok"
    qualified = cell_status == "ok"

    print(f"\n=== {n_streams}x DISTINCT REAL-PROSE SUMMARY (status={cell_status}) ===")
    print(f"  Submitted streams            : {n_streams} (est_prompt={submitted_est:,}, reported_prompt={reported_prompt:,})")
    print(f"  Completed streams            : {len(completed)}/{n_streams} (prompt={completed_prompt:,}, completion={completed_gen:,})")
    print(f"  Verified cold streams        : {len(cold_completed)}/{n_streams} (cold_completed_prompt={cold_prompt:,})")
    if failed_names:
        print(f"  Failed streams               : {', '.join(failed_names)}")
    if undersized_names:
        print(f"  Undersized cells (<{min_prompt_tokens:,} tok): {', '.join(undersized_names)}")
    if observed_kv is not None:
        print(f"  Observed engine KV residency : {observed_kv:,} tokens")
    print(f"  Wall clock                   : {wall:.1f}s")
    print(f"  Unique marker recall         : {tot_hits}/{3 * n_streams}")

    return {
        "mode": "distinct-long", "target_tokens": target_tokens, "min_prompt_tokens": min_prompt_tokens,
        "cell_status": cell_status, "qualified": qualified, "submitted_streams": n_streams,
        "completed_streams": len(completed), "cold_verified_streams": len(cold_completed),
        "failed_streams": failed_names, "submitted_est_prompt_tokens": submitted_est,
        "reported_prompt_tokens": reported_prompt, "completed_prompt_tokens": completed_prompt,
        "completed_completion_tokens": completed_gen, "cold_completed_prompt_tokens": cold_prompt,
        "undersized_streams": undersized_names, "observed_kv_resident_tokens": observed_kv,
        "wall_s": wall, "marker_hits": tot_hits, "streams": results,
    }


def main(argv=None) -> int:
    """CLI entry point returning 0 on qualified/valid runs and 1 when verification fails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="model")
    ap.add_argument("--prose-dir", default="/tmp/prose_8")
    ap.add_argument("--mode", choices=["prose-decode", "prose-loss-dump", "prose-loss-compare", "distinct-long"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--bf16", default="")
    ap.add_argument("--fp8", default="")
    ap.add_argument("--tokens", type=int, default=502000)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--levels", default="1,4,8")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tf-tokens", type=int, default=48)
    ap.add_argument("--tf-file", default="")
    ap.add_argument("--min-prompt-ratio", type=float, default=0.95)
    a = ap.parse_args(argv)

    if a.mode == "prose-decode":
        try:
            levels = tuple(int(x) for x in a.levels.split(","))
            _validate_decode_args(levels, a.gen_tokens, len(DISTINCT_PROSE_8))
        except ValueError as exc:
            ap.error(str(exc))
        report = run_prose_decode(a.url, a.prose_dir, model=a.model, levels=levels, gen_tokens=a.gen_tokens, seed=a.seed)
    elif a.mode == "prose-loss-dump":
        if a.tf_tokens < 1:
            ap.error("--tf-tokens must be positive")
        if not a.out:
            ap.error("--out is required for prose-loss-dump")
        report = dump_prose_loss(a.url, a.prose_dir, a.out, model=a.model, tf_tokens=a.tf_tokens, tf_file=a.tf_file)
    elif a.mode == "prose-loss-compare":
        if not a.bf16 or not a.fp8:
            ap.error("--bf16 and --fp8 are required for prose-loss-compare")
        report = compare_prose_loss(a.bf16, a.fp8)
    else:
        if a.tokens < 1:
            ap.error("--tokens must be positive")
        if a.streams < 1 or a.streams > len(STREAM_SOURCES_500K):
            ap.error(f"--streams must be between 1 and {len(STREAM_SOURCES_500K)}")
        if not math.isfinite(a.min_prompt_ratio) or not (0.0 < a.min_prompt_ratio <= 1.0):
            ap.error("--min-prompt-ratio must be finite and in (0, 1]")
        report = run_distinct_long(a.url, a.prose_dir, a.tokens, a.streams, model=a.model, min_prompt_ratio=a.min_prompt_ratio)

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if report.get("qualified", True) else 1


if __name__ == "__main__":
    sys.exit(main())
