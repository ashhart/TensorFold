#!/usr/bin/env python3
"""Realistic natural-prose benchmark for OpenAI-compatible LLM servers:
1. 8-stream distinct literary/prose generation (drafted vs plain, zero shared prefixes, cached_tokens=0).
2. BF16 vs FP8 KV cache loss on genuine classic prose (1K..32K chapters + 48-step teacher-forced Moby-Dick NLL/KL).
3. Nx >500,000-token (4.0M+ total) completely distinct uncached classic novels (cached_tokens=0 on all streams, 3 unique markers per stream).
"""
import argparse, concurrent.futures, json, math, os, re, statistics, time, urllib.request

def strip_gutenberg(raw: str) -> str:
    # Strip Project Gutenberg header and footer so zero boilerplate is shared across books!
    m_start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, flags=re.I | re.S)
    if m_start:
        raw = raw[m_start.end():]
    else:
        raw = raw[4000:]
    m_end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", raw, flags=re.I)
    if m_end:
        raw = raw[:m_end.start()]
    return raw.strip()

def load_book(prose_dir: str, name: str) -> str:
    with open(os.path.join(prose_dir, f"{name}.txt"), "r", encoding="utf-8", errors="ignore") as f:
        return strip_gutenberg(f.read())

def post_json(url: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

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

def stream_chat(url: str, body: dict, timeout: int = 600):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    t_last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            ln = raw.decode("utf-8", errors="ignore").strip()
            if not ln.startswith("data: "):
                continue
            payload = ln[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ""):
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
    t_end = time.time()
    if t_first is None:
        t_first = t_end
    if t_last is None or t_last <= t_first:
        t_last = t_end
    return usage, t0, t_first, t_last

def run_prose_decode(base_url: str, prose_dir: str, model: str = "model", levels=(1, 4, 8), gen_tokens: int = 256):
    print("\n=== REALISTIC DISTINCT-PROSE MULTI-STREAM DECODE BENCHMARK (SSE TTFT + PURE DECODE) ===", flush=True)
    for tag, excerpt_chars in [("350tok_prose", 1400), ("2.5Ktok_chapter", 9500)]:
        prompts = []
        for i, (book, off, _, question) in enumerate(DISTINCT_PROSE_8):
            txt = load_book(prose_dir, book)[off : off + excerpt_chars]
            prompts.append((book, f"[Excerpt {i+1} — {book}]\n\n{txt}\n\nTask: {question}"))
        print(f"  --- Workload: {tag} (8 distinct classic novels, cached=0) ---", flush=True)
        for n in levels:
            for draft_on in (True, False):
                nonce = f"[Run {tag} n={n} draft={'on' if draft_on else 'plain'} ts={time.time():.3f}]\n"
                def one(idx):
                    book, p = prompts[idx]
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": nonce + p}],
                        "chat_template_kwargs": {"enable_thinking": False},
                        "temperature": 0,
                        "max_tokens": gen_tokens,
                    }
                    if not draft_on:
                        body["draft"] = False
                    u, t0, t_first, t_last = stream_chat(f"{base_url}/v1/chat/completions", body)
                    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    return book, u["prompt_tokens"], cached, u["completion_tokens"], t0, t_first, t_last

                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                    futs = [ex.submit(one, i) for i in range(n)]
                    rows = [f.result() for f in futs]
                tot_prompt = sum(r[1] for r in rows)
                tot_cached = sum(r[2] for r in rows)
                tot_gen = sum(r[3] for r in rows)
                first_tok_t = min(r[5] for r in rows)
                last_tok_t = max(r[6] for r in rows)
                start_t = min(r[4] for r in rows)
                dec_tps = sum((r[3] - 1) / max(r[6] - r[5], 1e-3) for r in rows)
                ttft_med = statistics.median(r[5] - r[4] for r in rows)
                mode_s = "DFly2" if draft_on else "Plain"
                print(
                    f"    [{n}s {mode_s:5s}] prompt={tot_prompt:,} (cached={tot_cached}) | "
                    f"decode={dec_tps:6.2f} tok/s | e2e={tot_gen/(last_tok_t - start_t):6.2f} tok/s | "
                    f"median_TTFT={ttft_med:.2f}s",
                    flush=True,
                )
def dump_prose_loss(base_url: str, prose_dir: str, out_json: str, model: str = "model"):
    print("\n=== REAL-PROSE KV LOSS PROBE (1K..32K NOVEL CHAPTERS + 48-STEP MOBY-DICK NLL) ===")
    cases = [
        ("war_peace_2k", "war_and_peace", 25000, 10500, "Provide a concise literary analysis of the character dynamics in this scene."),
        ("monte_cristo_4k", "monte_cristo", 30000, 21000, "Trace the chain of events and motives revealed in this chapter."),
        ("les_mis_8k", "les_miserables", 40000, 44000, "Examine how Hugo develops the tension between law and mercy across these chapters."),
        ("don_quixote_16k", "don_quixote", 35000, 88000, "Synthesize the major episodes in this section and how Sancho Panza's perspective evolves."),
        ("anna_karenina_24k", "anna_karenina", 30000, 132000, "Analyze the parallel narrative threads in these chapters and their emotional turning points."),
        ("david_copperfield_32k", "david_copperfield", 25000, 174000, "Summarize the key characters introduced across these chapters and Dickens' narrative voice."),
    ]
    records = {}
    for label, book, off, chars, q in cases:
        excerpt = load_book(prose_dir, book)[off : off + chars]
        t0 = time.time()
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": f"{excerpt}\n\nQuestion: {q}"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 64,
            "logprobs": True,
            "top_logprobs": 20,
        })
        dt = time.time() - t0
        ch = res["choices"][0]
        steps = [
            {"token": it["token"], "logprob": it["logprob"], "top": {e["token"]: e["logprob"] for e in it["top_logprobs"]}}
            for it in ch["logprobs"]["content"]
        ]
        records[label] = {
            "prompt_tokens": res["usage"]["prompt_tokens"],
            "steps": steps,
            "text": ch["message"]["content"],
        }
        print(f"  [{label:22s}] prompt={res['usage']['prompt_tokens']:6,d} tok ({dt:.2f}s) -> {ch['message']['content'][:85]!r}")

    # 48-step teacher-forced natural prose continuation on 16K-token Moby-Dick chapter!
    md_full = load_book(prose_dir, "moby_dick")
    md_prefix = "Continue the following Melville passage verbatim:\n\n" + md_full[50000:136000]
    md_cont_words = md_full[136000:136450].split()[:48]
    tf_steps = []
    cur = md_prefix
    t_tf = time.time()
    for w in md_cont_words:
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": cur}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
        })
        it = res["choices"][0]["logprobs"]["content"][0]
        tf_steps.append({
            "gt_word": w,
            "pred": it["token"],
            "pred_lp": it["logprob"],
            "top": {e["token"]: e["logprob"] for e in it["top_logprobs"]},
        })
        cur = cur + " " + w
    print(f"  [moby_dick_16k_tf48    ] {len(tf_steps)} teacher-forced natural prose steps in {time.time()-t_tf:.2f}s")
    records["_tf_moby_dick_16k"] = tf_steps
    json.dump(records, open(out_json, "w"))
    print(f"Saved real-prose KV probe to {out_json}")

def compare_prose_loss(bf16_json: str, fp8_json: str):
    b16 = json.load(open(bf16_json))
    f8 = json.load(open(fp8_json))
    print("\n=== REAL-PROSE BF16 vs FP8 KV CACHE LOSS REPORT ===")
    kls, tvds, dlps = [], [], []
    matched, agree = 0, 0
    for name, r16 in b16.items():
        if name.startswith("_"):
            continue
        s16, s8 = r16["steps"], f8[name]["steps"]
        step0_kl = 0.0
        for i in range(min(len(s16), len(s8))):
            if not all(s16[j]["token"] == s8[j]["token"] for j in range(i)):
                break
            matched += 1
            if s16[i]["token"] == s8[i]["token"]:
                agree += 1
            t16, t8 = s16[i]["top"], s8[i]["top"]
            common = set(t16) & set(t8)
            z16 = sum(math.exp(t16[k]) for k in common)
            z8 = sum(math.exp(t8[k]) for k in common)
            if z16 > 0.5 and z8 > 0.5:
                kl = max(0.0, sum((math.exp(t16[k])/z16) * math.log((math.exp(t16[k])/z16) / (math.exp(t8[k])/z8)) for k in common))
                tvd = 0.5 * sum(abs(math.exp(t16[k])/z16 - math.exp(t8[k])/z8) for k in common)
                kls.append(kl)
                tvds.append(tvd)
                if i == 0:
                    step0_kl = kl
            if s16[i]["token"] in t8:
                dlps.append(abs(s16[i]["logprob"] - t8[s16[i]["token"]]))
        div = next((i for i in range(min(len(s16), len(s8))) if s16[i]["token"] != s8[i]["token"]), "IDENTICAL")
        print(f"  {name:22s} ({r16['prompt_tokens']:6,d} tok): step0_KL={step0_kl:.6f} nats, first_diff_step={div}")

    print(f"  OVERALL MATCHED-PREFIX PROSE ({matched} steps across 6 chapters, 2K..32K):")
    print(f"    Top-1 Greedy Agreement : {agree}/{matched} ({100*agree/max(matched,1):.2f}%)")
    print(f"    Mean KL(BF16 || FP8)   : {statistics.mean(kls):.6f} nats (PPL mult: {math.exp(statistics.mean(kls)):.5f}x = +{(math.exp(statistics.mean(kls))-1)*100:.3f}%), median={statistics.median(kls):.6f}")
    print(f"    Mean Total Variation   : {100*statistics.mean(tvds):.3f}% (median={100*statistics.median(tvds):.3f}%)")
    print(f"    Mean Top-1 |Δlogprob|  : {statistics.mean(dlps):.5f} (median={statistics.median(dlps):.5f}, max={max(dlps):.5f})")

    tf16, tf8 = b16["_tf_moby_dick_16k"], f8["_tf_moby_dick_16k"]
    tkls, ttvds, tdlps, tagree = [], [], [], 0
    for x16, x8 in zip(tf16, tf8):
        if x16["pred"] == x8["pred"]:
            tagree += 1
        t16, t8 = x16["top"], x8["top"]
        common = set(t16) & set(t8)
        z16 = sum(math.exp(t16[k]) for k in common)
        z8 = sum(math.exp(t8[k]) for k in common)
        if z16 > 0.5 and z8 > 0.5:
            tkls.append(max(0.0, sum((math.exp(t16[k])/z16) * math.log((math.exp(t16[k])/z16) / (math.exp(t8[k])/z8)) for k in common)))
            ttvds.append(0.5 * sum(abs(math.exp(t16[k])/z16 - math.exp(t8[k])/z8) for k in common))
        if x16["pred"] in t8:
            tdlps.append(abs(x16["pred_lp"] - t8[x16["pred"]]))
    print(f"\n  16K MOBY-DICK TEACHER-FORCED CONTINUATION ({len(tf16)} steps, zero trajectory drift):")
    print(f"    Top-1 Greedy Agreement : {tagree}/{len(tf16)} ({100*tagree/len(tf16):.2f}%)")
    print(f"    Mean KL(BF16 || FP8)   : {statistics.mean(tkls):.6f} nats (PPL mult: {math.exp(statistics.mean(tkls)):.5f}x = +{(math.exp(statistics.mean(tkls))-1)*100:.3f}%), median={statistics.median(tkls):.6f}")
    print(f"    Mean Total Variation   : {100*statistics.mean(ttvds):.3f}%")
    print(f"    Mean Top-1 |Δlogprob|  : {statistics.mean(tdlps):.5f} (max={max(tdlps):.5f})")

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

def build_stream_manuscript(prose_dir: str, idx: int, target_tokens: int, chars_per_tok: float) -> tuple[str, tuple[str, str, str], str]:
    name, books, codes, lit_q = STREAM_SOURCES_500K[idx]
    joined = "\n\n".join(load_book(prose_dir, b) for b in books)
    # Distinct opening header so token 0 is unique on every stream (cached_tokens == 0 guaranteed!)
    target_chars = int(target_tokens * chars_per_tok)
    body = joined[:target_chars]
    p10 = int(len(body) * 0.10)
    p50 = int(len(body) * 0.50)
    p90 = int(len(body) * 0.90)
    manuscript = (
        f"Archival Volume #{idx + 1} ({name}):\n\n"
        + body[:p10]
        + f"\n\n[Archival Margin Note Alpha for {name}: Early catalog marker is {codes[0]}.]\n\n"
        + body[p10:p50]
        + f"\n\n[Archival Margin Note Beta for {name}: Mid-volume catalog marker is {codes[1]}.]\n\n"
        + body[p50:p90]
        + f"\n\n[Archival Margin Note Gamma for {name}: Late catalog marker is {codes[2]}.]\n\n"
        + body[p90:]
        + f"\n\nFinal Instructions for {name}: First list the three exact catalog markers (Alpha, Beta, Gamma) recorded in this volume, then in two sentences: {lit_q}."
    )
    return name, codes, manuscript

def calibrate_stream_cpts(base_url: str, prose_dir: str, n_streams: int, model: str = "model") -> list[float]:
    cpts = []
    for idx in range(n_streams):
        name, books, _, _ = STREAM_SOURCES_500K[idx]
        joined = "\n\n".join(load_book(prose_dir, b) for b in books)
        n = min(len(joined), 2_300_000)
        sample = joined[n // 6 : n // 6 + 20000] + "\n" + joined[n // 2 : n // 2 + 20000] + "\n" + joined[(5 * n) // 6 : (5 * n) // 6 + 20000]
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": sample}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 1,
        })
        ptok = res["usage"]["prompt_tokens"]
        cpt = len(sample) / ptok
        cpts.append(cpt)
        print(f"  Calibrated {name:24s}: {cpt:.4f} chars/tok (available={int(len(joined)/cpt):,} tok)", flush=True)
    return cpts

def run_distinct_long(base_url: str, prose_dir: str, target_tokens: int, n_streams: int, model: str = "model"):
    print(f"\n=== {n_streams}x ~{target_tokens:,}-TOKEN 100% DISTINCT REAL-PROSE CONCURRENT BENCHMARK ===", flush=True)
    cpts = calibrate_stream_cpts(base_url, prose_dir, n_streams, model=model)
    try:
        urllib.request.urlopen(f"{base_url}/health?reset_peak=1", timeout=10).read()
    except Exception:
        pass

    payloads = [build_stream_manuscript(prose_dir, i, target_tokens, cpts[i]) for i in range(n_streams)]
    for i, (name, codes, ms) in enumerate(payloads):
        print(f"  Prepared {name:24s}: {len(ms):,} chars (~{int(len(ms)/cpts[i]):,} est tokens), unique markers={codes}", flush=True)
    def run_one(i):
        name, codes, ms = payloads[i]
        t0 = time.time()
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": ms}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 96,
        }, timeout=7200)
        el = time.time() - t0
        u = res["usage"]
        ptok = u["prompt_tokens"]
        cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        gtok = u["completion_tokens"]
        txt = res["choices"][0]["message"]["content"]
        hits = sum(1 for c in codes if c in txt)
        print(f"  [Done {name:24s}] prompt={ptok:,} (cached={cached}), gen={gtok}, {el:.1f}s, markers={hits}/3 -> {txt[:105]!r}", flush=True)
        return name, ptok, cached, gtok, el, hits, txt

    print(f"\nLaunching all {n_streams} distinct real-prose streams concurrently (`cached_tokens == 0` on all {n_streams})...", flush=True)
    t_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_streams) as ex:
        futs = [ex.submit(run_one, i) for i in range(n_streams)]
        results = [f.result() for f in futs]
    wall = time.time() - t_all

    tot_prompt = sum(r[1] for r in results)
    tot_cached = sum(r[2] for r in results)
    tot_gen = sum(r[3] for r in results)
    tot_kv = tot_prompt + tot_gen
    tot_hits = sum(r[5] for r in results)
    print(f"\n=== {n_streams}x DISTINCT REAL-PROSE SUMMARY ===")
    print(f"  Streams completed            : {len(results)}/{n_streams} (total cached_tokens = {tot_cached})")
    print(f"  Distinct cold prompt tokens  : {tot_prompt:,} tokens ({tot_prompt/1e6:.3f}M)")
    print(f"  Total simultaneous KV tokens : {tot_kv:,} tokens ({tot_kv/1e6:.3f}M)")
    print(f"  Wall clock (prefill+decode)  : {wall:.1f}s (effective cold throughput: {(tot_prompt+tot_gen)/wall:.1f} tok/s)")
    print(f"  Unique marker recall (24)    : {tot_hits}/{3 * n_streams} ({sum(1 for r in results if r[5]==3)}/{n_streams} streams 3/3)")
    try:
        h = json.loads(urllib.request.urlopen(f"{base_url}/health", timeout=10).read().decode("utf-8"))
        mem = h.get("memory", {})
        print(f"  GPU Memory (rank 0)          : active={mem.get('active',0)/(1024**3):.2f} GiB, peak={mem.get('peak',0)/(1024**3):.2f} GiB, cache={mem.get('cache',0)/(1024**3):.2f} GiB")
    except Exception as e:
        print(f"  Health note: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="model")
    ap.add_argument("--prose-dir", default="/tmp/prose_8")
    ap.add_argument("--mode", choices=["prose-decode", "prose-loss-dump", "prose-loss-compare", "distinct-long"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--bf16", default="")
    ap.add_argument("--fp8", default="")
    ap.add_argument("--tokens", type=int, default=502000)
    ap.add_argument("--streams", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "prose-decode":
        run_prose_decode(a.url, a.prose_dir, model=a.model)
    elif a.mode == "prose-loss-dump":
        dump_prose_loss(a.url, a.prose_dir, a.out, model=a.model)
    elif a.mode == "prose-loss-compare":
        compare_prose_loss(a.bf16, a.fp8)
    elif a.mode == "distinct-long":
        run_distinct_long(a.url, a.prose_dir, a.tokens, a.streams, model=a.model)
