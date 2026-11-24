"""Prose KV loss probe: chapter logprob dumps, immutable ground-truth continuation scoring, and comparison."""
import hashlib
import json
import math
import statistics
import time

CLEAN_FINISH = {"stop", "length"}

LOSS_CASES = [
    ("war_peace_2k", "war_and_peace", 25000, 10500, "Provide a concise literary analysis of the character dynamics in this scene."),
    ("monte_cristo_4k", "monte_cristo", 30000, 21000, "Trace the chain of events and motives revealed in this chapter."),
    ("les_mis_8k", "les_miserables", 40000, 44000, "Examine how Hugo develops the tension between law and mercy across these chapters."),
    ("don_quixote_16k", "don_quixote", 35000, 88000, "Synthesize the major episodes in this section and how Sancho Panza's perspective evolves."),
    ("anna_karenina_24k", "anna_karenina", 30000, 132000, "Analyze the parallel narrative threads in these chapters and their emotional turning points."),
    ("david_copperfield_32k", "david_copperfield", 25000, 174000, "Summarize the key characters introduced across these chapters and Dickens' narrative voice."),
]


def finite_num(value) -> float | None:
    """Return finite float for int/float (excluding bool), else None."""
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return None


def sha256_token_ids(ids: list[int]) -> str:
    """Return lowercase 64-hex SHA-256 of comma-joined decimal token IDs."""
    return hashlib.sha256(",".join(str(i) for i in ids).encode("ascii")).hexdigest()


def valid_token_ids(value, expected_len: int | None = None) -> list[int] | None:
    """Validate a complete list of non-negative integer token IDs."""
    if not isinstance(value, list) or not value:
        return None
    if not all(type(x) is int and x >= 0 for x in value):
        return None
    if expected_len is not None and len(value) != expected_len:
        return None
    return list(value)


def _fmt(val, fmt_s: str = ".6f") -> str:
    """Format a numeric value or report unavailable when None."""
    return "unavailable" if val is None else format(val, fmt_s)


def _tokenize_text(post_json, base_url: str, text: str, add_special_tokens: bool) -> list[int]:
    """Tokenize text via /v1/tokenize and validate the integer token ID list."""
    res = post_json(f"{base_url}/v1/tokenize", {"prompt": text, "add_special_tokens": add_special_tokens})
    ids = valid_token_ids(res.get("tokens"))
    if ids is None:
        raise ValueError("invalid /v1/tokenize response: missing non-empty integer token list")
    return ids


def score_teacher_forced_continuation(post_json, base_url: str, model: str, prefix_ids: list[int], cont_ids: list[int]) -> dict:
    """Score one immutable tokenized ground-truth continuation via /v1/completions prompt_logprobs."""
    full_ids = prefix_ids + cont_ids
    t0 = time.perf_counter()
    res = post_json(f"{base_url}/v1/completions", {
        "model": model,
        "prompt": full_ids,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 20,
    })
    dt = time.perf_counter() - t0
    if res.get("error"):
        raise ValueError(f"teacher-forced request error: {res['error']}")
    choices = res.get("choices") or []
    prompt_lps = (choices[0].get("prompt_logprobs") if choices else None) or res.get("prompt_logprobs")
    if not isinstance(prompt_lps, list) or len(prompt_lps) != len(full_ids):
        raise ValueError("prompt_logprobs missing or length mismatch")

    steps = []
    for offset, gt_id in enumerate(cont_ids):
        row = prompt_lps[len(prefix_ids) + offset]
        if not isinstance(row, dict) or str(gt_id) not in row:
            raise ValueError(f"missing ground-truth token {gt_id} in prompt_logprobs[{len(prefix_ids) + offset}]")
        gt_entry = row[str(gt_id)]
        gt_lp = finite_num(gt_entry.get("logprob") if isinstance(gt_entry, dict) else None)
        if gt_lp is None:
            raise ValueError(f"invalid ground-truth logprob for token {gt_id}")
        top = {}
        top1_id = None
        top1_lp = -math.inf
        for tid_s, entry in row.items():
            lp = finite_num(entry.get("logprob") if isinstance(entry, dict) else None)
            if lp is None:
                continue
            top[str(tid_s)] = lp
            if entry.get("rank") == 1:
                top1_id, top1_lp = str(tid_s), lp
            elif top1_id is None or lp > top1_lp:
                top1_id, top1_lp = str(tid_s), lp
        steps.append({
            "gt_token_id": gt_id,
            "gt_token": gt_entry.get("decoded_token", ""),
            "gt_logprob": gt_lp,
            "gt_prob": math.exp(gt_lp),
            "gt_rank": gt_entry.get("rank"),
            "top1_token_id": top1_id,
            "top": top,
        })

    mean_nll = statistics.mean(-s["gt_logprob"] for s in steps)
    return {
        "prefix_token_count": len(prefix_ids),
        "continuation_token_count": len(cont_ids),
        "continuation_token_ids": cont_ids,
        "continuation_sha256": sha256_token_ids(cont_ids),
        "mean_gt_nll": mean_nll,
        "gt_perplexity": math.exp(mean_nll),
        "elapsed_s": dt,
        "steps": steps,
    }


def dump_prose_loss(
    post_json,
    load_book,
    parse_usage,
    base_url: str,
    prose_dir: str,
    out_json: str,
    model: str = "model",
    cases=None,
    tf_tokens: int = 48,
    tf_file: str = "",
) -> dict:
    """Dump chapter top-logprobs and immutable tokenized ground-truth continuation scores to JSON."""
    case_list = cases if cases is not None else LOSS_CASES
    print("\n=== REAL-PROSE KV LOSS PROBE (CHAPTER LOGPROBS + IMMUTABLE GROUND-TRUTH CONTINUATION) ===")
    records: dict = {}
    for label, book, off, chars, q in case_list:
        raw_book = load_book(prose_dir, book)
        excerpt = raw_book[off : off + chars] if len(raw_book) > off else raw_book[:chars]
        t0 = time.perf_counter()
        res = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": f"{excerpt}\n\nQuestion: {q}"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "seed": 1234,
            "max_tokens": 64,
            "logprobs": True,
            "top_logprobs": 20,
        })
        dt = time.perf_counter() - t0
        if res.get("error"):
            raise ValueError(f"{label}: server error {res['error']}")
        ch = res["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if ch.get("finish_reason") not in CLEAN_FINISH or not (text or reasoning):
            raise ValueError(f"{label}: incomplete or empty response")
        usage = parse_usage(res.get("usage"), complete=True)
        lp_items = (ch.get("logprobs") or {}).get("content") or []
        steps = [
            {
                "token": it["token"],
                "token_id": it.get("token_id"),
                "logprob": finite_num(it.get("logprob")),
                "top": {e["token"]: lp for e in it.get("top_logprobs") or [] if (lp := finite_num(e.get("logprob"))) is not None},
            }
            for it in lp_items
        ]
        tf_meta = res.get("tensorfold") if isinstance(res.get("tensorfold"), dict) else {}
        tids = valid_token_ids(tf_meta.get("token_ids") or ch.get("token_ids"), usage["tokens"])
        records[label] = {
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "cache_state": usage["cache_state"],
            "completion_tokens": usage["tokens"],
            "finish_reason": ch.get("finish_reason"),
            "token_ids": tids,
            "token_sha256": tf_meta.get("token_sha256") or (sha256_token_ids(tids) if tids else None),
            "steps": steps,
            "text": text,
            "reasoning": reasoning,
        }
        print(f"  [{label:22s}] prompt={usage['prompt_tokens']} tok ({dt:.2f}s) -> {text[:85]!r}")

    if tf_file:
        with open(tf_file, "r", encoding="utf-8") as f:
            tf_spec = json.load(f)
        prefix_ids = valid_token_ids(tf_spec.get("prefix_ids"))
        cont_ids = valid_token_ids(tf_spec.get("continuation_ids"))
        if prefix_ids is None or cont_ids is None:
            raise ValueError("--tf-file must contain non-empty integer prefix_ids and continuation_ids")
    else:
        md_full = load_book(prose_dir, "moby_dick")
        prefix_text = "Continue the following Melville passage verbatim:\n\n" + (
            md_full[50000:136000] if len(md_full) > 50000 else md_full[: max(1, len(md_full) // 2)]
        )
        cont_text = md_full[136000:137500] if len(md_full) > 136000 else md_full[max(1, len(md_full) // 2) :]
        prefix_ids = _tokenize_text(post_json, base_url, prefix_text, add_special_tokens=True)
        cont_ids = _tokenize_text(post_json, base_url, cont_text, add_special_tokens=False)[:tf_tokens]

    tf_record = score_teacher_forced_continuation(post_json, base_url, model, prefix_ids, cont_ids)
    records["_tf_moby_dick_16k"] = tf_record
    print(
        f"  [moby_dick_16k_tf      ] {tf_record['continuation_token_count']} gt tokens, "
        f"NLL={tf_record['mean_gt_nll']:.5f}, PPL={tf_record['gt_perplexity']:.4f} ({tf_record['elapsed_s']:.2f}s)"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved real-prose KV probe to {out_json}")
    return records


def _cond_intersection_step(t16: dict, t8: dict) -> dict:
    """Compute conditional intersection-only KL/TVD and unobserved probability mass outside the shared top-k."""
    common = [k for k in (set(t16) & set(t8)) if finite_num(t16[k]) is not None and finite_num(t8[k]) is not None]
    z16 = sum(math.exp(t16[k]) for k in common)
    z8 = sum(math.exp(t8[k]) for k in common)
    missing_ref, missing_cand = max(0.0, 1.0 - z16), max(0.0, 1.0 - z8)
    if not common or z16 <= 0.5 or z8 <= 0.5:
        return {"kl": None, "tvd": None, "common_mass_ref": z16, "common_mass_cand": z8,
                "missing_support_ref": missing_ref, "missing_support_cand": missing_cand}
    kl = max(0.0, sum((math.exp(t16[k]) / z16) * math.log((math.exp(t16[k]) / z16) / (math.exp(t8[k]) / z8)) for k in common))
    tvd = 0.5 * sum(abs(math.exp(t16[k]) / z16 - math.exp(t8[k]) / z8) for k in common)
    return {"kl": kl, "tvd": tvd, "common_mass_ref": z16, "common_mass_cand": z8,
            "missing_support_ref": missing_ref, "missing_support_cand": missing_cand}


def _compare_sequences(r16: dict, r8: dict) -> dict:
    """Compare complete token sequences, token IDs, and lengths before declaring IDENTICAL."""
    s16, s8 = r16.get("steps") or [], r8.get("steps") or []
    ids16, ids8 = r16.get("token_ids"), r8.get("token_ids")
    if not s16 or not s8:
        return {"identical": False, "first_diff_step": None, "status": "unavailable (empty sequence)",
                "len_ref": len(s16), "len_cand": len(s8)}
    for rec, steps in ((r16, s16), (r8, s8)):
        ctok = rec.get("completion_tokens")
        if type(ctok) is int and ctok != len(steps):
            return {"identical": False, "first_diff_step": None, "status": "incomplete_steps",
                    "len_ref": len(s16), "len_cand": len(s8)}
    shared = min(len(s16), len(s8))
    for i in range(shared):
        tok_diff = s16[i].get("token") != s8[i].get("token")
        id_diff = isinstance(ids16, list) and isinstance(ids8, list) and i < len(ids16) and i < len(ids8) and ids16[i] != ids8[i]
        if tok_diff or id_diff:
            return {"identical": False, "first_diff_step": i, "status": f"step {i}",
                    "len_ref": len(s16), "len_cand": len(s8)}
    if len(s16) != len(s8):
        return {"identical": False, "first_diff_step": shared,
                "status": f"length_mismatch ({len(s16)} vs {len(s8)})", "len_ref": len(s16), "len_cand": len(s8)}
    if (ids16 is not None or ids8 is not None) and ids16 != ids8:
        return {"identical": False, "first_diff_step": None, "status": "token_ids_mismatch",
                "len_ref": len(s16), "len_cand": len(s8)}
    if r16.get("text") != r8.get("text") or r16.get("reasoning", "") != r8.get("reasoning", ""):
        return {"identical": False, "first_diff_step": None, "status": "text_mismatch",
                "len_ref": len(s16), "len_cand": len(s8)}
    return {"identical": True, "first_diff_step": None, "status": "IDENTICAL",
            "len_ref": len(s16), "len_cand": len(s8)}


def compare_prose_loss(bf16_json: str, fp8_json: str) -> dict:
    """Compare two prose-loss dumps using conditional intersection-only diagnostics and exact ground-truth NLL."""
    with open(bf16_json, "r", encoding="utf-8") as f:
        b16 = json.load(f)
    with open(fp8_json, "r", encoding="utf-8") as f:
        f8 = json.load(f)

    print("\n=== REAL-PROSE KV DIAGNOSTIC REPORT (CONDITIONAL INTERSECTION-ONLY + GROUND-TRUTH NLL) ===")
    kls, tvds, dlps, miss_ref, miss_cand = [], [], [], [], []
    matched, agree = 0, 0
    cases_out = {}

    for name, r16 in b16.items():
        if name.startswith("_"):
            continue
        r8 = f8.get(name)
        if not isinstance(r8, dict):
            cases_out[name] = {"status": "unavailable (missing candidate)", "identical": False, "step0_cond_kl": None}
            print(f"  {name:22s}: unavailable (missing in candidate)")
            continue
        seq = _compare_sequences(r16, r8)
        s16, s8 = r16.get("steps") or [], r8.get("steps") or []
        step0_kl = None
        for i in range(min(len(s16), len(s8))):
            if i > 0 and s16[i - 1].get("token") != s8[i - 1].get("token"):
                break
            matched += 1
            if s16[i].get("token") == s8[i].get("token"):
                agree += 1
            diag = _cond_intersection_step(s16[i].get("top") or {}, s8[i].get("top") or {})
            miss_ref.append(diag["missing_support_ref"])
            miss_cand.append(diag["missing_support_cand"])
            if diag["kl"] is not None:
                kls.append(diag["kl"])
                tvds.append(diag["tvd"])
                if i == 0:
                    step0_kl = diag["kl"]
            t8 = s8[i].get("top") or {}
            lp16 = finite_num(s16[i].get("logprob"))
            lp8 = finite_num(t8.get(s16[i].get("token")))
            if lp16 is not None and lp8 is not None:
                dlps.append(abs(lp16 - lp8))
        cases_out[name] = {**seq, "prompt_tokens": r16.get("prompt_tokens"), "step0_cond_kl": step0_kl}
        print(f"  {name:22s} ({r16.get('prompt_tokens')} tok): step0_cond_KL={_fmt(step0_kl)}, seq={seq['status']}")

    chapter_summary = {
        "matched_steps": matched,
        "top1_agree": agree,
        "top1_agree_pct": (100.0 * agree / matched) if matched else None,
        "conditional_intersection_kl_mean": statistics.mean(kls) if kls else None,
        "conditional_intersection_kl_median": statistics.median(kls) if kls else None,
        "conditional_intersection_tvd_mean": statistics.mean(tvds) if tvds else None,
        "conditional_intersection_tvd_median": statistics.median(tvds) if tvds else None,
        "missing_support_ref_mean": statistics.mean(miss_ref) if miss_ref else None,
        "missing_support_cand_mean": statistics.mean(miss_cand) if miss_cand else None,
        "top1_abs_dlogprob_mean": statistics.mean(dlps) if dlps else None,
        "top1_abs_dlogprob_max": max(dlps) if dlps else None,
        "metric_scope": "conditional_intersection_only_not_full_distribution_kl_or_ppl",
    }
    print(f"  MATCHED-PREFIX CHAPTERS ({matched} steps; conditional intersection-only diagnostics, not full KL/PPL):")
    print(f"    Top-1 Greedy Agreement           : {agree}/{matched} ({_fmt(chapter_summary['top1_agree_pct'], '.2f')}%)")
    print(f"    Conditional Intersection KL      : mean={_fmt(chapter_summary['conditional_intersection_kl_mean'])}, median={_fmt(chapter_summary['conditional_intersection_kl_median'])}")
    print(f"    Conditional Intersection TVD     : mean={_fmt(chapter_summary['conditional_intersection_tvd_mean'])}, median={_fmt(chapter_summary['conditional_intersection_tvd_median'])}")
    print(f"    Missing Support Outside Intersect: ref={_fmt(chapter_summary['missing_support_ref_mean'])}, cand={_fmt(chapter_summary['missing_support_cand_mean'])}")
    print(f"    Top-1 |Δlogprob|                 : mean={_fmt(chapter_summary['top1_abs_dlogprob_mean'], '.5f')}, max={_fmt(chapter_summary['top1_abs_dlogprob_max'], '.5f')}")

    tf_summary: dict = {"available": False}
    tf16, tf8 = b16.get("_tf_moby_dick_16k"), f8.get("_tf_moby_dick_16k")
    if (
        isinstance(tf16, dict)
        and isinstance(tf8, dict)
        and tf16.get("continuation_token_ids")
        and tf16.get("continuation_token_ids") == tf8.get("continuation_token_ids")
        and len(tf16.get("steps") or []) == len(tf16["continuation_token_ids"]) == len(tf8.get("steps") or [])
    ):
        s16, s8 = tf16["steps"], tf8["steps"]
        gt_lps16 = [finite_num(x.get("gt_logprob")) for x in s16]
        gt_lps8 = [finite_num(x.get("gt_logprob")) for x in s8]
        if all(v is not None for v in gt_lps16) and all(v is not None for v in gt_lps8):
            tkls, ttvds, tmiss_r, tmiss_c = [], [], [], []
            tagree = 0
            for x16, x8 in zip(s16, s8):
                if x16.get("top1_token_id") is not None and x16.get("top1_token_id") == x8.get("top1_token_id"):
                    tagree += 1
                diag = _cond_intersection_step(x16.get("top") or {}, x8.get("top") or {})
                tmiss_r.append(diag["missing_support_ref"])
                tmiss_c.append(diag["missing_support_cand"])
                if diag["kl"] is not None:
                    tkls.append(diag["kl"])
                    ttvds.append(diag["tvd"])
            n_tok = len(s16)
            nll16 = statistics.mean(-v for v in gt_lps16)
            nll8 = statistics.mean(-v for v in gt_lps8)
            ppl16, ppl8 = math.exp(nll16), math.exp(nll8)
            gt_diffs = [abs(a - b) for a, b in zip(gt_lps16, gt_lps8)]
            tf_summary = {
                "available": True,
                "continuation_token_count": n_tok,
                "continuation_sha256": tf16.get("continuation_sha256"),
                "gt_nll_ref": nll16,
                "gt_nll_cand": nll8,
                "gt_nll_delta": nll8 - nll16,
                "gt_ppl_ref": ppl16,
                "gt_ppl_cand": ppl8,
                "gt_ppl_ratio": ppl8 / ppl16,
                "gt_abs_dlogprob_mean": statistics.mean(gt_diffs),
                "gt_abs_dlogprob_max": max(gt_diffs),
                "top1_agree": tagree,
                "top1_agree_pct": 100.0 * tagree / n_tok,
                "conditional_intersection_kl_mean": statistics.mean(tkls) if tkls else None,
                "conditional_intersection_tvd_mean": statistics.mean(ttvds) if ttvds else None,
                "missing_support_ref_mean": statistics.mean(tmiss_r) if tmiss_r else None,
                "missing_support_cand_mean": statistics.mean(tmiss_c) if tmiss_c else None,
            }
            print(f"\n  IMMUTABLE GROUND-TRUTH CONTINUATION ({n_tok} tokens, sha256={tf_summary['continuation_sha256']}):")
            print(f"    Ground-Truth NLL / PPL           : ref={nll16:.5f} ({ppl16:.4f}), cand={nll8:.5f} ({ppl8:.4f}), ratio={ppl8/ppl16:.5f}x")
            print(f"    Ground-Truth |Δlogprob|          : mean={tf_summary['gt_abs_dlogprob_mean']:.5f}, max={tf_summary['gt_abs_dlogprob_max']:.5f}")
            print(f"    Top-1 Greedy Token Agreement     : {tagree}/{n_tok} ({tf_summary['top1_agree_pct']:.2f}%)")
            print(f"    Conditional Intersection KL/TVD  : kl={_fmt(tf_summary['conditional_intersection_kl_mean'])}, tvd={_fmt(tf_summary['conditional_intersection_tvd_mean'])}")
    else:
        print("\n  IMMUTABLE GROUND-TRUTH CONTINUATION: unavailable (missing or mismatched continuation)")

    return {"mode": "prose-loss-compare", "cases": cases_out, "chapters": chapter_summary, "teacher_forced": tf_summary}
