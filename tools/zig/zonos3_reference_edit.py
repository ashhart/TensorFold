#!/usr/bin/env python3
"""Reference dumps for the ZONOS3 Mini port's speech edits, speaker encoder and aligner, from the publisher's code."""

USAGE = """Run in the environment zonos3_reference.py runs in (the publisher's `zonos3` package and its fusealign_hub).

    edit-cases RELEASE SHORT LONG OUT    the edit checks' cases from two `rows` dumps (greedy_embedding, laci_long)
    edit     LM_DIR ADAPTER.pt CASE OUT [--steer-bundle B --laci-heads H --text-rows R]  one edit, fill by fill
    speaker  CKPT.pth AUDIO.npy OUT      the speaker encoder's stages and embedding on 44.1 kHz mono audio
    aligner  WEB_ALIGNER AUDIO.npy TOKENS.json OUT   fusealign_hub on the public aligner weights
    clips    SHORT LONG OUT              the served checks' two clips: the same dumps' audio as WAVs, .npy and tokens
    served-cases RELEASE OUT             the served checks' edits, once OUT holds short_clip, speaker_short, align_short
                                         (zonos3_reference.py clip, and speaker and aligner here) and the same for long

CASE: {"codes": the clip's codes .npy, "embedding": its speaker embedding .npy or null, "request": the EditRequest
without audio_base64}. edit writes response.json, codes.npy, spliced.npy, audio.s16 and, per fill, fill_N/ with what
`rows` writes plus fill.json (the layout its prompt was built from). speaker writes fbank.npy [frames, 80], head.npy
and stage_1..4.npy ([2560, frames], bin-major), mixed.npy, pooled.npy and embedding.npy. aligner writes words.json
(word, start, end, presence, placed), text_feats.npy, joint.npy, start_ms.npy, end_ms.npy and presence.npy; it runs
the package in fp32 on the CPU (its Dia encoder in bf16, as the package runs it there).
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

TEXT = "The quick brown fox jumps over the lazy dog, and then it rests by the river."
VOICE = "gguf/voices/English/Savannah_en-American_F_1"


def save(out, taps):
    for name, value in taps.items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(value, dtype=np.float32))


def edit_cases(args):
    """The edit checks' cases: clips from two `rows` dumps, the voice's embedding, word timings by letter count."""
    from zonos3.tts.delay import shear_up

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    embedding = str(Path(args.release) / f"{VOICE}.embedding.npy")

    def clip(name, dump):
        rows = torch.as_tensor(np.load(Path(dump) / "rows.npy"), dtype=torch.int64)[:, :10]
        frames = shear_up(rows, 4097)
        codes = frames[: int(((frames >= 0) & (frames < 4096)).all(dim=1).sum())].numpy().astype(np.int32)
        np.save(out / f"{name}.npy", codes)
        return str(out / f"{name}.npy"), len(codes) / 15.0

    def words(text, seconds):
        parts = text.split()
        total = sum(len(w) for w in parts)
        timed, at = [], 0
        for w in parts:
            timed.append({"word": w, "start": seconds * at / total, "end": seconds * (at + len(w)) / total})
            at += len(w)
        return timed

    short, short_s = clip("clip_short", args.short)
    long_text = " ".join([TEXT] * 4)
    long, long_s = clip("clip_long", args.long)
    greedy = {"temperature": 0.0, "seed": 7}
    cat = TEXT.replace("fox", "cat")
    late = long_text.replace("fox", "cat", 1)[::-1].replace("revir", "egdirb", 1)[::-1]
    cases = {
        "edit_span": {"target_text": cat, "spans": [[1.0, 1.6]], **greedy},
        "edit_sampled": {"target_text": cat, "spans": [[1.0, 1.6]], "seed": 11},
        "edit_plain_row": {"target_text": cat, "spans": [[1.0, 1.6]], "speaker_embedding": False, "min_fill_ratio": 0.0, **greedy},
        "edit_three": {
            "target_text": TEXT.replace("quick", "slow").replace("lazy", "sleepy").replace("river", "bridge"),
            "spans": [[0.3, 0.7], [2.2, 2.7], [3.7, 4.2]],
            "controls": {"mood.valence": 1.0},
            "region_controls": [None, {"delivery.whisper": 1.5}, {"identity.age": 1.0, "mood.valence": -1.0}],
            "steer_onset_seconds": 0.2,
            "condition_values": {"speaking_rate": 18.0},
            **greedy,
        },
        "edit_words": {"target_text": TEXT.replace("brown fox", "small grey cat").replace("rests", "sleeps"), "original_text": TEXT, "original_words": words(TEXT, short_s), **greedy},
        "edit_windows": {"target_text": late, "original_text": long_text, "original_words": words(long_text, long_s), **greedy},
        "edit_window_spans": {"target_text": late, "original_text": long_text, "original_words": words(long_text, long_s), "spans": [[1.0, 1.6], [long_s - 1.2, long_s - 0.4]], "context_seconds": 2.0, **greedy},
    }
    for name, request in cases.items():
        long_clip = name.startswith("edit_window")
        case = {"codes": long if long_clip else short, "embedding": embedding, "request": request}
        (out / f"{name}.json").write_text(json.dumps(case))
    print(f"{len(cases)} edit cases -> {out} (clips of {short_s:.2f} s and {long_s:.2f} s)")


def edit(args):
    """One /v2/edit of zonos3.tts.edit_runtime on zonos3.reference with the edit adapter, fill by fill."""
    import asyncio

    from zonos3.models.accent_format import load_adapter
    from zonos3.models.text_extra import TextExtraRows
    from zonos3.reference.accents import AccentState
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import EditLayout, ReferenceAssets, ReferenceContext, Sampling, generate_rows, output_gain, steer_fn
    from zonos3.server.schemas import EditRequest
    from zonos3.tokenizer import matched_codec
    from zonos3.tokenizer.matched_codec import decode_model
    from zonos3.tts import plan as P
    from zonos3.tts.edit import edit_marker_rows
    from zonos3.tts.edit_runtime import EditModel, EditRuntime, fill_steering, steer_fields, steer_release
    from zonos3.tts.laci import discover_laci_heads
    from zonos3.tts.directions import RAW_SPACE
    from zonos3.tts.steering import check_controls

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    case = json.loads(Path(args.case).read_text())
    device = torch.device(args.device)
    adapter = load_adapter(args.adapter)
    lm = Zonos3LM.load(args.lm, device="cpu", dtype=torch.float32).grow(adapter.grown)
    if args.text_rows:
        lm.append_text_rows(TextExtraRows.load(args.text_rows))
    lm = lm.to(device)
    config = lm.config
    heads = discover_laci_heads(args.laci_heads) if args.laci_heads else None
    assets = ReferenceAssets.load(steer_bundle=args.steer_bundle, laci_heads=heads)
    ctx = ReferenceContext(lm, assets, device)
    lora = AccentState.for_model(lm, adapter, 1.0)
    clip = torch.as_tensor(np.load(case["codes"]), dtype=torch.int64)
    embedding = None
    if case.get("embedding"):
        embedding = torch.as_tensor(np.load(case["embedding"]).astype(np.float32).reshape(-1))
    hop = int(matched_codec.entry(config.audio_codec)["hop"])
    fills = []

    class Engine:
        model = EditModel.from_config(config, hop=hop, max_spans=edit_marker_rows(config.text_extra_tokens, ctx.text_tokenizer_spec))

        async def decode_audio(self, audio):
            return np.zeros(len(clip) * hop, dtype=np.float32), 44100

        async def embed(self, audio):
            return embedding

        async def encode(self, audio, max_seconds):
            return clip.numpy()

        async def check(self, target_text, doses, condition_buckets, condition_values, steer):
            steer_release(assets.steer_bundle, steer, hint=ctx.steer_hint)
            P.validate_text_markers(ctx.text_tokenizer_spec, target_text, edit=True)
            raw = set()
            if doses is not None:
                for region in doses:
                    check_controls(assets.steer_bundle, region)
                raw = {did for region in doses for did in region if assets.steer_bundle.directions[did].space == RAW_SPACE}
            P.resolve_condition_buckets(ctx, condition_buckets, condition_values)
            return raw

        async def fill(self, job):
            # servers/space/app.py's SpaceEditEngine.fill and zonos3.reference.synth.edit_region, keeping the plan
            steering = fill_steering(assets.steer_bundle, job, hint=ctx.steer_hint, sources=ctx.steering_sources, hidden_size=ctx.hidden_size, frame_rate=ctx.frame_rate)
            request = {
                "text": job.text,
                "language": job.language,
                "text_normalization": job.text_normalization,
                **job.sampling,
                "seed": job.seed,
                "max_tokens": job.max_tokens,
                "speaker_mode": "embedding",
                "condition_buckets": job.condition_buckets,
                "condition_values": job.condition_values,
                "stream": False,
                **steer_fields(job),
            }
            layout = EditLayout(torch.as_tensor(job.window_codes, dtype=torch.int64), job.spans, [torch.as_tensor(f, dtype=torch.int32) for f in job.fills])
            moved = None if steering.speaker_embedding is None else steering.speaker_embedding.to(device)
            with torch.inference_mode():
                plan = P.plan_prompt(ctx, request, lambda req, mode: P.Speaker(mode, moved, None), hidden_delta=steering.hidden_delta, release_delta=steering.release_delta, edit=layout)
                sampling = Sampling.from_request(plan.request)
                sampling.min_frames = int(job.min_frames)
                attempt = generate_rows(lm, plan.cond, sampling=sampling, seed=plan.seed, max_tokens=plan.max_tokens, stop_id=config.span_end_id, twin=plan.twin, cfg_scale=plan.cfg_scale, accents=lora, steer=steer_fn(plan, None))
            if plan.twin is not None:
                raise SystemExit("an edit fill with a guidance twin: not dumped")
            d = out / f"fill_{len(fills)}"
            d.mkdir(exist_ok=True)
            generated = torch.stack([r.to(torch.int32) for r in attempt.rows]).numpy()
            np.save(d / "input_ids.npy", plan.cond.input_ids.to(torch.float32).numpy())
            (d / "input_ids.cond.json").write_text(json.dumps(plan.cond_json()) + "\n")
            np.save(d / "rows.npy", generated)
            steer = steer_fn(plan, None)
            deltas = np.zeros((len(generated) + 1, len(plan.sources), 1024), dtype=np.float32)
            active = [False] * (len(generated) + 1)
            for f in range(1, len(generated) + 1):
                delta = None if steer is None else steer(f)
                if delta is not None:
                    active[f] = True
                    for j, source in enumerate(plan.sources):
                        deltas[f, j] = delta[source].numpy()
            if any(active):  # as `rows` writes a steered request's deltas
                np.save(d / "steer.npy", deltas)
                (d / "steer.json").write_text(json.dumps({"sources": list(plan.sources), "active": active, "twin_steered": False}) + "\n")
            np.save(d / "window_codes.npy", np.asarray(job.window_codes, dtype=np.int32))
            for j, f in enumerate(job.fills):
                np.save(d / f"prior_{j}.npy", np.asarray(f, dtype=np.int32))
            meta = {"text": job.text, "planned_text": plan.text, "language": job.language, "seed": plan.seed, "max_tokens": plan.max_tokens, "min_frames": int(job.min_frames)}
            meta.update({"spans": job.spans, "priors": len(job.fills), "controls": job.controls, "controls_text": job.controls_text, "sources": list(plan.sources)})
            meta.update({"eos_frame": attempt.eos_frame, "frames": len(generated), "sampling": job.sampling, **steer_fields(job)})
            (d / "fill.json").write_text(json.dumps(meta) + "\n")
            fills.append(meta)
            n_cb = config.n_codebooks
            rows = generated[:-1, :n_cb] if len(generated) > 1 else np.zeros((0, n_cb), dtype=np.int32)
            return rows, -1 if attempt.eos_frame is None else int(attempt.eos_frame)

        async def decode(self, codes, decoder):
            np.save(out / "spliced.npy", np.asarray(codes, dtype=np.int32))
            model = assets.decoder(config.audio_codec, device)
            audio = decode_model(model, torch.as_tensor(codes, dtype=torch.int64).unsqueeze(0), matched_codec.entry(config.audio_codec))[0].numpy()
            return (audio * np.float32(output_gain(config.audio_codec, None))).astype(np.float32)

    request = EditRequest(audio_base64=base64.b64encode(b"clip").decode(), **case["request"])
    response = asyncio.run(EditRuntime(Engine(), None, None).edit(request))
    take = response["takes"][0]
    import io, wave

    with wave.open(io.BytesIO(base64.b64decode(take.pop("audio_base64")))) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    pcm.tofile(out / "audio.s16")
    np.save(out / "codes.npy", clip.numpy().astype(np.int32))
    if embedding is not None:
        np.save(out / "embedding.npy", embedding.numpy())
    (out / "request.json").write_text(json.dumps(case["request"]) + "\n")
    (out / "response.json").write_text(json.dumps({**response, "fills": len(fills), "max_spans": Engine.model.max_spans}) + "\n")
    print(f"{len(fills)} fills, spans {response['spans']}, fill seconds {take['fill_seconds']}, stopped {take['stopped']} -> {out}")


def speaker(args):
    """The release's speaker encoder (MECT-B2, fp32, CPU) on 44.1 kHz audio: its stages and embedding.npy [192]."""
    from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding, registry_entry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entry = registry_entry("ownmect_s2")
    entry["checkpoint"] = args.checkpoint
    model = MECTFbankSpeakerEmbedding(entry, device="cpu")
    audio = torch.as_tensor(np.load(args.audio), dtype=torch.float32).reshape(1, -1)
    taps = {}
    net = model.model
    with torch.inference_mode():
        wav = model.prepare_input(audio, 44100)
        feats = model.fbank(wav[0].float())
        x = feats[None].transpose(1, 2)
        x = x - torch.mean(x, -1, keepdim=True)
        fe = net.feature_extractor
        x = fe.head(x.unsqueeze(1))
        taps["head"] = x[0].permute(1, 0, 2).reshape(-1, x.shape[-1])
        stages = []
        for i, block in enumerate((fe.mect_block1, fe.mect_block2, fe.mect_block3, fe.mect_block4), 1):
            x = block(x)
            stages.append(x.permute(0, 2, 1, 3).reshape(1, -1, x.shape[-1]))
            taps[f"stage_{i}"] = stages[-1][0]
        mixed = fe.dense3(torch.cat([f.unsqueeze(1) for f in stages], dim=1))
        taps["mixed"] = mixed[0]
        pooled = net.encoding_layers(mixed)
        taps["pooled"] = pooled[0]
        embedding = net.linear(net.bn(pooled))[0]
        whole = model(audio, 44100)[0]
    if not torch.equal(embedding, whole):
        raise SystemExit("the staged run is not the model's forward")
    np.save(out / "audio.npy", audio[0].numpy())
    np.save(out / "wav16.npy", wav[0].numpy())
    np.save(out / "fbank.npy", feats.numpy())
    np.save(out / "embedding.npy", embedding.numpy())
    save(out, {k: v.numpy() for k, v in taps.items()})
    print(f"{audio.shape[1] / 44100:.2f} s, {feats.shape[0]} frames, embedding norm {float(embedding.norm()):.4f} -> {out}")


def web_aligner(folder):
    """The release's web/aligner export as fusealign_hub's modules in fp32: 8-bit matrices times their scales."""
    from fusealign_hub.model import FuseAlign, FuseAlignArgs
    from fusealign_hub.text_encoder import DiaTextEncoder

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    meta, entries = manifest["meta"], manifest["tensors"]
    shards = [np.memmap(folder / name, dtype=np.uint8, mode="r") for name in manifest["shards"]]

    def get(name):
        e = entries[name]
        raw = shards[e["shard"]]
        shape = list(reversed(e["shape"]))  # the manifest lists ggml's order, innermost first
        if e["dtype"] == "f32":
            return torch.from_numpy(np.frombuffer(raw[e["offset"] : e["offset"] + e["nbytes"]], dtype="<f4").reshape(shape).copy())
        if e["dtype"] != "q8":
            raise SystemExit(f"{name}: unexpected dtype {e['dtype']}")
        q = np.frombuffer(raw[e["offset"] : e["offset"] + e["nbytes"]], dtype=np.int8).astype(np.float32).reshape(-1, 32)
        scales = np.frombuffer(raw[e["scales_offset"] : e["scales_offset"] + e["scales_nbytes"]], dtype="<f4")
        return torch.from_numpy((q * scales[:, None]).reshape(shape))

    dim, feat, hop, nup = meta["dim"], meta["feat_dim"], meta["patch_hop"], meta["n_upsample"]
    text = {"embedding.weight": get("dia.embedding"), "norm.weight": get("dia.norm")}
    for i in range(meta["dia_layers"]):
        q, k, v = get(f"dia.{i}.qkv").chunk(3)
        pre = f"layers.{i}."
        text.update({pre + "pre_sa_norm.weight": get(f"dia.{i}.pre_norm"), pre + "post_sa_norm.weight": get(f"dia.{i}.post_norm")})
        text.update({pre + "self_attention.q_proj.weight": q, pre + "self_attention.k_proj.weight": k, pre + "self_attention.v_proj.weight": v})
        text.update({pre + "self_attention.o_proj.weight": get(f"dia.{i}.o"), pre + "mlp.gate_up_proj.weight": get(f"dia.{i}.gate_up")})
        text[pre + "mlp.down_proj.weight"] = get(f"dia.{i}.down")
    config = {"hidden_size": meta["dia_hidden"], "num_hidden_layers": meta["dia_layers"], "num_attention_heads": meta["dia_heads"]}
    config.update({"num_key_value_heads": meta["dia_heads"], "head_dim": meta["dia_head_dim"], "intermediate_size": meta["dia_intermediate"]})
    config.update({"norm_eps": meta["dia_norm_eps"], "rope_theta": meta["dia_rope_theta"], "max_position_embeddings": meta["dia_max_positions"], "vocab_size": 256})
    encoder = DiaTextEncoder(config, text)
    # the export folds mod_emb into the two input biases
    state = {"mod_emb": torch.zeros(2, dim), "audio_patch.bias": get("fa.audio_bias"), "text_proj.bias": get("fa.text_bias")}
    state["audio_patch.weight"] = get("fa.audio_patch").permute(1, 0, 2).reshape(dim, 1, 2 * hop)
    state["text_proj.weight"] = get("fa.text_proj")
    for i in range(meta["n_layers"]):
        w1, w3 = get(f"fa.{i}.w13").chunk(2)
        pre = f"blocks.{i}."
        state.update({pre + "attn_norm.weight": get(f"fa.{i}.attn_norm"), pre + "attn.qkv.weight": get(f"fa.{i}.qkv"), pre + "attn.out.weight": get(f"fa.{i}.out")})
        state.update({pre + "ffn_norm.weight": get(f"fa.{i}.ffn_norm"), pre + "ffn.w1.weight": w1, pre + "ffn.w3.weight": w3, pre + "ffn.w2.weight": get(f"fa.{i}.w2")})
    state["final_norm.weight"] = get("fa.final_norm")
    for head in ("start", "end"):
        pre = f"{head}_refinement."
        state.update({pre + "0.weight": get(f"fa.{head}.in").reshape(feat, dim, 1), pre + "0.bias": get(f"fa.{head}.in_b")})
        for s in range(nup):
            even, odd = get(f"fa.{head}.up{s}.even"), get(f"fa.{head}.up{s}.odd")  # [tap][out][in]
            up = torch.stack((odd[1].T, even[1].T, odd[0].T, even[0].T), dim=2)  # [in][out][4]
            state.update({pre + f"{1 + 2 * s}.weight": up.contiguous(), pre + f"{1 + 2 * s}.bias": get(f"fa.{head}.up{s}.b")})
        k = 1 + 2 * nup
        state.update({pre + f"{k}.weight": get(f"fa.{head}.smooth").permute(1, 2, 0).contiguous(), pre + f"{k}.bias": get(f"fa.{head}.smooth_b")})
        state.update({pre + f"{k + 2}.weight": get(f"fa.{head}.out").reshape(feat, feat, 1), pre + f"{k + 2}.bias": get(f"fa.{head}.out_b")})
    first, bias = get("fa.mlp0").chunk(3), get("fa.mlp0_b").chunk(3)
    for j, name in enumerate(("start", "end", "presence")):
        state.update({f"{name}_mlp.0.weight": first[j].contiguous(), f"{name}_mlp.0.bias": bias[j].contiguous()})
        state.update({f"{name}_mlp.2.weight": get(f"fa.{name}_mlp2"), f"{name}_mlp.2.bias": get(f"fa.{name}_mlp2_b")})
    args = FuseAlignArgs(dim=dim, n_layers=meta["n_layers"], n_heads=meta["n_heads"], ffn_dim=meta["ffn_dim"], feat_dim=feat, patch_window=meta["patch_window"], patch_hop=hop, n_upsample=nup, rope_theta=meta["rope_theta"])
    net = FuseAlign(args)
    net.load_state_dict(state, strict=True)
    return net.eval(), encoder


def aligner(args):
    """fusealign_hub's aligner on the web/aligner weights for 44.1 kHz audio and its tokens: words.json and taps."""
    from fusealign_hub.aligner import FuseAlignAligner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net, encoder = web_aligner(args.weights)
    hub = FuseAlignAligner(net, encoder, batch_size=1)
    audio = torch.as_tensor(np.load(args.audio), dtype=torch.float32).reshape(-1)
    tokens = json.loads(Path(args.tokens).read_text())
    taps = {}
    hooks = [encoder.register_forward_hook(lambda m, i, o: taps.__setitem__("text_feats", o[0].float().numpy()))]
    hooks.append(net.final_norm.register_forward_hook(lambda m, i, o: taps.__setitem__("joint", o[0].float().numpy())))
    decoded = {}
    forward = hub._forward_group

    def logged(items):
        results = forward(items)
        decoded["intervals"], decoded["presence"] = results[0]
        return results

    hub._forward_group = logged
    with torch.inference_mode():
        words = hub.align(audio, 44100, tokens)
    for h in hooks:
        h.remove()
    np.save(out / "audio.npy", audio.numpy())
    save(out, taps)
    np.save(out / "presence.npy", np.asarray(decoded["presence"], dtype=np.float32))
    ms = [(-1, -1) if iv is None else (round(iv[0] * 1000), round(iv[1] * 1000)) for iv in decoded["intervals"]]
    np.save(out / "start_ms.npy", np.asarray([a for a, _ in ms], dtype=np.int32))
    np.save(out / "end_ms.npy", np.asarray([b for _, b in ms], dtype=np.int32))
    listed = [{"word": w.word, "start": w.start, "end": w.end, "presence": w.presence, "placed": w.placed} for w in words]
    (out / "words.json").write_text(json.dumps({"tokens": tokens, "words": listed}) + "\n")
    print(f"{len(audio) / 44100:.2f} s, {len(tokens)} tokens, {sum(w.placed for w in words)} placed -> {out}")


def clips(args):
    """The served checks' two clips: the audio of two `rows` dumps as 16-bit WAVs, OUT/short.* and OUT/long.*."""
    import wave

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, dump, text in (("short", args.short, TEXT), ("long", args.long, " ".join([TEXT] * 4))):
        audio = np.fromfile(Path(dump) / "audio.f32", dtype="<f4")
        pcm = np.clip(np.round(audio * 32767), -32768, 32767).astype("<i2")
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(pcm.tobytes())
        np.save(out / f"{name}_audio.npy", pcm.astype(np.float32) / 32768.0)
        (out / f"{name}_tokens.json").write_text(json.dumps(text.split()))
        print(f"{name}: {len(pcm) / 44100:.2f} s -> {out}")


def served_cases(args):
    """The served checks' edits of those clips: OUT/<name>.json, `request` for the reference, `served` for us."""
    from zonos3.tts import textmatch
    from zonos3.tts.edit_align import aligned_words

    out = Path(args.out)
    voice = str(Path(args.release) / f"{VOICE}.embedding.npy")
    long_text = " ".join([TEXT] * 4)
    seconds = {name: len(np.load(out / f"{name}_clip" / "codes.npy")) / 15.0 for name in ("short", "long")}
    texts = {"short": TEXT, "long": long_text}

    def spread(name, lead=0.2):
        parts, total, timed, at = texts[name].split(), sum(len(w) for w in texts[name].split()), [], 0
        for w in parts:
            timed.append({"word": w, "start": lead + (seconds[name] - lead) * at / total, "end": lead + (seconds[name] - lead) * (at + len(w)) / total})
            at += len(w)
        return timed

    def aligned(name):
        found = json.loads((out / f"align_{name}" / "words.json").read_text())["words"]
        words = aligned_words(found, [textmatch.tokens(w["word"]) for w in found])
        return [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence, "tokens": w.tokens} for w in words]

    g = {"temperature": 0.0, "seed": 7}
    cat = TEXT.replace("fox", "cat")
    late = long_text.replace("fox", "cat", 1)[::-1].replace("revir", "egdirb", 1)[::-1]
    diff = TEXT.replace("brown fox", "small grey cat").replace("rests", "sleeps")
    three = {"target_text": TEXT.replace("quick", "slow").replace("lazy", "sleepy").replace("river", "bridge"), "spans": [[0.5, 0.9], [2.4, 2.9], [3.9, 4.4]]}
    three.update({"controls": {"mood.valence": 1.0}, "region_controls": [None, {"delivery.whisper": 1.5}, {"identity.age": 1.0, "mood.valence": -1.0}]})
    three.update({"steer_onset_seconds": 0.2, "condition_values": {"speaking_rate": 18.0}})
    # (clip, the request, whose embedding fills the speaker row, whether the server aligns the words itself)
    cases = {
        "s_span": ("short", {"target_text": cat, "spans": [[1.2, 1.8]], **g}, "voice", False),
        "s_plain": ("short", {"target_text": cat, "spans": [[1.2, 1.8]], "speaker_embedding": False, "min_fill_ratio": 0.0, **g}, "voice", False),
        "s_three": ("short", {**three, **g}, "voice", False),
        "s_words": ("short", {"target_text": diff, "original_text": TEXT, "original_words": spread("short"), **g}, "voice", False),
        "s_windows": ("long", {"target_text": late, "original_text": long_text, "original_words": spread("long"), **g}, "voice", False),
        "s_window_spans": ("long", {"target_text": late, "original_text": long_text, "original_words": spread("long"), "spans": [[1.2, 1.8], [seconds["long"] - 1.2, seconds["long"] - 0.4]], "context_seconds": 2.0, **g}, "voice", False),
        "v_span": ("short", {"target_text": cat, "spans": [[1.2, 1.8]], **g}, "clip", False),
        "v_windows": ("long", {"target_text": late, "original_text": long_text, "original_words": spread("long"), **g}, "clip", False),
        "a_span": ("short", {"target_text": cat, "original_text": TEXT, "spans": [[1.45, 1.85]], **g}, "clip", True),
        "a_words": ("short", {"target_text": diff, "original_text": TEXT, **g}, "clip", True),
        "a_windows": ("long", {"target_text": late, "original_text": long_text, **g}, "clip", True),
    }
    for name, (clip, request, row, aligns) in cases.items():
        embedding = voice if row == "voice" else str(out / f"speaker_{clip}" / "embedding.npy")
        case = {"codes": str(out / f"{clip}_clip" / "codes.npy"), "embedding": embedding, "wav": str(out / f"{clip}.wav"), "served": request}
        case["own_embedding"] = row == "clip"
        case["request"] = {**request, "original_words": aligned(clip)} if aligns else request
        (out / f"{name}.json").write_text(json.dumps(case))
    print(f"{len(cases)} served cases -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("edit-cases")
    p.add_argument("release")
    p.add_argument("short", help="a `rows` dump of the greedy_embedding request")
    p.add_argument("long", help="a `rows` dump of the laci_long request")
    p.add_argument("out")
    p.set_defaults(run=edit_cases)
    p = commands.add_parser("clips")
    p.add_argument("short", help="a `rows` dump of the greedy_embedding request")
    p.add_argument("long", help="a `rows` dump of the laci_long request")
    p.add_argument("out")
    p.set_defaults(run=clips)
    p = commands.add_parser("served-cases")
    p.add_argument("release")
    p.add_argument("out", help="the `clips` folder, with each clip's `clip`, `speaker` and `aligner` dumps beside it")
    p.set_defaults(run=served_cases)
    p = commands.add_parser("edit")
    p.add_argument("lm")
    p.add_argument("adapter", help="the release's lm/zonos3_mini/edit_lora/z3m_omni3b_edit_lora.pt")
    p.add_argument("case")
    p.add_argument("out")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steer-bundle")
    p.add_argument("--laci-heads")
    p.add_argument("--text-rows")
    p.set_defaults(run=edit)
    p = commands.add_parser("speaker")
    p.add_argument("checkpoint", help="the release's speaker_encoders/ownmect_s2/ckpt_0037.pth")
    p.add_argument("audio", help="44.1 kHz mono float32 .npy")
    p.add_argument("out")
    p.set_defaults(run=speaker)
    p = commands.add_parser("aligner")
    p.add_argument("weights", help="the release's web/aligner folder")
    p.add_argument("audio", help="44.1 kHz mono float32 .npy")
    p.add_argument("tokens", help="a JSON list of the transcript's tokens")
    p.add_argument("out")
    p.set_defaults(run=aligner)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
