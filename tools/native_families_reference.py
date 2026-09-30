"""Python numerical oracle for native Nemotron/Flash Next; never launches native code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def deepseek_fixture(directory, output, wide=False, packed=False):
    import mlx.core as mx
    if wide:
        from tests import dsv4_fakes as fake
        fake.D = 4096
        fake.TEXT.update(hidden_size=4096, num_hidden_layers=3, num_attention_heads=64, head_dim=512,
                         q_lora_rank=512, index_head_dim=128, n_routed_experts=16, moe_intermediate_size=512,
                         compress_ratios=[0, 4, 128, 0])
        if packed:
            original_hc = fake._hc
            def packed_hc(t, name, mixes):
                original_hc(t, name, mixes)
                t[f"{name}.fn"] = t[f"{name}.fn"].astype(mx.bfloat16)
            fake._hc = packed_hc
    from tests.dsv4_fakes import write_checkpoint, write_mtp
    from tensorfold.families.deepseek_v4.weights import load_backbone
    from tensorfold.families.deepseek_v4.model import Block
    from tensorfold.families.glm5_next.model import hc_expand
    write_checkpoint(directory)
    write_mtp(directory / "drafter")
    (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}, "merges": []}, "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}}))
    output.mkdir(parents=True, exist_ok=True)
    model = load_backbone(directory)
    cache = model.make_cache()
    from tensorfold.families.deepseek_v4.mtp import load as load_mtp, MTPCache
    mtp = load_mtp(model, directory / "drafter/model.safetensors")
    mtp_cache = MTPCache(model.args.sliding_window)
    previous_streams = None
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    for i, layer in enumerate(model.layers):
        save(f"frequencies-{i}", layer.attn.inv_freq)
    layer_ids = {id(layer): i for i, layer in enumerate(model.layers)}
    if wide:
        from tensorfold.families.deepseek_v4.attention import Attention
        from tensorfold.families.deepseek_v4.moe import MoE
        attn_ids = {id(layer.attn): i for i, layer in enumerate(model.layers)}
        moe_ids = {id(layer.moe): i for i, layer in enumerate(model.layers)}
        attn_call, moe_call = Attention.__call__, MoE.__call__
        layer_positions = {}
        def traced_attention(self, x, caches, lengths, decode, positions=None):
            if id(self) in attn_ids:
                layer = attn_ids[id(self)]
                position = caches[0].offset
                layer_positions[layer] = position
                save(f"trace-{position}-{layer}-attn-input", x)
            out = attn_call(self, x, caches, lengths, decode, positions)
            if id(self) in attn_ids:
                save(f"trace-{position}-{layer}-attn-output", out)
            return out
        def traced_moe(self, x, ids, decode):
            if id(self) in moe_ids:
                layer = moe_ids[id(self)]
                position = layer_positions[layer]
                save(f"trace-{position}-{layer}-ffn-input", x)
            out = moe_call(self, x, ids, decode)
            if id(self) in moe_ids:
                save(f"trace-{position}-{layer}-ffn-output", out)
            return out
        Attention.__call__, MoE.__call__ = traced_attention, traced_moe
    def traced(self, x, ids, caches, lengths, decode, positions=None):
        if id(self) not in layer_ids:
            return original(self, x, ids, caches, lengths, decode, positions)
        layer = layer_ids[id(self)]
        position = caches[0].offset
        xc, post, comb = self.attn_hc.split(x, decode)
        ax = mx.fast.rms_norm(xc, self.attn_norm, self.eps)
        save(f"trace-{position}-{layer}-attn-input", ax)
        branch = self.attn(ax, caches, lengths, decode, positions)
        save(f"trace-{position}-{layer}-attn-output", branch)
        x = hc_expand(branch, x, post, comb, decode)
        xc, post, comb = self.ffn_hc.split(x, decode)
        fx = mx.fast.rms_norm(xc, self.ffn_norm, self.eps)
        save(f"trace-{position}-{layer}-ffn-input", fx)
        branch = self.moe(fx, ids, decode)
        save(f"trace-{position}-{layer}-ffn-output", branch)
        x = hc_expand(branch, x, post, comb, decode)
        save(f"trace-{position}-{layer}-streams", x)
        return x
    original = Block.__call__
    Block.__call__ = traced
    try:
        for position in range(137):
            hidden = model.hidden(mx.array([[position % 250 + 1]], dtype=mx.uint32), cache)
            save(f"hidden-{position}", hidden[0])
            save(f"logits-{position}", model.head(hidden)[0])
            if previous_streams is not None:
                drafted = mtp(model, previous_streams, mx.array([position % 250 + 1], dtype=mx.uint32), [mtp_cache], (1,), True)
                save(f"mtp-streams-{position}", drafted)
                save(f"mtp-logits-{position}", mtp.logits(model, drafted))
            previous_streams = model.last_streams
            end = position + 1
            if end % 16 == 0 or end == 137:
                for i, c in enumerate(cache):
                    save(f"cache-{end}-{i}-keys", c.window_keys(position))
                    ratio = model.args.ratio(i)
                    if ratio:
                        lo = max(0, end - ratio * (2 if ratio == 4 else 1))
                        save(f"cache-{end}-{i}-proj", c.proj_rows(lo, end))
                        if end // ratio:
                            save(f"cache-{end}-{i}-pool", c.pool[:end // ratio])
                            if ratio == 4:
                                save(f"cache-{end}-{i}-ipool", c.ipool[:end // ratio])
    finally:
        Block.__call__ = original
        if wide:
            Attention.__call__, MoE.__call__ = attn_call, moe_call
    from tensorfold.engine.exact_sampling import Sampling, sample_rows
    for temperature in (0.0, 0.8):
        settings = Sampling(seed=1234, temperature=temperature, top_k=20, top_p=0.95)
        cache = model.make_cache()
        prompt = [1, 2, 3, 4]
        logits = model.head(model.hidden(mx.array([prompt], dtype=mx.uint32), cache))[0]
        token = int(sample_rows(logits[-1:], [len(prompt)], settings)[0])
        generated = []
        for step in range(12):
            generated.append(token)
            if token in model.args.eos_token_id:
                break
            logits = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache))[0]
            token = int(sample_rows(logits, [len(prompt) + step + 1], settings)[0])
        save(f"generated-{int(temperature > 0)}", mx.array(generated, dtype=mx.int32))
    print("Saved DeepSeek backbone oracle through 137 tokens", flush=True)


def deepseek_dspark_fixture(directory, output, sorted_experts=False, wide=False):
    import mlx.core as mx
    from tests import dsv4_fakes as fake
    from tensorfold.families.deepseek_v4.weights import load_backbone
    from tensorfold.families.deepseek_v4.dspark import load as load_dspark
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.gpu_sampling import sample
    if wide:
        fake.D = 4096
        fake.TEXT.update(hidden_size=4096, num_hidden_layers=3, num_attention_heads=64, head_dim=512,
                         q_lora_rank=512, index_head_dim=128, n_routed_experts=16, moe_intermediate_size=512,
                         compress_ratios=[0, 4, 128, 0])
        fake.DSPARK["dspark_target_layer_ids"] = [0, 1, 2]
    if sorted_experts:
        fake.TEXT["num_experts_per_tok"] = 4
        fake.DSPARK["dspark_block_size"] = 16
    fake.write_checkpoint(directory)
    fake.write_dspark(directory / "drafter")
    (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}, "merges": []}, "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}}))
    model = load_backbone(directory)
    drafter = load_dspark(model, directory / "drafter/model.safetensors", fake.DSPARK)
    model.tap_layers = drafter.taps
    target_cache, cache = model.make_cache(), drafter.make_cache()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    position = 0
    for round_id, count in enumerate((3, 5, 16, 7)):
        ids = mx.array([[1 + (position + j) % 250 for j in range(count)]], dtype=mx.uint32)
        hidden = model.hidden(ids, target_cache)
        save(f"target-{round_id}", model.head(hidden)[0])
        save(f"taps-{round_id}", model.last_taps)
        drafter.absorb(model.last_taps, cache)
        position += count
        for i, c in enumerate(cache):
            save(f"keys-{round_id}-{i}", c.ring_rows(c.keys, max(0, c.offset - drafter.window), c.offset))
        token = mx.array([55 + round_id], dtype=mx.uint32)
        logits = drafter.logits(model, token, cache)
        save(f"draft-logits-{round_id}", logits)
        for mode, temperature in enumerate((0.0, 0.8)):
            settings = Sampling(seed=1234, temperature=temperature, top_k=20, top_p=0.95)
            draws = drafter.draw(logits, token, drafter.size, lambda row, j: sample(row, None if temperature == 0 else settings, [position + j + 1]))
            save(f"draw-{round_id}-{mode}", draws)
    print("Saved DSpark target taps, context rings, block logits and Markov draws", flush=True)


def dflash_fixture(directory, output, case):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from tensorfold.drafters.dflash_drafter import _vendor
    from tensorfold.drafters.dflash_attention import _dflash_attend, concat_updates
    from tensorfold.drafters.dflash_block import _parts
    from types import SimpleNamespace
    vendor = _vendor()
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(421 + case)
    width, vocab = (2816, 262144) if case == 3 else (128, 256)
    bits = (8, 0, 4, 8)[case]
    rope = dict(rope_type="proportional", partial_rotary_factor=.5, factor=2.) if case == 1 else dict(rope_type="linear", factor=2.) if case == 2 else None
    config = dict(hidden_size=width, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=64, intermediate_size=256, vocab_size=vocab, rms_norm_eps=1e-6,
                  rope_theta=10000., max_position_embeddings=262144, num_target_layers=30,
                  layer_types=["sliding_attention", "full_attention"], sliding_window=17,
                  is_causal=case == 2, rope_scaling=rope,
                  dflash_config=dict(block_size=16, target_layer_ids=[4, 14, 24], mask_token_id=100,
                                     input_embedding_scale=.5, output_multiplier=.75, final_logit_softcapping=30.))
    dc = vendor.DFlashConfig(**{k: v for k, v in config.items() if k != "dflash_config"},
                             **config["dflash_config"])
    model = vendor.DFlashDraftModel(dc)
    model.set_dtype(mx.bfloat16)
    raw = dict(tree_flatten(model.parameters()))
    mx.eval(raw)
    if case != 2:
        mx.save_safetensors(str(directory / "model.safetensors"), raw)
    if bits:
        nn.quantize(model, group_size=64, bits=bits, class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    mx.eval(model.parameters())
    if case == 2:
        mx.save_safetensors(str(directory / "model.safetensors"), dict(tree_flatten(model.parameters())))
        config["quantization"] = dict(bits=bits, group_size=64)
    (directory / "config.json").write_text(json.dumps(config))
    inputs = {}
    cache = concat_updates(model.make_cache())
    draft = SimpleNamespace(model=model)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    for step, count in enumerate((3, 5, 23, 1, 16)):
        taps = (mx.random.normal((1, count, 3 * width)) * .3).astype(mx.bfloat16)
        embeddings = (mx.random.normal((1, 16, width)) * .2).astype(mx.bfloat16)
        mx.eval(taps, embeddings)
        inputs[f"taps-{step}"] = taps[0]
        inputs[f"embeddings-{step}"] = embeddings
        context = model.hidden_norm(model.fc(taps))
        h = embeddings
        for (pre, post), layer, item in zip(_parts(draft), model.layers, cache):
            h = post(h, _dflash_attend(layer.self_attn, pre(h), context, model.rope, item, {}))
        save(f"hidden-{step}", model.norm(h[:, 1:]))
        for i, item in enumerate(cache):
            keys, values = item.state
            save(f"keys-{step}-{i}", keys)
            save(f"values-{step}-{i}", values)
    (directory / "fixtures").mkdir(exist_ok=True)
    (directory / "inputs.safetensors").unlink(missing_ok=True)
    mx.save_safetensors(str(directory / "fixtures/inputs.safetensors"), inputs)
    print(f"Saved DFlash case {case}: float/quantized blocks and sliding/full caches", flush=True)


def gemma_dflash_fixture(directory, draft_dir, output):
    import mlx.core as mx
    from tensorfold.families.gemma4.model import load
    from tensorfold.families.qwen3_5.dflash_head import _Context
    model, _ = load(directory, backend="rows", check=False, drafter=str(draft_dir))
    cache = model.make_cache()
    draft = model.mtp
    proposer = draft.proposer(sampling=None)
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    position = 0
    for step, (count, budget) in enumerate(zip((3, 5, 16, 1, 16), (3, 15, 1, 7, 15))):
        ids = mx.array([[1000 + position + j for j in range(count)]], dtype=mx.uint32)
        hidden = model.hidden(ids, cache)
        save(f"target-{step}", model.head(hidden)[0])
        taps = draft.taps()
        save(f"taps-{step}", taps[0])
        position += count
        if not proposer.ready:
            proposer.prefill_taps(position, taps)
        else:
            proposer.absorb(taps)
        proposal = proposer.propose(_Context(position + 1, 42), budget)
        save(f"proposal-{step}", mx.array([42] + proposal))
        for i, item in enumerate(proposer.cache):
            save(f"keys-{step}-{i}", item.state[0])
            save(f"values-{step}-{i}", item.state[1])
    print("Saved full Gemma target taps and DFlash proposals", flush=True)


def glm_prefill_fixture(directory, output, custom_tiles=False):
    import mlx.core as mx
    from tensorfold.families.glm5_next.weights import load_backbone
    from tensorfold.families.glm5_next.model import Layer
    from tensorfold.families.glm5_next.mtp import load as load_mtp
    from tensorfold.families.glm5_next.linear import project
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    if custom_tiles:
        prefill_mm._tiles[:] = [True]
    model = load_backbone(directory)
    cache = model.make_cache()
    head = load_mtp(model)
    head_cache = head.make_cache()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    def save_cache(prefix, item):
        for key, attr in (("conv", "conv"), ("state", "ssm"), ("keys", "keys"), ("ik", "ik"), ("ig", "ig"), ("pool", "pool")):
            value = getattr(item, attr, None)
            if value is not None:
                if key in ("keys", "ik", "ig", "pool"):
                    value = value[:item.offset // model.args.index_kpool if key == "pool" else item.offset]
                save(prefix + key, value)
    original = Layer.__call__
    layer_ids = {id(layer): i for i, layer in enumerate(model.layers)}
    position = 0
    def record_layer(self, x, *args, **kwargs):
        out = original(self, x, *args, **kwargs)
        if id(self) in layer_ids and x.shape[0] > 16:
            save(f"trace-{position}-{layer_ids[id(self)]}", out)
        return out
    Layer.__call__ = record_layer
    try:
        for step, count in enumerate((17, 63, 64, 511, 512, 513, 2048, 17, 1)):
            tokens = mx.array([1 + (position + j) % 97 for j in range(count)])
            next_tokens = mx.array([1 + (position + j + 1) % 97 for j in range(count)])
            hidden = model.hidden(tokens, cache)[0]
            save(f"hidden-{step}", hidden)
            save(f"logits-{step}", model.head(hidden[-1:]))
            position += count
            for i, item in enumerate(cache):
                save_cache(f"cache-{step}-{i}-", item)
            projected_input = mx.concatenate([mx.fast.rms_norm(model.embed_tokens(next_tokens), head.enorm, head.eps),
                                               mx.fast.rms_norm(hidden, head.hnorm, head.eps)], axis=-1)
            projected = project(projected_input, head.eh_proj, rows_exact=count <= 16)
            out = head(model, hidden, next_tokens, [head_cache], (count,), count <= 16)
            save(f"head-{step}-mtp_input", projected_input)
            save(f"head-{step}-mtp_projection", projected)
            save(f"head-{step}-hidden", out)
            save(f"head-{step}-logits", head.logits(model, out))
            if step == 7:
                head_cache.trim(3)
                replay = head(model, hidden[-3:], next_tokens[-3:], [head_cache], (3,), True)
                save("head-replay", head.logits(model, replay))
            save_cache(f"head-cache-{step}-", head_cache)
            print(f"GLM prefill and draft oracle at {position} tokens", flush=True)
        for step in range(4):
            save(f"continuation-{step}", model.head(model.hidden(mx.array([200 + step]), cache))[0])
    finally:
        Layer.__call__ = original


def flash_prefill_fixture(directory, output, simd=False, custom_tiles=False):
    import mlx.core as mx
    from tensorfold.families.qwen4_exp.model import load, select_by_kernels
    from tensorfold.families.qwen4_exp import decode
    from tensorfold.families.qwen4_exp.runtime import FlashNext
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from types import SimpleNamespace
    decode.DENSE = "rows"
    if simd:
        prefill_mm._tensor_units = lambda: False
    if custom_tiles:
        prefill_mm._tiles[:] = [True]
    model, _ = load(directory, lazy=True, ple_on_ssd=True)
    model.__dict__["fused"] = decode.FusedDecode(model)
    select_by_kernels(model.layers)
    runtime = SimpleNamespace(model=model)
    cache = model.make_cache()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    position = 0
    for step, count in enumerate((17, 63, 64, 2048, 2048, 17, 1)):
        tokens = np.array([[1000 + (position + j) % 37 for j in range(count)]], dtype=np.int64)
        hidden = model.hidden(tokens, cache)
        streams = model.fused.last_streams if count <= model.fused_rows else model.__dict__["last_streams"]
        save(f"hidden-{step}", streams)
        save(f"logits-{step}", FlashNext.head(runtime, hidden[:, -1:])[0])
        position += count
        for i, (layer, item) in enumerate(zip(model.layers, cache)):
            if layer.is_linear:
                save(f"cache-{step}-{i}-0", item.conv[0])
                save(f"cache-{step}-{i}-1", item.ssm[0])
            else:
                save(f"cache-{step}-{i}-0", item.keys[:, :, :item.offset])
                save(f"cache-{step}-{i}-1", item.values[:, :, :item.offset])
                save(f"raw-{step}-{i}", item.index_keys[0, :item.offset])
                if item.pooled is not None:
                    save(f"pooled-{step}-{i}", item.pooled[0])
            if "ple" in layer:
                save(f"ple-{step}", item.ple_conv[0])
                save(f"history-{step}", mx.array(item.history[0]))
        print(f"Flash prefill oracle at {position} tokens", flush=True)
    for step in range(4):
        save(f"continuation-{step}", FlashNext.head(runtime, model.hidden(np.array([[2000 + step]]), cache))[0])


def nemotron_prefill_fixture(directory, output, simd=False):
    import mlx.core as mx
    from mlx_lm import load
    from tensorfold.families.nemotron_h.model import NemotronH
    from tensorfold.kernels.nemotron.lightning.v1.kernels import FusedDecode
    from tensorfold.kernels.nemotron.lightning.v1 import kernels, rows
    if simd:
        kernels.tensor_units = lambda: False
    model = NemotronH.__new__(NemotronH)
    model.model, _ = load(str(directory))
    model.args = model.model.args
    model.fused = FusedDecode(model.model)
    model.mtp = None
    if kernels.tensor_units():
        model._install_lane_matmul()
    else:
        rows.install(model)
    cache = model.make_cache()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    position = 0
    for step, count in enumerate((17, 255, 16, 257, 2048, 1)):
        tokens = mx.array([[1000 + (position + j) % 37 for j in range(count)]], dtype=mx.uint32)
        hidden = model.hidden(tokens, cache)
        save(f"hidden-{step}", hidden[0])
        save(f"logits-{step}", model.head(hidden[:, -1:])[0])
        position += count
        index = 0
        for i, layer in enumerate(model.layers):
            if layer.block_type not in "M*":
                continue
            state = cache[index].state
            for j, value in enumerate(state):
                save(f"cache-{step}-{i}-{j}", value)
            index += 1
        print(f"Nemotron prefill oracle at {position} tokens", flush=True)
    for step in range(4):
        save(f"continuation-{step}", model.head(model.hidden(mx.array([[2000 + step]], dtype=mx.uint32), cache))[0])


def gemma_prefill_fixture(directory, output):
    import mlx.core as mx
    from tensorfold.families.gemma4.model import load
    model, _ = load(directory, backend="rows", check=False)
    cache = model.make_cache()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        np.save(output / f"{name}.npy", np.asarray(value.astype(mx.float32)))
    position = 0
    for step, count in enumerate((1, 7, 129, 1024, 2048, 3)):
        tokens = mx.array([[1000 + (position + j) % 37 for j in range(count)]], dtype=mx.uint32)
        hidden = model.prefill(tokens, cache)
        save(f"hidden-{step}", hidden[0])
        save(f"logits-{step}", model.head(hidden[:, -1:])[0])
        position += count
        for i, item in enumerate(cache):
            keys, values = item.state
            if not item.ring:
                keys, values = keys[:, :, :position], values[:, :, :position]
            save(f"keys-{step}-{i}", keys)
            save(f"values-{step}-{i}", values)
        print(f"Gemma prefill oracle at {position} tokens", flush=True)
    for step in range(4):
        save(f"continuation-{step}", model.head(model.hidden(mx.array([[2000 + step]], dtype=mx.uint32), cache))[0])


def chat_fixture(directory, output):
    from transformers import AutoTokenizer
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from tensorfold.server.text import render_prompt_ids
    from tensorfold.engine.call_gate import CallGate, call_format
    from tensorfold.engine.lane_engine import LaneStream
    from tensorfold.engine.lane_family import FamilyRounds
    from tensorfold.server.request_options import RequestOptions, thinking_fields
    from tensorfold.server.app import ChatApp
    from tensorfold.server.text import template_late_system
    from threading import Lock
    tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(str(directory), local_files_only=True))
    model_type = json.loads((directory / 'config.json').read_text()).get('model_type')
    deepseek = model_type == 'deepseek_v4'
    if deepseek:
        from tensorfold.families.deepseek_v4.prompts import DeepSeekTokenizer
        tokenizer = DeepSeekTokenizer(tokenizer)
    if model_type == 'glm5_next':
        from tensorfold.families.glm5_next.prompts import GlmTokenizer
        tokenizer = GlmTokenizer(tokenizer)
    openers = ("<tool_call>", "<|tool_call>", "<｜DSML｜tool_calls>")
    probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
    form = call_format(tokenizer.decode(render_prompt_ids(tokenizer, probe, add_generation_prompt=False)), "tfprobe_fn", openers)
    def token_id(text):
        token = tokenizer.convert_tokens_to_ids(text)
        return token if isinstance(token, int) and token >= 0 and token != tokenizer.unk_token_id else -1
    if form is None:
        form = next(((opener, None, None) for opener in openers if token_id(opener) >= 0), None)
    is_gemma = "gemma" in directory.name.lower()
    think_open = tokenizer.encode("<|channel>thought", add_special_tokens=False)[0] if is_gemma else token_id("<think>")
    think_end = token_id("<channel|>" if is_gemma else "</think>")
    eos = tokenizer.eos_token_id
    options = RequestOptions()
    options.tokenizer = tokenizer
    options.tokenizer_lock = Lock()
    options._think_tokens = None
    budget_close, budget_end = options._think_close()
    def gate_cases(prompt):
        if form is None:
            return []
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        decode = lambda token: tokenizer.decode([token], skip_special_tokens=False)
        names = ["weather", "forecast"]
        result = []
        for script in ("I refuse to call a tool.", "<think>reason</think>" + form[0] + (form[1] or "") + "wrong" + (form[2] or "")):
            proposed = ([eos] + encode(script)) * 3
            for budget in (-1, 0, 1, 2, 5, 10):
                for required in (False, True):
                    if required and token_id(form[0]) < 0:
                        continue
                    gate = CallGate.after_prompt(prompt, token_id(form[0]), lambda token: token != eos and not decode(token).strip(), think_open=think_open, think_end=think_end, text=decode, encode=encode, lead=form[1] or "", names=names if form[1] is not None else (), tail=form[2] or "") if required else None
                    stream = LaneStream("fixture", prompt, len(proposed), call_gate=gate, think_budget=budget, think_end=budget_end, think_close=budget_close, think_open=budget > 0 and budget_end >= 0)
                    for token in proposed:
                        forced = FamilyRounds._forced_next(stream, np.array(token))
                        stream.commit([token if forced is None else forced])
                    result.append({"names": names, "required": required, "budget": budget, "proposed": proposed, "eos": eos, "expected": stream.emitted})
        return result
    tool = {"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    conversations = [
        [{"role": "user", "content": "Hello, æøå 世界 👋\n123456789"}],
        [{"role": "system", "content": "Be brief."}, {"role": "developer", "content": "Use Danish."}, {"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Next?"}],
        [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]}],
        [{"role": "user", "content": "Weather in Copenhagen?"}],
        [{"role": "user", "content": "Weather in Copenhagen?"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Copenhagen"}'}}]}, {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "Sunny"}, {"role": "user", "content": "Summarize."}],
    ]
    cases = []
    app = ChatApp.__new__(ChatApp)
    app.tokenizer = tokenizer
    app.tokenizer_lock = Lock()
    app.late_system = template_late_system(tokenizer)
    for thinking, effort in ((False, None), (True, "low"), (True, "medium"), (True, "xhigh")):
        for index, messages in enumerate(conversations):
            tools = [tool] if index >= 4 else None
            body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}}
            if effort:
                body["reasoning_effort"] = effort
            if tools:
                body["tools"] = tools
            ids = render_prompt_ids(tokenizer, messages, tools=tools, enable_thinking=thinking, reasoning_effort=effort)
            history = render_prompt_ids(tokenizer, messages, tools=tools, enable_thinking=thinking, reasoning_effort=effort, add_generation_prompt=False)
            history_len = len(history) if 0 < len(history) < len(ids) and ids[:len(history)] == history else 0
            app.enable_thinking, app.reasoning_effort = thinking, effort
            system_len = app.system_prefix_len(messages, tools, ids, thinking)
            cases.append({"body": body, "tokens": ids, "history_len": history_len, "system_len": system_len, "gates": gate_cases(ids)})
    for words in (480, 511, 512, 520, 1024, 2560, 4096):
        for thinking in (False, True):
            messages = [{"role": "system", "content": "word " * words}, {"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Explain."}]
            body = {"messages": messages, "tools": [tool], "chat_template_kwargs": {"enable_thinking": thinking}}
            ids = render_prompt_ids(tokenizer, messages, tools=[tool], enable_thinking=thinking)
            app.enable_thinking, app.reasoning_effort = thinking, None
            cases.append({"body": body, "tokens": ids, "system_len": app.system_prefix_len(messages, [tool], ids, thinking)})
    output.parent.mkdir(parents=True, exist_ok=True)
    from tensorfold.engine.prefill_plan import message_markers
    marks, assistant = message_markers(tokenizer)
    cases[0]['markers'] = dict(openers=marks, assistant=assistant, deepseek=deepseek)
    cases[0]['call_form'] = form
    cases[0]['required_supported'] = form is not None and token_id(form[0]) >= 0
    if not cases[0]['required_supported']:
        from tensorfold.server.errors import RequestError
        try:
            options._call_gate({'tool_call_required': True}, cases[0]['tokens'], [tool])
        except RequestError:
            pass
        else:
            raise AssertionError('Expected upstream to reject an unsupported required call')
    if deepseek:
        cases.extend(deepseek_chat_cases(tokenizer, tool))
    for default_thinking in (False, True):
        for default_effort in (None, "medium"):
            for controls in ({}, {"reasoning_effort": None}, *({"reasoning_effort": e} for e in ("none", "minimal", "low", "medium", "high", "xhigh")),
                             {"chat_template_kwargs": {"reasoning_effort": "high"}},
                             {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}},
                             {"chat_template_kwargs": {"reasoning_effort": "none", "enable_thinking": True}},
                             *({"chat_template_kwargs": {"enable_thinking": v}} for v in (False, None, 0, 1, "", "false", [], {}))):
                body = {"messages": [{"role": "user", "content": "Hello"}], **controls}
                fields = thinking_fields(body, options.effort_levels)
                thinking = fields.get("enable_thinking", default_thinking)
                effort = fields.get("reasoning_effort", default_effort)
                ids = render_prompt_ids(tokenizer, body["messages"], enable_thinking=thinking, reasoning_effort=effort)
                cases.append({"body": body, "tokens": ids, "default_thinking": default_thinking, "default_effort": default_effort, "thinking": thinking})
    from tensorfold.server.text import reasoning_count
    cases[0]['reasoning_counts'] = [dict(tokens=tokens, end=end, expected=reasoning_count(tokens, end))
                                  for end in (None, -1, think_end)
                                  for tokens in ([], [1, 2], [think_end], [1, think_end, 2], [1, think_end, think_end])
                                  if all(t >= 0 for t in tokens)]
    output.write_text(json.dumps(cases, ensure_ascii=False))
    print(f"Saved {len(cases)} upstream chat fixtures for {directory.name}")


def deepseek_chat_cases(tokenizer, tool):
    import copy
    import random

    calls = [{"id": f"call_{i}", "type": "function", "function": {"name": "weather", "arguments": json.dumps(args, ensure_ascii=False)}} for i, args in enumerate([
        {"city": "æøå 世界\u0000\n\"", "values": [True, False, None, 1, -7, 1.0, -0.0, 1e-5, 1e-4, 1e15, 1e16], "object": {"a": 2}},
        {"city": "Copenhagen"},
        {},
    ])]
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "Earlier reasoning", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "call_2", "content": "Third"},
        {"role": "user", "content": "Interleaved user"},
        {"role": "tool", "tool_call_id": "call_0", "content": [{"type": "text", "text": "First"}, {"type": "image"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Second"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Latest reasoning"},
    ]
    conversations = [
        [], [{"role": "system", "content": "Only system"}],
        [{"role": "assistant", "content": None}],
        [{"role": "user", "content": "First"}, {"role": "user", "content": "Second"}],
        history, history[:-1], history + [{"role": "user", "content": "Next turn"}],
        [{"role": "developer", "content": "Old instruction"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "reasoning_content": "Discard me"}, {"role": "developer", "content": "Latest instruction"}],
        [{"role": "system", "content": "Schema", "response_format": {"type": "json", "schema": {"type": "object", "enum": ["æøå", 1.0]}}}, {"role": "user", "content": "Hi"}],
        [{"role": "developer", "content": "Schema", "tools": [tool], "response_format": {"type": "object"}}, {"role": "assistant", "content": "Result", "reasoning_content": "Reason"}],
        [{"role": "user", "content": "Hi"}, {"role": "latest_reminder", "content": "Remember"}, {"role": "assistant", "content": "Partial", "wo_eos": True}],
    ]
    for args in ("not JSON", {"nested": [True, 0.00001, "世界"]}, None):
        conversations.append([{"role": "user", "content": "Call"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "weather", "arguments": args}}]}])
    for task in ("action", "query", "authority", "domain", "title", "read_url"):
        messages = [{"role": "user", "content": "Classify", "task": task}]
        conversations.extend([messages, messages + [{"role": "assistant", "content": "Result", "reasoning_content": "No thinking for task"}]])
    cases = []
    def add(messages, tools, kwargs, generation):
        before = copy.deepcopy(messages)
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=generation, **kwargs)
        assert messages == before
        cases.append({"raw": True, "body": {"messages": messages, "tools": tools, "chat_template_kwargs": kwargs, "add_generation_prompt": generation}, "text": text, "tokens": tokenizer.encode(text, add_special_tokens=False)})
    for messages in conversations:
        for thinking, effort in ((False, None), (True, None), (True, "high"), (True, "xhigh"), (True, "max")):
            for tools in (None, [tool], [tool["function"]]):
                for generation in (False, True):
                    add(messages, tools, {"enable_thinking": thinking, "reasoning_effort": effort}, generation)
    rng = random.Random(8172)
    for _ in range(100):
        messages = copy.deepcopy(history)
        rng.shuffle(messages[1]["tool_calls"])
        for call in messages[1]["tool_calls"]:
            if rng.randrange(2):
                call["function"]["id"] = call.pop("id")
        messages[2]["tool_call_id"] = rng.choice(["call_0", "call_1", "call_2", "unknown"])
        messages[4]["tool_call_id"] = messages[2]["tool_call_id"]
        messages = messages[:rng.randrange(1, len(messages) + 1)]
        mode = rng.choice(["chat", "thinking"])
        add(messages, rng.choice([None, [tool]]), {"enable_thinking": mode != "thinking", "thinking_mode": mode, "reasoning_effort": rng.choice([None, "low", "xhigh"])}, bool(rng.randrange(2)))
    return cases


def tool_draft_fixtures(directory, output):
    import random
    from transformers import AutoTokenizer
    from tensorfold.engine.tool_draft import ToolCallProposer
    from tensorfold.engine.lane_engine import SuffixLookupProposer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    rng = random.Random(51713)
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {
        'properties': {'offset': {}, 'path': {}, 'limit': {}, 'body': {}}, 'required': ['path', 'missing', 'path']}}},
        {'name': 'read_dir', 'input_schema': {'properties': {'directory': {}, 'file_id': {}}}},
        {'name': 'empty'}, {'name': 'empty', 'parameters': {}}]
    scripts = [
        '<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a\n</parameter>\n<parameter=offset>\n3\n</parameter>\n</function>\n</tool_call>\n',
        '<tool_call><function=empty></function></tool_call>',
        '<tool_call>\n<function=unknown>\n<parameter=path>\na\n',
        '<tool_call><function=read_dir><parameter=directory>\n/a\n<parameter=file_id>\nb\n',
        '<tool_call><function=read_file>\n<parameter=body>\nmulti\nline\n',
        'Some prose.\n<tool_call>\n<function=read_file>\n<parameter=path>\nfile\n',
    ]
    texts = ['', ' \n\t', '\u2003\u0085', 'plain prose', '<tool_call>\n<function=read_\n',
             '<tool_call>\n<function=read_file>\n<parameter=pa\n',
             '<tool_call></tool_call> </tool_call>', '<tool_call></tool_call> trailing',
             '<tool_call><function=read_file><parameter=fİle>\nx\n',
             '<tool_call><function=read_file><parameter=offset>\n\n',
             '<tool_call><function=read_file><parameter=x<parameter=pa',
             '<tool_call><function=read_file><parameter=>\n<parameter=pa',
             scripts[0] + scripts[1]]
    for script in scripts:
        texts.extend(script[:i] for i in range(len(script) + 1))
    structures = []
    for specs in (None, tools, [{'name': 'read_file', 'input_schema': {'properties': {'file': {}}}}]):
        for opening in (False, True):
            proposer = ToolCallProposer(tokenizer, specs, 0, open_at_start=opening)
            for ending in (False, True):
                proposer.end_text = ending
                structures.extend(dict(tools=specs, text=text, open_at_start=opening,
                                       end_text=ending, expected=proposer.structure(text)) for text in texts)
    streams = []
    prompt = tokenizer.encode('Read a file.\n/tmp/a\nRead a file.\n/tmp/a\n', add_special_tokens=False)
    for fallback in (False, True):
        for opening in (False, True):
            for script in scripts[:3] + ['Read a file.\n/tmp/a\n' * 3]:
                proposer = ToolCallProposer(tokenizer, tools, len(prompt),
                    fallback=SuffixLookupProposer(min_match=4) if fallback else None, open_at_start=opening)
                emitted = tokenizer.encode(script, add_special_tokens=False)
                events = []
                offsets = list(range(len(emitted) + 1)) + [0, len(emitted) // 2, len(emitted)]
                for n in offsets:
                    context = prompt + emitted[:n]
                    budget = rng.choice([0, 1, 3, 15, 31])
                    tree = rng.choice([False, True])
                    if tree:
                        tokens, parents = proposer.propose_tree(context, budget)
                    else:
                        tokens = proposer.propose(context, budget)
                        parents = list(range(-1, len(tokens) - 1))
                    confident, match = proposer.last_confident, proposer.last_match
                    accepted = rng.randrange(len(tokens) + 1)
                    proposer.observe(len(tokens), accepted)
                    events.append(dict(context=context, max_draft=budget, tree=tree, accepted=accepted,
                                       tokens=tokens, parents=parents, confident=confident, match=match, telemetry=proposer.telemetry()))
                streams.append(dict(tools=tools, prompt_len=len(prompt), open_at_start=opening, fallback=fallback, events=events))
    copies = []
    for _ in range(20):
        proposer = SuffixLookupProposer(min_match=4)
        context = list(range(32)) * 3
        events = []
        for i in range(80):
            if i == 40:
                context = context[:36]
            elif i == 60:
                context = [100] * 72
            elif i > 20:
                context.append(rng.randrange(8))
            budget = rng.choice([0, 1, 15, 31])
            tokens = proposer.propose(context, budget)
            confident, match = proposer.last_confident, proposer.last_match
            accepted = 0 if i < 20 else rng.randrange(len(tokens) + 1)
            proposer.observe(len(tokens), accepted)
            events.append(dict(context=context.copy(), max_draft=budget, accepted=accepted, tokens=tokens,
                               confident=confident, match=match, silent_for=proposer._silent_for))
        copies.append(events)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(structures=structures, streams=streams, copies=copies), ensure_ascii=False))
    print(f'Saved {len(structures)} tool structure cases, {len(streams)} token streams and {len(copies)} copy sequences')


def tool_fixtures(output):
    from tensorfold.server.tools import parse_tool_calls_from_content
    from tensorfold.server.tool_policy import ToolCallPolicy
    properties = {"city": {"type": "string"}, "days": {"type": "integer"}, "flag": {"type": "boolean"}, "data": {"type": "object"}, "items": {"type": "array"}, "value": {"type": "number"}, "empty": {"type": "null"}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"properties": properties}}}]
    payloads = [
        '{"name":"weather","arguments":{"city":"Paris","days":2}}',
        '{"function":{"name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"}}',
        '{"tool":"weather","city":"Paris","days":2}',
        '{"name":"missing","arguments":{}}', '{"name":"weather","arguments":[]}',
        '{"name":"weather","arguments":null}', '{"name":"weather","arguments":""}',
        '[{"name":"weather","args":{}},{"name":"weather","args":{"days":2}}]',
        'call:weather{city:<|"|>Paris<|"|>,days:2}',
        'call:weather{data:{city:<|"|>æøå 世界<|"|>},items:[1,2]}',
        '<function=weather><parameter=city>Paris</parameter><parameter=days>2</parameter></function>',
        'weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value>',
        '<FUNCTION=weather><PARAMETER=days>2</PARAMETER></FUNCTION>',
        '<function=x</function>', '<function=weather>garbage</function>', 'weather',
    ]
    for key in properties:
        for value in ('2', 'true', 'null', '1.5', '[]', '{}', '"two"', 'not json', '1e999', '\n x \n',
                      '[1,2', '{"a":[1,2', '{"a":"x\\\"y"', '[1,', '[1}', '{"a":"unfinished'):
            payloads.append(f'<function=weather><parameter={key}>\n{value}\n</parameter></function>')
    texts = ['  prose  ', '{"answer":"plain JSON"}', '```json\n{"name":"weather","arguments":{}}\n```']
    for payload in payloads:
        texts.extend((payload, f'before <tool_call>{payload}</tool_call> after', f'<x:tool_call>{payload}</x:tool_call>', f'<|tool_call>{payload}<tool_call|>'))
    dsml = '<｜DSML｜tool_calls><｜DSML｜invoke name="weather"><｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter><｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
    texts.extend((dsml, dsml + dsml, dsml.replace('>2<', '>bad<')))
    for text in (dsml, '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>', '<tool_call><function=x</function></tool_call>'):
        texts.extend(text[:i] for i in range(len(text) + 1))
    cases = []
    for limit in (None, 1):
        for text in texts:
            content, calls = parse_tool_calls_from_content(text, tools, max_calls=limit)
            cases.append({"text": text, "tools": tools, "max_calls": limit, "content": content, "single_content": ToolCallPolicy({"parallel_tool_calls": False}).content(content), "calls": [call["function"] for call in calls or []]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False))
    print(f"Saved {len(cases)} upstream tool parser fixtures")
    from tensorfold.engine.tool_draft import ToolCallStreamer
    stream_texts = [f'<tool_call>{payload}</tool_call>' for payload in payloads]
    stream_texts.extend((
        '<tool_call><function=weather></function></tool_call>' * 2,
        'before <tool_call>\n<function=WEATHER>\n<parameter=city>\n  æøå 世界 👋 "quoted" \\ path\n\n</parameter>\n</function>\n</tool_call> after',
        '<tool_call><function=weather><parameter=city>Paris</parameter>',
        '<tool_call><function=missing><parameter=city>Paris</parameter></function></tool_call>',
        '<tool_call><function=weather>' + 'x' * 270,
        dsml, 'ordinary prose', '{"name":"weather","arguments":{}}',
    ))
    streams = []
    for text in stream_texts:
        for step in (1, 3, 17, len(text)):
            streamer = ToolCallStreamer(tools)
            frames = []
            for end in list(range(step, len(text), step)) + [len(text)]:
                deltas = streamer.feed(text[:end])
                for delta in deltas:
                    for call in delta["tool_calls"]:
                        call.pop("id", None)
                frames.append({"end": len(text[:end].encode()), "deltas": deltas})
            streams.append({"text": text, "tools": tools, "frames": frames})
    output.with_name("tool-stream.json").write_text(json.dumps(streams, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {len(streams)} upstream incremental tool streams")


def conversion_fixture(directory):
    import struct
    import mlx.core as mx
    from tests import dsv4_fakes as fake
    from tensorfold.families.deepseek_v4.convert import convert_mtp, convert_dspark, fp8_block

    raw = directory / "raw"
    expected = directory / "expected"
    raw.mkdir(parents=True, exist_ok=True)
    expected.mkdir(parents=True, exist_ok=True)

    def write(path, tensors, tagged=True):
        mx.eval(*tensors.values())
        mx.save_safetensors(str(path), tensors)
        if not tagged:
            return
        data = path.read_bytes()
        size, = struct.unpack("<Q", data[:8])
        header = json.loads(data[8:8 + size])
        for name, info in header.items():
            if name == "__metadata__" or info["dtype"] != "U8":
                continue
            info["dtype"] = "F8_E8M0" if name.endswith(".scale") else "I8" if ".experts." in name else "F8_E4M3"
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + size:])

    mtp = fake.write_official_mtp(raw / "mtp.safetensors")
    write(raw / "mtp.safetensors", mtp)
    convert_mtp(raw / "mtp.safetensors", expected / "mtp")
    edge = {key.replace("mtp.0.", "mtp.3."): value for key, value in mtp.items()}
    codes = np.arange(129 * 192, dtype=np.uint32).reshape(129, 192) % 256
    codes = np.where((codes & 127) == 127, codes - 1, codes).astype(np.uint8)
    edge["mtp.3.e_proj.weight"] = mx.array(codes)
    edge["mtp.3.e_proj.scale"] = mx.array([[0, 1], [120, 128]], dtype=mx.uint8)
    edge["model.layers.0.unrelated"] = mx.ones((1,))
    write(raw / "layer3.safetensors", edge)
    convert_mtp(raw / "layer3.safetensors", expected / "layer3", layer=3)
    shards = fake.write_official_dspark(raw)
    for shard in shards:
        write(shard, mx.load(str(shard)))
    (raw / "config.json").write_text(json.dumps({**fake.DSPARK, "unrelated": "excluded"}))
    convert_dspark(shards, expected / "dspark")
    # A block split between files exercises native multi-shard assembly against
    # the same upstream conversion from the original complete blocks.
    from tensorfold.families.deepseek_v4.convert import read_raw
    merged = {}
    for shard in shards:
        merged.update(read_raw(shard, "mtp."))
    for part in range(2):
        write(raw / f"split-{part}.safetensors", {key: merged[key] for key in sorted(merged)[part::2]})
    fp8 = mx.array(np.tile(np.arange(256, dtype=np.uint8), (256 * 128, 1)))
    scales = mx.broadcast_to(mx.arange(256, dtype=mx.uint8)[:, None], (256, 2))
    write(raw / "codes.safetensors", {"mtp.codes.weight": fp8, "mtp.codes.scale": scales})
    mx.save_safetensors(str(expected / "codes.safetensors"), {"decoded": fp8_block(fp8, scales)})
    # Invalid source geometry must fail without replacing an existing output.
    malformed = dict(mtp)
    malformed["mtp.0.e_proj.scale"] = mx.zeros((3, 1), dtype=mx.uint8)
    write(raw / "bad-scale.safetensors", malformed)
    malformed = dict(mtp)
    del malformed["mtp.0.ffn.experts.3.w2.scale"]
    write(raw / "missing-expert.safetensors", malformed)
    fake.write_checkpoint(directory / "target")
    (directory / "target/tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}, "merges": []}, "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}}))
    (directory / "unloadable").mkdir(parents=True, exist_ok=True)
    (directory / "unloadable/config.json").write_text(json.dumps(fake.TEXT))
    from tensorfold.families.deepseek_v4.runtime import drafter_config
    folder_cases = []
    for name, config_text, weights in [
        ("missing-config", None, "model.safetensors"),
        ("malformed", "{", "model.safetensors"),
        ("array-config", "[]", "model.safetensors"),
        ("wrong-type", '{"model_type":"deepseek_v4"}', "model.safetensors"),
        ("missing-type", "{}", "model.safetensors"),
        ("null-type", '{"model_type":null}', "model.safetensors"),
        ("legacy", '{"model_type":"deepseek_v4_mtp"}', "mtp.safetensors"),
        ("no-weights", '{"model_type":"deepseek_v4_mtp"}', None),
        ("directory-weights", '{"model_type":"deepseek_v4_mtp"}', "directory"),
        ("mtp", '{"model_type":"deepseek_v4_mtp"}', "model.safetensors"),
        ("dspark", '{"model_type":"deepseek_v4_dspark","dspark_block_size":8}', "model.safetensors"),
    ]:
        folder = directory / "folders" / name
        folder.mkdir(parents=True, exist_ok=True)
        if config_text is not None:
            (folder / "config.json").write_text(config_text)
        if weights == "directory":
            (folder / "model.safetensors").mkdir(exist_ok=True)
        elif weights:
            (folder / weights).write_bytes(b"")
        try:
            value = drafter_config(folder)
        except (ValueError, AttributeError):
            value = None
        folder_cases.append({"folder": name, "accepted": value is not None, "config": value})
    (directory / "folders.json").write_text(json.dumps(folder_cases))
    for folder, shard in (("no-config", shards[0]), ("alias", raw / "mtp.safetensors")):
        destination = directory / folder / "model.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.hardlink_to(shard)
    print("Saved upstream MTP/DSpark conversion fixtures, split shards and exhaustive FP8/E8M0 codes")


def bonsai_widening_fixture(directory, output):
    import mlx.core as mx
    from tensorfold.families.bonsai import pack
    output.mkdir(parents=True, exist_ok=True)
    model, layers, rest = pack.widening(directory)
    boundaries = [model + pack.ROOM, model + pack.ROOM + sum(layers) + rest]
    for index in range(len(layers) + 1):
        boundaries.append(model + pack.ROOM + sum(layers[:index]))
    budgets = sorted({0, 1, 2**63, *(max(0, b + delta) for b in boundaries for delta in (-1, 0, 1))})
    paths = [record['path'] for record in pack.contract(directory)[0]['modules']]
    paths.extend(['model.layers.0', 'model.layers.01.x', 'model.layers.-1.x', 'model.layers.x.x', 'lm_head', 'other.layers.0.x'])
    forms = ['lanes', 'packed', 'widened', 'widened:0', 'widened:1', 'widened:32', 'widened:64', 'widened:100']
    files = []
    rng = np.random.default_rng(8129)
    for index, (rows, words) in enumerate([(1, 1), (7, 3), (8191, 2), (8192, 3), (8193, 4), (16387, 5)]):
        weight = mx.array(rng.integers(0, 2**32, (rows, words), dtype=np.uint32))
        expected = pack.widen(weight)
        name = f'widen-{index}.safetensors'
        mx.save_safetensors(str(output / name), {'input': weight, 'output': expected})
        files.append(name)
    (output / 'widening.json').write_text(json.dumps(dict(model=model, layers=layers, rest=rest,
        budgets=[dict(budget=b, form=pack.pre_m5_form(directory, b)) for b in budgets],
        modules=[dict(form=f, path=p, result=pack.module_form(f, p)) for f in forms for p in paths],
        invalid=['', 'auto', 'widened:', 'widened:-1', 'widened:1.2', 'widened:+1', 'widened:1:2'], weights=files)))
    print('Saved Bonsai widening policy and bit-exact code conversion fixtures')


def flash_checkpoint_fixture(output):
    import mlx.core as mx
    from tensorfold.families.qwen4_exp.model import sanitize
    from tensorfold.families.qwen4_exp.mtp import sanitize as sanitize_mtp
    from tensorfold.families.qwen4_exp.host_table import from_checkpoint

    output.mkdir(parents=True, exist_ok=True)
    root = "language_model.model.layers.1.ple.ple_embedding."
    tables = []
    ple = {}
    for shard in range(128):
        values = mx.sin(mx.arange(3 * 160, dtype=mx.float32) * .13 + shard).reshape(3, 160).astype(mx.bfloat16)
        tensors = mx.quantize(values, group_size=32, bits=4)
        tables.append(tensors)
        rows = mx.arange(16) % 3
        ple[f"ple-{shard}"] = mx.dequantize(*(t[rows] for t in tensors), group_size=32, bits=4)
    cases = []
    q4_tables = tables
    for bits, style in [(4, s) for s in ("shard_", "shards.", "mixed")] + [(b, "shards.") for b in (2, 3, 5, 6, 8)]:
        tables, ple = [], {}
        for shard in range(128):
            values = mx.sin(mx.arange(3 * 160, dtype=mx.float32) * .13 + shard).reshape(3, 160).astype(mx.bfloat16)
            tensors = mx.quantize(values, group_size=32, bits=bits)
            tables.append(tensors)
            rows = mx.arange(16) % 3
            ple[f"ple-{shard}"] = mx.dequantize(*(t[rows] for t in tensors), group_size=32, bits=bits)
        for prefix in ("language_model.mtp.", "mtp."):
            for indexed in (False, True):
                name = f"bits{bits}-{style}-{prefix}-{int(indexed)}"
                folder = output / name
                folder.mkdir(exist_ok=True)
                for path in folder.glob("model*.safetensors"):
                    path.unlink()
                raw = {"language_model.model.norm.weight": mx.array([1, 2, 3, 4], mx.bfloat16),
                       root + "layer_multipliers": mx.arange(16, dtype=mx.int64) + (1 << 60),
                       root + "ngram_embedding.weight_scale": mx.ones((2,), mx.bfloat16),
                       "visual.weight": mx.array([999]),
                       "language_model.other.mtp.weight": mx.array([888]),
                       prefix + "fc_hidden.weight": mx.array([[3, 4], [5, 6]], mx.bfloat16)}
                for shard, tensors in enumerate(tables):
                    spelling = ("shard_" if shard % 2 else "shards.") if style == "mixed" else style
                    for suffix, value in zip(("weight", "scales", "biases"), tensors):
                        raw[f"{root}ngram_embedding.{spelling}{shard}.{suffix}"] = value
                main, extras = sanitize(raw)
                expected = {key.replace(".ple_embedding.shards.", ".ple_embedding.ngram_embedding.shard_"): value
                            for key, value in (main | extras).items()}
                mx.save_safetensors(str(folder / "expected-0.safetensors"), expected)
                expected.update({"mtp." + key: value for key, value in sanitize_mtp(raw).items()})
                mx.save_safetensors(str(folder / "expected-1.safetensors"), expected)
                weight_map = {}
                for part in (0, 1):
                    filename = f"model-{part + 1:05}-of-00002.safetensors" if indexed else f"model-{'mtp' if part else 'main'}.safetensors"
                    weights = {key: value for key, value in raw.items() if key.startswith(prefix) == bool(part)}
                    mx.save_safetensors(str(folder / filename), weights)
                    weight_map.update({key: filename for key in weights})
                index = folder / "model.safetensors.index.json"
                if indexed:
                    index.write_text(json.dumps({"weight_map": weight_map}))
                else:
                    index.unlink(missing_ok=True)
                table = from_checkpoint(folder, root + "ngram_embedding", 128, ssd=bits == 4)
                try:
                    for shard in range(128):
                        w, s, b = table.gather(shard * 3 + np.arange(16) % 3)
                        gathered = mx.dequantize(mx.array(w), mx.array(s).view(mx.bfloat16), mx.array(b).view(mx.bfloat16), group_size=32, bits=bits)
                        assert mx.array_equal(gathered, ple[f"ple-{shard}"]).item()
                finally:
                    if bits == 4:
                        table.close()
                    else:
                        table._pool.shutdown(wait=True)
                mx.save_safetensors(str(folder / "ple.safetensors"), ple)
                cases.append(dict(name=name, indexed=indexed))
    tables = q4_tables
    for case in ("missing", "split", "duplicate", "format"):
        folder = output / ("ple-" + case)
        folder.mkdir(exist_ok=True)
        raw = {f"{root}ngram_embedding.shards.{shard}.{suffix}": value
               for shard, tensors in enumerate(tables)
               for suffix, value in zip(("weight", "scales", "biases"), tensors)}
        key = root + "ngram_embedding.shards.127.scales"
        second = {}
        if case == "format":
            changed = mx.quantize(mx.ones((3, 160), mx.bfloat16), group_size=32, bits=8)
            for suffix, value in zip(("weight", "scales", "biases"), changed):
                raw[root + "ngram_embedding.shards.127." + suffix] = value
        elif case == "duplicate":
            second[key] = raw[key]
        else:
            value = raw.pop(key)
            if case == "split":
                second[key] = value
        mx.save_safetensors(str(folder / "model-00001-of-00002.safetensors"), raw)
        mx.save_safetensors(str(folder / "model-00002-of-00002.safetensors"), second or {"unused": mx.array([0])})
        try:
            table = from_checkpoint(folder, root + "ngram_embedding", 128, ssd=True)
        except ValueError:
            pass
        else:
            table.close()
            raise AssertionError(f"upstream accepted {case} PLE tensors")
        cases.append(dict(name=folder.name, indexed=False, ple_error=case))
    for name, value in (("zero-scale", 0.), ("nan-scale", float("nan")), ("inf-scale", float("inf"))):
        folder = output / name
        folder.mkdir(exist_ok=True)
        raw = {root + "ngram_embedding.weight_scale": mx.array([1., value], mx.float32)}
        try:
            sanitize(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("upstream accepted a non-identity PLE scale")
        mx.save_safetensors(str(folder / "model.safetensors"), raw)
        cases.append(dict(name=name, indexed=False, scale_error=True))
    (output / "cases.json").write_text(json.dumps(cases))
    print(f"Exported {len(cases)} Flash checkpoint naming/scale fixtures")


def responses_fixture(output):
    from tensorfold.server import responses
    from tensorfold.server.errors import RequestError
    from dataclasses import asdict
    import random

    function = {"type": "function", "name": "weather", "parameters": {"type": "object"}, "strict": True}
    items = [{"role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
             {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"}]},
             {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hm"}]},
             {"role": "assistant", "content": [{"type": "output_text", "text": "Sure."}]},
             {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
             {"type": "function_call", "call_id": "c2", "name": "weather"},
             {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "sunny"}]},
             {"type": "function_call_output", "call_id": "c2", "output": "none"}]
    bodies = [None, [], {}, {"input": ""}, {"input": items}, {"input": [items[2], items[4]]},
              {"input": [{"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}]}]
    controls = [{}, {"instructions": "Be kind", "metadata": {"æ": "🌍"}, "user": "u"},
                {"store": False, "stream": True, "max_output_tokens": 0, "reasoning": {"effort": "none", "summary": "auto"}},
                {"model": "test", "temperature": .7, "top_p": .9, "top_k": 20, "min_p": .2, "seed": 42,
                 "stop": ["END"], "draft": False, "thinking_budget": 3, "ignore_eos": True, "priority": "background",
                 "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": True}},
                {"tools": [function], "tool_choice": {"type": "function", "name": "weather"}},
                {"tools": [function], "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [function]}},
                {"tools": [function], "tool_choice": {"type": "allowed_tools", "tools": []}},
                *({"tool_choice": c, "tools": [function], "parallel_tool_calls": p} for c in (None, "auto", "none", "required") for p in (True, False)),
                *({"text": {"format": f}} for f in (None, {"type": "text"}, {"type": "json_object"},
                   {"type": "json_schema", "name": "n", "schema": {"type": "object"}, "strict": False, "description": "D"})),
                *({"store": v, "metadata": v, "reasoning": v} for v in (None, False, 0, "", [], {})),
                {"instructions": 5}, {"input": []}, {"input": [False]}, {"reasoning": [1]}, {"text": "x"},
                {"tools": {}}, {"tools": [{"type": "web_search"}]}, {"tools": [{"type": "function", "name": ""}]},
                {"metadata": {"k": 1}}, {"metadata": {"k" * 65: "v"}}, {"metadata": {"k": "v" * 513}},
                {"metadata": {str(n): "v" for n in range(17)}}, {"metadata": {"🌍" * 64: "ø" * 512}},
                *({k: True} for k in ("background", "conversation", "prompt", "context_management", "top_logprobs")),
                {"include": ["reasoning.encrypted_content"]},
                {"truncation": "auto"}, {"previous_response_id": "missing"}, {"previous_response_id": 1},
                {"input": [{"type": "reasoning", "encrypted_content": "hidden"}]},
                {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "f"}]}]},
                {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
                {"input": [{"role": "tool", "content": "x"}]}, {"input": [{"type": "item_reference", "id": "x"}]},
                {"input": [{"type": "function_call", "call_id": "c"}]},
                {"input": [{"type": "function_call_output", "call_id": "c", "output": 7}]},
                {"tool_choice": {"type": "allowed_tools", "mode": "none"}},
                {"tool_choice": {"type": "allowed_tools", "tools": [{"type": "web_search"}]}},
                {"text": {"format": {"type": "grammar"}}}]
    bodies.extend({"input": "Hello", **control} for control in controls)
    cases = []
    for body in bodies:
        try:
            cases.append(dict(body=body, expected=asdict(responses.translate(body, responses.Store()))))
        except RequestError:
            cases.append(dict(body=body, error=True))
    rng = random.Random(90210)
    scenarios = []
    for limit, max_bytes in ((4, 1 << 20), (100, 1300), (0, 1 << 20), (10, 0)):
        store = responses.Store(limit=limit, max_bytes=max_bytes)
        actions = []
        for n in range(150):
            action = {"op": ("put", "put", "translate", "get", "delete", "conversation")[n] if n < 6 else rng.choice(("put", "put", "get", "delete", "conversation", "translate"))}
            rid = ("r1" if n in (2, 3, 5) else "r0") if n < 6 else f"r{rng.randrange(max(1, n))}"
            if action['op'] == 'put':
                response = {"id": f"r{n}", "previous_response_id": "r0" if n == 1 else rng.choice([None, *store.entries]),
                            "output": [{"role": "assistant", "content": "æ🌍\n" + str(n)}], "temperature": .7}
                added = [{"role": "user", "content": str(n)}]
                action.update(response=response, added=added)
                store.put(response, added)
                result = None
            else:
                action['id'] = rid
                try:
                    if action['op'] == 'get':
                        result = store.get(rid)
                    elif action['op'] == 'delete':
                        result = store.delete(rid)
                    elif action['op'] == 'conversation':
                        result = store.conversation(rid)
                    else:
                        action['body'] = {"input": "next", "instructions": "new", "previous_response_id": rid}
                        result = asdict(responses.translate(action['body'], store))
                except RequestError:
                    action['error'] = True
                    result = None
            action.update(expected=result, ids=list(store.entries), bytes=store.bytes)
            actions.append(action)
        scenarios.append(dict(limit=limit, max_bytes=max_bytes, actions=actions))
    from tensorfold.server import responses_translate
    from unittest.mock import patch
    from copy import deepcopy
    reply_cases = []
    chat_usage = {"prompt_tokens": 5, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 2},
                  "completion_tokens_details": {"reasoning_tokens": 3}}
    sample_calls = [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Oslo"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]
    for message in ({}, {"content": "Hi æ🌍"}, {"reasoning_content": "hmm"},
                    {"reasoning_content": "hmm", "content": "Hi"}, {"tool_calls": sample_calls},
                    {"reasoning_content": "hmm", "content": "Hi", "tool_calls": sample_calls}):
        for reason in ("stop", "length", "tool_calls"):
            completion = {"choices": [{"message": message, "finish_reason": reason}], "usage": chat_usage,
                          "tensorfold": {"token_sha": "fixed"}}
            reply_cases.append(dict(completion=completion))
            chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
            for key in ("reasoning_content", "content"):
                chunks.extend({"choices": [{"delta": {key: c}}]} for c in message.get(key, ""))
            for i, call in enumerate(message.get('tool_calls', [])):
                chunks.append({"choices": [{"delta": {"tool_calls": [{"index": i, **call, "function": {"name": call['function']['name'], "arguments": ""}}]}}]})
                chunks.extend({"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"arguments": c}}]}}]} for c in call['function']['arguments'])
            chunks.extend([{"choices": [{"delta": {}, "finish_reason": reason}], "usage": chat_usage, "tensorfold": {"token_sha": "fixed"}}, None,
                           {"error": {"message": "ignored after completion"}}])
            reply_cases.append(dict(chunks=chunks))
    reply_cases.extend(dict(chunks=c) for c in (
        [None], [{"choices": [{"delta": {"content": "partial"}}]}, None],
        [{"error": {"message": "boom", "type": "invalid_request_error"}}],
        [{"choices": [{"delta": {"reasoning": "thinking"}}]}, {"error": {"message": "boom"}}],
        [{"choices": [{"delta": {"tool_calls": [{"function": {"name": "f", "arguments": "{"}}]}}]}, {"error": {}}],
        [{"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}]))
    for case in reply_cases:
        counter = 0
        def next_id(prefix):
            nonlocal counter
            value = f"{prefix}_resp_fixture_{counter}"
            counter += 1
            return value
        store = responses.Store()
        events, kept = [], []
        base = {"id": "resp_fixture", "object": "response", "status": "in_progress", "output": [], "usage": None, "error": None, "previous_response_id": None}
        added = [{"role": "user", "content": "Hi"}]
        def emit(event):
            events.append(deepcopy(event))
            kept.append(store.get('resp_fixture') is not None)
        with patch.object(responses_translate, '_id', next_id), patch.object(responses_translate.time, 'time', return_value=1234):
            reply = responses.Reply(base, emit, lambda final: store.put(final, added))
            reply.start()
            if 'completion' in case:
                reply.completion(case['completion'])
            else:
                for chunk in case['chunks']:
                    reply.chunk(chunk)
        case.update(base=base, added=added, events=events, kept=kept, expected=reply.final, stored=store.get('resp_fixture'))
    routes = ["/v1/responses", "/responses/", "/v1/responses/r?x=1", "/responses/r///", "/v1/responses/r/input_items", "/v1/chat/completions", "/responsesx/r"]
    output.parent.mkdir(parents=True, exist_ok=True)
    sizes = [None, True, False, 0, -1, 1.0, -0.0, .0001, .00001, 1e16, 1e15, -1e100, 1e-100, 5e-324,
             "ascii\x00\x7fæ🌍", {"f": 1.0, "s": "\t\r\n"}, ["x", None, False]]
    output.write_text(json.dumps(dict(requests=cases, stores=scenarios, replies=reply_cases, sizes=[dict(value=v, expected=len(json.dumps(v))) for v in sizes], routes=[dict(path=p, expected=responses.route(p)) for p in routes])))
    print(f"Saved {len(cases)} upstream Responses requests, {sum(len(s['actions']) for s in scenarios)} store operations and {len(reply_cases)} reply/event sequences")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chat-fixtures", action="store_true")
    p.add_argument("--tool-fixtures", action="store_true")
    p.add_argument("--responses-fixtures", action="store_true")
    p.add_argument("--tool-draft-fixtures", action="store_true")
    p.add_argument("--tokens", help="Explicit prompt IDs, including for generation")
    p.add_argument("--dump-logits", type=Path)
    p.add_argument("--generate", type=int, default=0)
    p.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--metal-sampling", action="store_true")
    p.add_argument("--simd", action="store_true")
    p.add_argument("--synthetic-glm", action="store_true")
    p.add_argument("--synthetic-deepseek", action="store_true")
    p.add_argument("--synthetic-deepseek-wide", action="store_true")
    p.add_argument("--synthetic-deepseek-packed", action="store_true")
    p.add_argument("--conversion-fixtures", action="store_true")
    p.add_argument("--bonsai-widening", action="store_true")
    p.add_argument("--flash-checkpoint", action="store_true")
    p.add_argument("--synthetic-dspark", action="store_true")
    p.add_argument("--synthetic-dspark-sorted", action="store_true")
    p.add_argument("--synthetic-dspark-wide", action="store_true")
    p.add_argument("--synthetic-dflash", type=int)
    p.add_argument("--gemma-drafter", type=Path)
    p.add_argument("--gemma-prefill", action="store_true")
    p.add_argument("--nemotron-prefill", action="store_true")
    p.add_argument("--flash-prefill", action="store_true")
    p.add_argument("--glm-prefill", action="store_true")
    p.add_argument("--custom-tiles", action="store_true")
    p.add_argument("--synthetic-glm-layout", action="store_true")
    p.add_argument("--synthetic-glm-mixed", action="store_true")
    p.add_argument("--serial-rows", action="store_true")
    p.add_argument("--trace-layers", action="store_true")
    p.add_argument("--state-directory", type=Path)
    args = p.parse_args()
    if args.responses_fixtures:
        responses_fixture(args.output)
        return
    if args.flash_checkpoint:
        flash_checkpoint_fixture(args.output)
        return
    if args.bonsai_widening:
        bonsai_widening_fixture(args.model, args.output)
        return
    if args.tool_fixtures:
        tool_fixtures(args.output)
        return
    if args.tool_draft_fixtures:
        tool_draft_fixtures(args.model, args.output)
        return
    if args.chat_fixtures:
        chat_fixture(args.model, args.output)
        return
    import mlx.core as mx
    import mlx.nn as nn
    if args.conversion_fixtures:
        conversion_fixture(args.model)
        return
    if args.gemma_prefill:
        gemma_prefill_fixture(args.model, args.state_directory)
        return
    if args.flash_prefill:
        flash_prefill_fixture(args.model, args.state_directory, args.simd, args.custom_tiles)
        return
    if args.nemotron_prefill:
        nemotron_prefill_fixture(args.model, args.state_directory, args.simd)
        return
    if args.gemma_drafter:
        gemma_dflash_fixture(args.model, args.gemma_drafter, args.state_directory)
        return
    if args.synthetic_dflash is not None:
        dflash_fixture(args.model, args.state_directory, args.synthetic_dflash)
        return
    if args.synthetic_dspark or args.synthetic_dspark_sorted or args.synthetic_dspark_wide:
        deepseek_dspark_fixture(args.model, args.state_directory, args.synthetic_dspark_sorted, args.synthetic_dspark_wide)
        return
    if args.synthetic_deepseek or args.synthetic_deepseek_wide or args.synthetic_deepseek_packed:
        deepseek_fixture(args.model, args.state_directory, args.synthetic_deepseek_wide or args.synthetic_deepseek_packed, args.synthetic_deepseek_packed)
        return
    if args.synthetic_glm or args.synthetic_glm_mixed:
        from tests.glm5_fakes import write_checkpoint
        formats = {
            "model.language_model.layers.0.self_attn.q_proj": (2, 32),
            "model.language_model.layers.0.self_attn.k_proj": (3, 64),
            "model.language_model.layers.0.self_attn.v_proj": (6, 128),
            "model.language_model.layers.0.self_attn.f_b_proj": (2, 32),
            "model.language_model.layers.0.self_attn.g_b_proj": (3, 32),
            "model.language_model.layers.0.mlp.gate_proj": (2, 128),
            "model.language_model.layers.3.mlp.shared_experts.gate_proj": (6, 64),
            "model.language_model.layers.3.mlp.shared_experts.up_proj": (3, 32),
            "model.language_model.embed_tokens": (3, 32),
            "lm_head": (2, 128),
        } if args.synthetic_glm_mixed else {}
        write_checkpoint(args.model, overrides={key: dict(bits=bits, group_size=group) for key, (bits, group) in formats.items()})
        args.synthetic_glm = True
    if args.synthetic_glm_layout:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_glm5_layouts import write_mlxlm_checkpoint
        if not (args.model / "mlxlm/config.json").exists():
            write_mlxlm_checkpoint(args.model)
        args.model = args.model / "mlxlm"
    if args.synthetic_glm or args.synthetic_glm_layout:
        (args.model / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}, "merges": []}, "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}}))
    if args.glm_prefill:
        glm_prefill_fixture(args.model, args.state_directory, args.custom_tiles)
        return
    from tensorfold.engine.exact_sampling import Sampling, sample_rows
    kind = json.loads((args.model / "config.json").read_text())["model_type"]
    if kind == "glm5_next":
        from tensorfold.families.glm5_next.weights import load_backbone
        model = load_backbone(args.model)
        cache = model.make_cache()
        tokenizer = None
        if args.trace_layers:
            from tensorfold.families.glm5_next.model import Layer, hc_expand
            original_layer = Layer.__call__
            layer_ids = {id(layer): i for i, layer in enumerate(model.layers)}
            positions = [0] * len(model.layers)
            args.state_directory.mkdir(parents=True, exist_ok=True)
            def traced_layer(self, x, *a, **kw):
                if id(self) in layer_ids:
                    i = layer_ids[id(self)]
                    caches, lengths, decode = a
                    def save_stage(label, value):
                        for row in range(value.shape[0]):
                            np.save(args.state_directory / f"trace-{positions[i] + row}-{i}-{label}.npy", np.asarray(value[row:row + 1].astype(mx.float32)))
                    xc, post, comb = self.attn_hc.split(x, decode)
                    normed = mx.fast.rms_norm(xc, self.in_norm, self.eps)
                    save_stage("attn-input", normed)
                    branch = self.attn(normed, caches, lengths, decode)
                    save_stage("attn-output", branch)
                    x = hc_expand(branch, x, post, comb, decode)
                    save_stage("attn-expanded", x)
                    xc, post, comb = self.ffn_hc.split(x, decode)
                    normed = mx.fast.rms_norm(xc, self.post_norm, self.eps)
                    save_stage("ffn-input", normed)
                    branch = self.mlp(normed, decode)
                    save_stage("ffn-output", branch)
                    result = hc_expand(branch, x, post, comb, decode)
                else:
                    result = original_layer(self, x, *a, **kw)
                if id(self) in layer_ids:
                    i = layer_ids[id(self)]
                    for row in range(result.shape[0]):
                        np.save(args.state_directory / f"trace-{positions[i] + row}-{i}.npy", np.asarray(result[row:row + 1].astype(mx.float32)))
                    positions[i] += result.shape[0]
                return result
            Layer.__call__ = traced_layer
        if args.synthetic_glm or args.synthetic_glm_layout:
            from tensorfold.families.glm5_next.mtp import load as load_mtp
            mtp = load_mtp(model)
            mtp_cache = mtp.make_cache()
        def forward(ids):
            hidden = model.hidden(mx.array([ids], dtype=mx.uint32), cache)
            return model.head(hidden[:, -1:] if len(ids) > 16 else hidden)
    elif kind in ("gemma4", "gemma4_text"):
        from tensorfold.families.gemma4.model import load
        model, tokenizer = load(args.model, backend="rows", check=False)
        cache = model.make_cache()
        forward = lambda ids: model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))
    elif kind == "nemotron_h":
        from mlx_lm import load
        from tensorfold.kernels.nemotron.lightning.v1 import kernels
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm
        from tools.native_legacy import nemotron_rows, nemotron as legacy_nemotron
        model, tokenizer = load(str(args.model))
        # Native retains the original combined conv/scan and per-slot experts.
        kernels.mamba_step = legacy_nemotron.mamba_step
        fused = kernels.FusedDecode(model)
        # Build the same explicit operations as native, without mx.compile
        # combining neighboring elementwise operations.
        fused._block = lambda index, kind, nxt: (fused._mamba_block(index, nxt) if kind == "M"
                                                else fused._moe_block(index, nxt))
        if args.simd:
            fused.lane_attention = False
        def experts(index, mixer, x):
            logits = kernels.router_logits(x, mixer.gate.weight)
            ids, weights = kernels.route(logits, fused.gate_bias[index], fused.top_k, fused.scaling)
            return nemotron_rows.experts(mixer.switch_mlp, x, ids), weights, mixer.shared_experts(x)
        fused._moe = experts
        holder = nn.Module()
        holder.model = model
        holder.stacked = [x for x, _ in fused.qkv.values()]
        if args.simd:
            from tensorfold.kernels.nemotron.lightning.v1 import rows
            from types import SimpleNamespace
            rows.install(SimpleNamespace(model=model, fused=fused, args=model.args, batch_rows=128))
        else:
            lane_qmm.install(holder, rows=128, tile=True, wide=True)
        cache = model.make_cache()
        forward = lambda ids: model.lm_head(fused(mx.array([ids], dtype=mx.uint32), cache) if len(ids) <= 16 else model.backbone(mx.array([ids], dtype=mx.uint32), cache)[:, -1:])
    elif kind == "qwen4_exp":
        from tensorfold.families.qwen4_exp.model import load, select_by_kernels
        from tensorfold.families.qwen4_exp.decode import FusedDecode
        from tensorfold.families.qwen4_exp.runtime import FlashNext
        from types import SimpleNamespace
        from tensorfold.kernels.qwen.flash_next.v1 import embed as flash_kernels
        from tensorfold.families.qwen4_exp import decode
        decode.DENSE = "rows"
        # Keep the 32 GB PLE tables sharded. The reference embedding performs the
        # same lookup/dequantization without materializing a second concatenated copy.
        flash_kernels.PleTables = lambda embedding: embedding
        flash_kernels.ple_lookup = lambda ids, tables: tables(ids)
        model, tokenizer = load(args.model, lazy=True)
        model.__dict__["fused"] = FusedDecode(model)
        select_by_kernels(model.layers)
        if args.trace_layers:
            if args.state_directory is None:
                raise ValueError("--trace-layers requires --state-directory")
            trace_directory = args.state_directory / "layers"
            trace_directory.mkdir(parents=True, exist_ok=True)
            fused = model.fused
            connections = {id(entry[key]): (i, key) for i, entry in enumerate(fused.layers)
                           for key in ("attn_hc", "mlp_hc")}
            original_hc = fused._hc
            def traced_hc(h, pending, conn):
                result = original_hc(h, pending, conn)
                i, key = connections.get(id(conn), (len(model.layers) - 1, "head"))
                values = (("input", result[0]), ("mixed", result[1])) if key == "attn_hc" else (
                    (("branch", pending[1][0]), ("moe-input", result[1])) if key == "mlp_hc" else
                    (("head-mixed", result[1]),))
                for label, value in values:
                    np.save(trace_directory / f"{i:02}-{label}.npy", np.asarray(value.astype(mx.float32)))
                return result
            fused._hc = traced_hc
        # The lookup adapter holds the original sharded embedding. Remove its
        # fused alias so calling it cannot recurse into this same adapter.
        for layer in model.layers:
            if "ple" in layer:
                layer.ple.ple_embedding.__dict__.pop("fused_tables", None)
        cache = model.make_cache()
        # The serving runtime uses a row-invariant vocabulary projection; the raw
        # model's __call__ uses MLX's batch-dependent quantized matmul instead.
        runtime = SimpleNamespace(model=model)
        def forward(ids):
            hidden = model.hidden(mx.array([ids], dtype=mx.int32), cache)
            return FlashNext.head(runtime, hidden[:, -1:] if len(ids) > 16 else hidden)
    else:
        raise ValueError(kind)
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else list(range(1, 41)) if args.synthetic_glm or args.synthetic_glm_layout else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    prompt_chunk = 2048 if kind in ("gemma4", "gemma4_text", "nemotron_h", "qwen4_exp") or (kind == "glm5_next" and not (args.synthetic_glm or args.synthetic_glm_layout)) else 16
    for start in range(0, len(tokens), prompt_chunk):
        chunk = tokens[start:start + prompt_chunk]
        if kind in ("gemma4", "gemma4_text"):
            logits = model.head(model.prefill(mx.array([chunk], dtype=mx.uint32), cache)[:, -1:])
        elif kind == "glm5_next" and args.serial_rows:
            hidden_rows = []
            logit_rows = []
            for token in chunk:
                logit_rows.append(forward([token]))
                hidden_rows.append(model.last_normed)
            logits = mx.concatenate(logit_rows, axis=1)
            model.last_normed = mx.concatenate(hidden_rows)
        else:
            logits = forward(chunk)
        mx.eval(logits)
        if kind == "glm5_next" and (args.synthetic_glm or args.synthetic_glm_layout):
            next_tokens = [t + 1 for t in tokens[start:start + 16]]
            if args.serial_rows:
                mtp_hidden = mx.concatenate([mtp(model, model.last_normed[j:j + 1], mx.array([token]), [mtp_cache], (1,), True) for j, token in enumerate(next_tokens)])
            else:
                mtp_hidden = mtp(model, model.last_normed, mx.array(next_tokens), [mtp_cache], (len(next_tokens),), True)
            mtp_logits = mtp.logits(model, mtp_hidden)
            from tensorfold.families.glm5_next.linear import project
            mtp_input = mx.concatenate([mx.fast.rms_norm(model.embed_tokens(mx.array(next_tokens)), mtp.enorm, mtp.eps), mx.fast.rms_norm(model.last_normed, mtp.hnorm, mtp.eps)], axis=-1)
            mtp_projection = project(mtp_input, mtp.eh_proj, rows_exact=True)
            if args.state_directory:
                args.state_directory.mkdir(parents=True, exist_ok=True)
                np.save(args.state_directory / f"mtp-projection-{start // 16}.npy", np.asarray(mtp_projection.astype(mx.float32)))
                np.save(args.state_directory / f"mtp-input-{start // 16}.npy", np.asarray(mtp_input.astype(mx.float32)))
                np.save(args.state_directory / f"hidden-{start // 16}.npy", np.asarray(model.last_normed.astype(mx.float32)))
                np.save(args.state_directory / f"logits-{start // 16}.npy", np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
            mx.eval(mtp_hidden, mtp_logits)
        if start % 512 == 0:
            print(f"Prefill {start + len(chunk)}/{len(tokens)}", flush=True)
    if kind == "qwen4_exp" and args.trace_layers:
        model.fused._hc = original_hc
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_directory:
        args.state_directory.mkdir(parents=True, exist_ok=True)
        if kind == "glm5_next" and (args.synthetic_glm or args.synthetic_glm_layout):
            np.save(args.state_directory / "mtp-hidden.npy", np.asarray(mtp_hidden.astype(mx.float32)))
            np.save(args.state_directory / "mtp-logits.npy", np.asarray(mtp_logits.astype(mx.float32)))
            np.save(args.state_directory / "mtp-projection.npy", np.asarray(mtp_projection.astype(mx.float32)))
            for key in ("keys", "ik", "ig", "pool"):
                array = getattr(mtp_cache, key)
                length = mtp_cache.offset // model.args.index_kpool if key == "pool" else mtp_cache.offset
                np.save(args.state_directory / f"mtp-{key}.npy", np.asarray(array[:length].astype(mx.float32)))
        for i, layer in enumerate(cache):
            for key, source in (("conv", "conv"), ("state", "ssm"), ("keys", "keys"), ("ik", "ik"), ("ig", "ig"), ("pool", "pool")):
                array = getattr(layer, source, None)
                if array is None:
                    continue
                if key in ("keys", "ik", "ig"):
                    array = array[:layer.offset]
                elif key == "pool":
                    array = array[:layer.offset // model.args.index_kpool]
                np.save(args.state_directory / f"layer{i}-{key}.npy", np.asarray(array.astype(mx.float32)))
    if not args.generate:
        np.save(args.output, np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1]))
        print(f"Saved {args.output}: {logits.shape}")
        return
    settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    pos = len(tokens)
    result = []
    eos = model.args.eos_token_id if kind == "glm5_next" else (1, 106, 50) if kind in ("gemma4", "gemma4_text") else (2, 11) if kind == "nemotron_h" else (248044, 248046)
    while len(result) < args.generate:
        if args.metal_sampling:
            from tensorfold.engine.gpu_sampling import sample
            token = int(sample(logits.reshape(-1, logits.shape[-1])[-1:], settings if args.temperature else None, [pos]).item())
        else:
            token = sample_rows(logits.reshape(-1, logits.shape[-1])[-1:], [pos], settings)[0] if args.temperature else int(mx.argmax(logits.reshape(-1, logits.shape[-1])[-1]).item())
        result.append(token)
        if token in eos:
            break
        logits = forward([token])
        mx.eval(logits)
        pos += 1
    digest = hashlib.sha256(np.asarray(result, dtype="<u4").tobytes()).hexdigest()
    args.output.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=result, token_sha256=digest,
                                         peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory())))
    print(f"Saved {args.output}: {len(result)} tokens, {digest}")


if __name__ == "__main__":
    main()
