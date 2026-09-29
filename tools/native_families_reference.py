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
    write_mtp(directory)
    (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}, "merges": []}, "pre_tokenizer": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"}}))
    output.mkdir(parents=True, exist_ok=True)
    model = load_backbone(directory)
    cache = model.make_cache()
    from tensorfold.families.deepseek_v4.mtp import load as load_mtp, MTPCache
    mtp = load_mtp(model, directory / "mtp.safetensors")
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
    drafter = load_dspark(model, directory / "drafter/dspark.safetensors", fake.DSPARK)
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
    from tensorfold.server.request_options import RequestOptions
    from threading import Lock
    tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(str(directory), local_files_only=True))
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
    for thinking, effort in ((False, None), (True, "low"), (True, "medium"), (True, "xhigh")):
        for index, messages in enumerate(conversations):
            tools = [tool] if index >= 4 else None
            body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}}
            if effort:
                body["reasoning_effort"] = effort
            if tools:
                body["tools"] = tools
            ids = render_prompt_ids(tokenizer, messages, tools=tools, enable_thinking=thinking, reasoning_effort=effort)
            cases.append({"body": body, "tokens": ids, "gates": gate_cases(ids)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False))
    print(f"Saved {len(cases)} upstream chat fixtures for {directory.name}")


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
        for value in ('2', 'true', 'null', '1.5', '[]', '{}', '"two"', 'not json', '1e999', '\n x \n'):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chat-fixtures", action="store_true")
    p.add_argument("--tool-fixtures", action="store_true")
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
    p.add_argument("--synthetic-dspark", action="store_true")
    p.add_argument("--synthetic-dspark-sorted", action="store_true")
    p.add_argument("--synthetic-dspark-wide", action="store_true")
    p.add_argument("--synthetic-dflash", type=int)
    p.add_argument("--gemma-drafter", type=Path)
    p.add_argument("--gemma-prefill", action="store_true")
    p.add_argument("--synthetic-glm-layout", action="store_true")
    p.add_argument("--synthetic-glm-mixed", action="store_true")
    p.add_argument("--serial-rows", action="store_true")
    p.add_argument("--trace-layers", action="store_true")
    p.add_argument("--state-directory", type=Path)
    args = p.parse_args()
    if args.tool_fixtures:
        tool_fixtures(args.output)
        return
    if args.chat_fixtures:
        chat_fixture(args.model, args.output)
        return
    import mlx.core as mx
    import mlx.nn as nn
    if args.gemma_prefill:
        gemma_prefill_fixture(args.model, args.state_directory)
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
        forward = lambda ids: model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))
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
            for _, module in holder.named_modules():
                if isinstance(module, nn.QuantizedLinear):
                    module.__class__ = nemotron_rows.RowLinear
        else:
            lane_qmm.install(holder, rows=128, tile=True, wide=True)
        cache = model.make_cache()
        forward = lambda ids: model.lm_head(fused(mx.array([ids], dtype=mx.uint32), cache))
    elif kind == "qwen4_exp":
        from tensorfold.families.qwen4_exp.model import load
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
        # The lookup adapter holds the original sharded embedding. Remove its
        # fused alias so calling it cannot recurse into this same adapter.
        for layer in model.layers:
            if "ple" in layer:
                layer.ple.ple_embedding.__dict__.pop("fused_tables", None)
        cache = model.make_cache()
        # The serving runtime uses a row-invariant vocabulary projection; the raw
        # model's __call__ uses MLX's batch-dependent quantized matmul instead.
        runtime = SimpleNamespace(model=model)
        forward = lambda ids: FlashNext.head(runtime, model.hidden(mx.array([ids], dtype=mx.int32), cache))
    else:
        raise ValueError(kind)
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else list(range(1, 41)) if args.synthetic_glm or args.synthetic_glm_layout else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    prompt_chunk = 2048 if kind in ("gemma4", "gemma4_text") else 16
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
