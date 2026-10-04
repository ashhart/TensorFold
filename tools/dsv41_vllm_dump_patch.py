"""vLLM activation dump for DeepSeek-V4.1 parity work (inserted into a PRIVATE COPY of the vLLM recipe).

How it was used (2026-09-30, found the V4.1 q-norm bug):
  1. cp -a the recipe service dir to a sibling (e.g. /home/docker/ai/vllm-serve/tf-dump); set ENFORCE_EAGER=1 in its .env.
  2. insert this file's body right after the `from __future__` line of the copy's overlay/patch_memory_log.py
     (runs on head and worker before `vllm serve`; only rank 0 writes).
  3. SKIP_BUILD=1 SKIP_PULL=1 SKIP_SHIP=1 SKIP_SYNC=1 ./start.sh start   (in the copy's recipe/)
  4. send a prompt of TF_DUMP_TOKENS tokens (default 365); dumps land in the triton cache mount under tfdump/
     (host: ~/.cache/vllm-dsv41-flash-exl3/triton/tfdump), copy them out with `docker cp`.
  5. run the reference with TF_REF_DUMP_DIR set and compare: tools/dsv41_dump_diff.py REFDIR VLLMDIR

Dumps: embed, l0{0,1,2}_{attn,ffn,gate,engram} (module hooks) and layerNN (full stream, pre/post/comb, residual).

The two anchors below are lines of vLLM's vllm/model_executor/models/deepseek_v4_1/nvidia/model.py
(https://github.com/vllm-project/vllm, Apache-2.0, Copyright contributors to the vLLM project). The inserted body is
TensorFold's; this script ships no vLLM or recipe-overlay code.
"""

def _tf_install_dump():
    import pathlib
    import vllm
    target = pathlib.Path(vllm.__file__).resolve().parent / "models" / "deepseek_v4_1" / "nvidia" / "model.py"
    src = target.read_text()
    if "_tf_dump(" in src:
        return
    anchor = "            if idx + 1 in self.aux_hidden_state_layers:\n"
    assert src.count(anchor) == 1, "tf dump: anchor not found once"
    src = src.replace(anchor, "            _tf_dump(idx, hidden_states, residual, post_mix, res_mix, pre_mix)\n" + anchor)
    loop = "        for idx, layer in enumerate(\n            islice(self.layers, self.start_layer, self.end_layer),\n"
    assert src.count(loop) == 1, "tf dump: loop anchor not found once"
    src = src.replace(loop, "        _tf_hooks(self, hidden_states)\n" + loop)
    src += '''

_TF_DUMP_DONE = set()
_TF_HOOKED = [False]
_TF_OUT = "/root/.triton/cache/tfdump"


def _tf_want(n):
    import os
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_rank() != 0:
        return False
    return n == int(os.environ.get("TF_DUMP_TOKENS", "365"))


def _tf_save(name, obj):
    import os
    import torch
    if name in _TF_DUMP_DONE:
        return
    _TF_DUMP_DONE.add(name)
    os.makedirs(_TF_OUT, exist_ok=True)
    torch.save(obj, f"{_TF_OUT}/{name}.pt")


def _tf_hooks(model, hidden_states):
    if _tf_want(hidden_states.shape[0]):
        _tf_save("embed", {"embed": hidden_states.cpu()})
    if _TF_HOOKED[0]:
        return
    _TF_HOOKED[0] = True
    for i in (0, 1, 2):
        layer = model.layers[i]

        def attn_hook(mod, args, out, i=i):
            x = args[1]
            if _tf_want(x.shape[0]):
                _tf_save(f"l{i:02d}_attn", {"x": x.cpu(), "out": out.cpu()})

        def ffn_hook(mod, args, out, i=i):
            x = args[0]
            if _tf_want(x.shape[0]):
                _tf_save(f"l{i:02d}_ffn", {"x": x.cpu(), "out": out.cpu()})

        def gate_hook(mod, args, out, i=i):
            x = args[0]
            o = out[0] if isinstance(out, tuple) else out
            if _tf_want(x.shape[0]):
                _tf_save(f"l{i:02d}_gate", {"x": x.cpu(), "logits": o.float().cpu()})

        layer.attn.register_forward_hook(attn_hook)
        layer.ffn.register_forward_hook(ffn_hook)
        if hasattr(layer.ffn, "gate"):
            layer.ffn.gate.register_forward_hook(gate_hook)
        if getattr(layer, "engram", None) is not None:
            def engram_hook(mod, args, out, i=i):
                x = args[0]
                if _tf_want(x.shape[0]):
                    _tf_save(f"l{i:02d}_engram", {"x": x.cpu(), "out": out.cpu(), "hashes": args[1].cpu()})
            layer.engram.register_forward_hook(engram_hook)


def _tf_dump(idx, hidden_states, residual, post_mix, res_mix, pre_mix):
    import torch
    if not _tf_want(hidden_states.shape[0]):
        return
    stream = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
    _tf_save(f"layer{idx:02d}", {"stream": stream.cpu(), "pre": pre_mix.float().cpu(), "ffn_out": hidden_states.cpu(),
                                 "residual": residual.cpu(), "post": post_mix.float().cpu(), "comb": res_mix.float().cpu()})
'''
    target.write_text(src)
    print("[tf-dump] installed into", target)


_tf_install_dump()
# ---- end tensorfold parity dump ----
