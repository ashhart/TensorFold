"""Qwen3-ASR's per-stage tensors from qwen-asr 0.0.6's transformers backend: tf-qwen3-asr's reference case."""
# usage: python3 -I tools/zig/qwen3_asr_case.py MODEL_DIR CLIP.wav OUT_DIR [--tokens N]; float32/int32 files a stage

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch


def save(out, manifest, name, t, kind="f32"):
    if manifest.get("ids_only") and name not in ("prompt_ids", "greedy_ids", "greedy_top2", "greedy_margin"):
        return
    a = t.detach().to(torch.float32 if kind == "f32" else torch.int32).cpu().numpy()
    a.astype("<f4" if kind == "f32" else "<i4").tofile(os.path.join(out, name + ".bin"))
    manifest["stages"][name] = {"shape": list(a.shape), "dtype": kind}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("clip")
    p.add_argument("out")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--ids-only", action="store_true", help="only the prompt, greedy ids and margins: no stage tensors")
    p.add_argument("--global-attention", action="store_true", help="the transformers backend as is: no windows")
    a = p.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    from qwen_asr import Qwen3ASRModel

    os.makedirs(a.out, exist_ok=True)
    asr = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu", max_new_tokens=a.tokens)
    # bf16 only while loading: under it the feature extractor's hann window and STFT run in bf16 too
    torch.set_default_dtype(torch.float32)
    model, proc = asr.model, asr.processor
    model.eval()
    wav, sr = sf.read(a.clip, dtype="float32")
    if sr != 16000 or wav.ndim != 1:
        sys.exit("want a 16 kHz mono clip")
    manifest = {"clip": os.path.basename(a.clip), "samples": int(wav.shape[0]), "stages": {}, "ids_only": a.ids_only}
    prompt = asr._build_text_prompt(context="", force_language=None)
    inputs = proc(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    feats = inputs["input_features"][0]
    flen = int(inputs["feature_attention_mask"][0].sum())
    manifest["frames"] = flen
    save(a.out, manifest, "mel", feats[:, :flen])
    tower = model.thinker.audio_tower
    taps = {}

    def tap(name):
        def hook(_m, args, out):
            taps.setdefault(name, []).append(out[0] if isinstance(out, tuple) else out)
        return hook

    def windows(_m, args, kwargs):
        # the vLLM backend's (and flash-attention-2's) windows: sdpa here gets no mask and attends over every row
        hidden, cu = args[0], args[1]
        kwargs["attention_mask"] = tower._prepare_attention_mask(hidden, cu)
        return args, kwargs

    def tap_in(name):
        def hook(_m, args):
            taps.setdefault(name, []).append(args[0])
        return hook

    hooks = [tower.conv2d1.register_forward_hook(tap("conv1")), tower.conv2d2.register_forward_hook(tap("conv2")),
             tower.conv2d3.register_forward_hook(tap("conv3")), tower.conv_out.register_forward_hook(tap("conv_out")),
             tower.layers[0].register_forward_pre_hook(tap_in("embed")), tower.ln_post.register_forward_hook(tap("ln_post")),
             tower.proj1.register_forward_hook(tap("proj1")), tower.proj2.register_forward_hook(tap("audio_rows"))]
    for i, layer in enumerate(tower.layers):
        hooks.append(layer.register_forward_hook(tap(f"layer{i}")))
        if not a.global_attention:
            hooks.append(layer.register_forward_pre_hook(windows, with_kwargs=True))
    lm = model.thinker.model
    manifest["attention"] = "global" if a.global_attention else "windows"
    hooks.append(lm.layers[0].register_forward_pre_hook(tap_in("lm_embed")))
    hooks.append(lm.norm.register_forward_hook(tap("lm_norm")))
    l0 = lm.layers[0]  # layer 0's steps: a wrong LM stage is found inside its first layer
    hooks += [l0.input_layernorm.register_forward_hook(tap("lm0_norm")), l0.self_attn.o_proj.register_forward_pre_hook(tap_in("lm0_attn")),
              l0.self_attn.o_proj.register_forward_hook(tap("lm0_o")), l0.post_attention_layernorm.register_forward_hook(tap("lm0_post")),
              l0.mlp.register_forward_hook(tap("lm0_mlp"))]
    for i in (0, len(lm.layers) // 2, len(lm.layers) - 1):
        hooks.append(lm.layers[i].register_forward_hook(tap(f"lm_layer{i}")))
    inputs = inputs.to(model.dtype)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=a.tokens, output_scores=True)
    for h in hooks:
        h.remove()
    for name in ("conv1", "conv2", "conv3", "conv_out"):
        save(a.out, manifest, name, torch.cat(taps[name], 0))
    for name in ["embed"] + [f"layer{i}" for i in range(len(tower.layers))] + ["ln_post", "proj1", "audio_rows"]:
        save(a.out, manifest, name, taps[name][0])
    ids = inputs["input_ids"][0]
    save(a.out, manifest, "prompt_ids", ids, "i32")
    for name in [k for k in taps if k.startswith("lm")]:
        save(a.out, manifest, name, taps[name][0][0])
    new = gen.sequences[0, ids.shape[0]:]
    save(a.out, manifest, "greedy_ids", new, "i32")
    # each step's two best logits: a differing transcript is listed with the reference's margin where it differs
    scores = torch.stack([s[0].float() for s in gen.scores], 0)
    top = scores.topk(2, dim=-1)
    save(a.out, manifest, "greedy_top2", top.indices, "i32")
    save(a.out, manifest, "greedy_margin", top.values[:, 0] - top.values[:, 1])
    manifest["text"] = proc.batch_decode([new], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    manifest["torch"] = torch.__version__
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"frames": flen, "tokens": int(new.shape[0]), "text": manifest["text"]}))


if __name__ == "__main__":
    main()
