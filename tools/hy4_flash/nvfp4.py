"""nvfp4.py <bf16 dir> <out dir> [--layers N] [--samples 512]: Hy4-flash to NVFP4 with ModelOpt 0.48.

NVFP4 (W4A4, 16-element blocks, FP8 scales) on every MLP: the 512 routed experts (fused 3-D gate_up_proj /
down_proj, ModelOpt's @use_experts_implementation path), the shared experts and the dense MLPs. BF16 stays
on attention, the compressed indexer, the router, the latent down/up projections, embeddings, lm_head and the
vision tower. Calibration: ModelOpt's own "nemotron-post-training-v3" mix (NVIDIA's chat, science, code,
agentic, math, SWE and multilingual sets). --layers N loads only the first N layers: a smoke test of the
whole path, export included.
"""

import argparse
import time

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--dataset", default="nemotron-post-training-v3")
a = ap.parse_args()

t0 = time.time()
cfg = AutoConfig.from_pretrained(a.src)
text_cfg = getattr(cfg, "text_config", cfg)
if a.layers:
    text_cfg.num_hidden_layers = a.layers
# 80 GiB a card of 95: the model spreads evenly over the 8 cards and each keeps room for calibration activations
mem = {i: "80GiB" for i in range(torch.cuda.device_count())}
model = AutoModelForImageTextToText.from_pretrained(
    a.src, config=cfg, dtype=torch.bfloat16, device_map="auto", max_memory=mem
)
model.eval()
tok = AutoTokenizer.from_pretrained(a.src)
print(f"loaded in {time.time() - t0:.0f} s", flush=True)

# ModelOpt 0.48: quant_cfg is a list of rules, a later rule wins; ours go last
q = {**mtq.NVFP4_DEFAULT_CFG, "quant_cfg": list(mtq.NVFP4_DEFAULT_CFG["quant_cfg"])}
for pat in ("*self_attn*", "*indexer*", "*router*", "*.gate.*", "*routed_expert_down_proj*",
            "*routed_expert_up_proj*", "*engram*", "*lm_head*", "*visual*", "*vision*", "*embed*"):  # fmt: skip
    q["quant_cfg"].append({"quantizer_name": pat, "enable": False})

loader = get_dataset_dataloader(
    dataset_name=a.dataset, tokenizer=tok, batch_size=1, num_samples=a.samples,
    max_sample_length=a.seq, device=model.device,
)  # fmt: skip


def forward(m):
    for i, batch in enumerate(loader):
        with torch.no_grad():
            m(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        if i % 16 == 0:
            print(f"calib {i}/{a.samples} at {time.time() - t0:.0f} s", flush=True)


model = mtq.quantize(model, q, forward)
mtq.print_quant_summary(model)
print(f"calibrated at {time.time() - t0:.0f} s", flush=True)
export_hf_checkpoint(model, export_dir=a.out)
tok.save_pretrained(a.out)
print(f"exported {a.out} at {time.time() - t0:.0f} s", flush=True)
