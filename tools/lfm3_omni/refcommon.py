"""The reference runtime for LFM3-Omni-2B's checks: transformers with the checkpoint's own code, on cuda, mps or cpu."""
import os
import subprocess
import time
from pathlib import Path

import torch


def dtype_of(name):
    """bf16 (the release) or f32 (the reference's own rounding noise)."""
    return {"bf16": torch.bfloat16, "f32": torch.float32}[name]


def sync(dev):
    """Waits for the device so host timers cover its work."""
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def load(model_dir, dev, dtype):
    """The model and processor, loaded on the CPU then moved (an MPS fp32 load straight to the device can fail)."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model = AutoModelForMultimodalLM.from_pretrained(Path(model_dir), dtype=dtype, device_map="cpu",
                                                     trust_remote_code=True)
    model = model.eval().to(dev)
    proc = AutoProcessor.from_pretrained(Path(model_dir), trust_remote_code=True)
    return model, proc


def chat(proc, system, wav, dev):
    """The processor's inputs for a system text and one spoken turn."""
    conv = [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "audio", "audio": str(wav)}]}]
    out = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt")
    return out.to(dev)


def wait_idle():
    """Waits while $LFM3_OMNI_BUSY (a command) exits 0, so a shared machine's interactive work goes first."""
    cmd = os.environ.get("LFM3_OMNI_BUSY")
    while cmd and subprocess.run(cmd, shell=True).returncode == 0:
        time.sleep(5)


def identity_order():
    """Greedy codes in identity codebook order: the reveal order is the only randomness left under greedy audio."""
    torch.randperm = lambda n, *a, device=None, **k: torch.arange(n, device=device)
