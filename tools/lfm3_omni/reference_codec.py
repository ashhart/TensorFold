"""The reference codec in one precision on each dump's codes: reference_codec.py MODEL DEV DTYPE DUMP_DIR..."""
import sys
from pathlib import Path

import numpy as np
import torch

import refcommon as rc


def decode(codec, codes):
    """The spectrum head's rows and the waveform for [12, frames] codes, as reference_dump.py takes them."""
    q = codec.quantizer.decode(codes[None]).mT.to(codec.from_quantizer.weight.dtype)
    e = codec.from_quantizer(q)
    dec = codec.decoder
    up = torch.nn.functional.interpolate(e.mT, e.shape[1] * dec.frame_stack, mode="nearest-exact").mT
    lin = dec.lin(dec.lfm(inputs_embeds=up, use_cache=False).last_hidden_state)[0]
    return lin.float().cpu().numpy(), codec.decode(codes[None]).audio_values[0].squeeze(0).float().cpu().numpy()


def main():
    """Writes codec_lin_DTYPE.npy and reply_DTYPE.npy beside each dump's gen_codes.npy."""
    from transformers import AutoProcessor
    dev, name = sys.argv[2], sys.argv[3]
    proc = AutoProcessor.from_pretrained(Path(sys.argv[1]), trust_remote_code=True)
    # bf16 is the codec as stored (its fp32 buffers kept); f32 casts the bf16 weights up
    codec = proc.audio_tokenizer.to(device=dev).eval()
    if name == "f32":
        codec = codec.float()
    for d in map(Path, sys.argv[4:]):
        rc.wait_idle()
        codes = torch.from_numpy(np.load(d / "gen_codes.npy")).long().to(dev)
        if codes.shape[1] == 0:
            continue
        with torch.no_grad():
            lin, wave = decode(codec, codes)
        np.save(d / f"codec_lin_{name}.npy", lin)
        np.save(d / f"reply_{name}.npy", wave)
        print(d.name, lin.shape, wave.shape, flush=True)


if __name__ == "__main__":
    main()
