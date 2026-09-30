"""Mean NLL of the actual next tokens: reference .pt files against the golden's vLLM logprobs."""

import json
import sys
from pathlib import Path

import torch

goldens = {tuple(g["ids"]): g for g in json.loads(Path(sys.argv[1]).read_text())["goldens"]}
for path in sys.argv[2:]:
    r = torch.load(path)
    g = goldens[tuple(r["ids"])]
    lp = torch.log_softmax(r["logits"].float(), -1)
    ids = g["ids"]
    ours = [-float(lp[i - 1, ids[i]]) for i in range(1, len(ids))]
    theirs = [-g["prompt_actual"][i] for i in range(1, len(ids))]
    print(f"{path}: {len(ids)} tokens  NLL ours {sum(ours) / len(ours):.3f}  vLLM {sum(theirs) / len(theirs):.3f}")
