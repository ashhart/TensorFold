"""Write zig/src/server/fixtures/clef_tiny.json: a tiny random Clef joint head and its logits from Cloudflare's own code.

usage: python3 tools/clef_tiny_golden.py JOINT_SCHEMA_MODEL_PY OUT_JSON   (needs torch; JOINT_SCHEMA_MODEL_PY is Cloudflare's
joint_schema_model.py from the Clef release). The weights are rounded to bf16 so the Zig head reads the same values.
"""
import importlib.util
import json
import sys

import torch

spec = importlib.util.spec_from_file_location("jsm", sys.argv[1])
jsm = importlib.util.module_from_spec(spec)
sys.modules["jsm"] = jsm
spec.loader.exec_module(jsm)

torch.manual_seed(1234)
H, W, ROUTING, LAYERS, HEADS, FF, VOCAB, N = 16, 8, 1, 1, 2, 16, 20, 12
head = jsm.JointSchemaHead(H, W, ROUTING, LAYERS, HEADS, FF).eval()
with torch.no_grad():
    for p in head.parameters():
        p.copy_((torch.randn_like(p) * 0.5).bfloat16().float())
    head.prior_logit_scale.fill_(0.5)
    head.joint_logit_scale.fill_(1.0)
    head.residual_gate.fill_(0.25)
    hidden = torch.randn(1, N, H) * 2
    rows = (torch.randn(VOCAB, H)).bfloat16().float()
ids = torch.randint(0, VOCAB, (1, N))
questions = [
    jsm.EncodedQuestion("q0", 1, (0, 3), ((3, 5), (5, 7)), ("a", "b")),
    jsm.EncodedQuestion("q1", 2, (7, 9), ((9, 10), (10, 11), (11, 12)), ("0", "1", "2")),
]
record = jsm.EncodedRecord(tuple(ids[0].tolist()), tuple(questions), "tiny")
with torch.no_grad():
    logits = head(hidden, ids, torch.ones(1, N, dtype=torch.long), [record], rows)[0]


def bits(t):
    return t.contiguous().view(-1).bfloat16().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).tolist()


out = {
    "config": {"hidden": H, "width": W, "routing": ROUTING, "layers": LAYERS, "heads": HEADS, "ff": FF},
    "tensors": {name: bits(p.detach()) for name, p in head.state_dict().items()},
    "lm_head": bits(rows),
    "hidden": hidden.view(-1).tolist(),
    "ids": ids[0].tolist(),
    "questions": [{"kind": q.question_type, "span": list(q.question_span), "options": [list(s) for s in q.option_spans]} for q in questions],
    "logits": [l.tolist() for l in logits],
}
with open(sys.argv[2], "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", sys.argv[2], "logits", out["logits"])
