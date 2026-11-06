#!/usr/bin/env python3
"""Unlimited-OCR-2's reference: the release's own code on the CPU, in torch bf16 and float32 (see --help)."""

USAGE = """Unlimited-OCR-2's reference: the release's own `infer` (modeling_unlimitedocr.py) on the CPU, greedy, in
torch bf16 and float32: the transcripts, stage dumps and block cases the native engine is compared with. It reads the
release's modules only through their inputs and outputs (forward hooks).

Pinned: baidu/Unlimited-OCR-2 at 0480966938291445591f69f08a13b041c073a96a, transformers 4.52.1 (the model card's),
torch 2.15.0 (the card tests 2.13.0 on CUDA; this runs on the CPU), Pillow 13.0.0, PyMuPDF 1.27.2.2.

  pages DIR MODEL          writes the check pages to DIR: two synthetic report pages (A4 at 150 dpi, macOS's Arial)
                           and pages 1, 3 and 10 of the paper the release ships (Unlimited-OCR-2.pdf) at 150 dpi
  capture MODEL OUT CASE=PAGE[,PAGE...] ... [--base]
                           runs `infer` on each case's pages in order (gundam mode, the publisher's server default;
                           --base for the global view alone) and writes OUT/<case>.json: the prompt ids, the
                           generated ids and the text
  dump MODEL OUT PAGE[,PAGE...] [--base]
                           the stage dumps tf-uocr-vision and tf-uocr-decode read: `infer` in bf16 with hooks writes
                           OUT.safetensors (each image encoder call's pixels{i}, sam{i}, clip{i} and projector{i}; the
                           decoder's input rows `embeds`, each layer's prompt rows layer{l}, the first `logits`) and
                           OUT.json (prompt and generated ids, text); the float32 model on the same bf16 pixels and
                           prompt writes OUT-f32.safetensors and OUT-f32.json (its own greedy reply)
  blocks MODEL DUMP OUT --call I --view V
                           the block cases tf-uocr-blocks reads, for one view of a dump: float32 runs SAM and CLIP on
                           its pixels and keeps each SAM block's and CLIP layer's input and output; torch bf16 then
                           runs each alone on its float32 input rounded to bf16. OUT.safetensors holds, per tower
                           (sam, clip), <tower>_in [blocks, rows, width] bf16, <tower>_f32 and <tower>_bf16 outputs;
                           and the steps between them the same way: SAM's neck and convolutions (neck_*), CLIP's
                           embedding and pre-norm (front_*) and the projector (project_cx, project_m3, project_*)

Compare a served reply with the reference's text: `tensorfold-native serve MODEL --vision`, then the same pages in
one chat request."""

import argparse
import json
import sys
import time
from pathlib import Path

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
PAPER_PAGES = (1, 3, 10)


def _report(lines, path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    y = 110
    for kind, text in lines:
        if kind == "h":
            d.text((110, y), text, font=ImageFont.truetype(BOLD, 44), fill="black")
            y += 80
        elif kind == "p":
            for row in text:
                d.text((110, y), row, font=ImageFont.truetype(FONT, 28), fill="black")
                y += 40
            y += 24
        elif kind == "t":
            f = ImageFont.truetype(FONT, 26)
            for row in text:
                for c, cell in enumerate(row):
                    d.text((120 + 330 * c, y + 8), cell, font=f, fill="black")
                d.line([(110, y), (1100, y)], fill="black", width=2)
                y += 48
            d.line([(110, y), (1100, y)], fill="black", width=2)
            y += 40
    im.save(path)


def pages(out: Path, model: Path) -> None:
    import fitz  # PyMuPDF
    out.mkdir(parents=True, exist_ok=True)
    _report([("h", "1. Quarterly Report"),
             ("p", ["The third quarter closed with revenue of 4.2 million dollars,", "an increase of 12 percent over the same period last year.",
                    "Operating costs rose by 3 percent, mostly from new hiring."]),
             ("t", [["Region", "Revenue", "Growth"], ["North", "1.6 M", "+14%"], ["South", "1.1 M", "+9%"], ["West", "1.5 M", "+12%"]]),
             ("p", ["The board approved a dividend of 0.25 per share, payable", "on the fifteenth of December to holders of record."])],
            out / "report-1.png")
    _report([("h", "2. Outlook"),
             ("p", ["For the fourth quarter we expect revenue between 4.3 and", "4.6 million dollars. The new warehouse in Leeds opens",
                    "in January and will shorten delivery times by two days."]),
             ("h", "3. Risks"),
             ("p", ["Currency movements and shipping costs remain the main risks.", "A 10 percent rise in freight rates would reduce the",
                    "operating margin by about one point."])],
            out / "report-2.png")
    doc = fitz.open(model / "Unlimited-OCR-2.pdf")
    for n in PAPER_PAGES:
        doc[n - 1].get_pixmap(dpi=150).save(out / f"paper-{n}.png")
    for p in sorted(out.glob("*.png")):
        print(p)


def _load(model: Path, dtype):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.Tensor.cuda = lambda self, *a, **k: self  # the release's code moves masks with .cuda(); this runs on the CPU
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModel.from_pretrained(model, trust_remote_code=True, use_safetensors=True, torch_dtype=dtype).eval()
    return m, tok


def capture(model: Path, out: Path, cases: list[str], base: bool) -> None:
    import torch
    m, tok = _load(model, torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        name, files = case.split("=", 1)
        files = files.split(",")
        t = time.time()
        res = m.infer(tok, image_file=files if len(files) > 1 else files[0], crop_mode=not base, return_dict=True)
        keep = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()
                if isinstance(v, (str, list, int, float)) or hasattr(v, "tolist")}
        keep.update(pages=files, mode="base" if base else "gundam", seconds=round(time.time() - t, 1))
        (out / f"{name}.json").write_text(json.dumps(keep))
        print(f"{name}: {len(keep.get('token_ids', []))} tokens in {keep['seconds']} s", flush=True)


class _Stop(Exception):
    pass


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _infer_hooked(m, tok, files, base, pixels=None):
    """`infer` with hooks; `pixels` (one per encoder call) replace the views' pixels when given."""
    import torch
    mm = m.model
    cap = {"ids": None, "emb": None, "layers": {}, "logits": None, "calls": []}
    hs = []

    def sam_pre(mod, a):
        i = len(cap["calls"])
        px = (a[0] if pixels is None else pixels[i]).to(next(mod.parameters()).dtype)
        cap["calls"].append({"pixels": px.detach().clone()})
        return (px,) + tuple(a[1:])

    def vision_pre(mod, a):
        px = a[0] if pixels is None else pixels[len(cap["calls"]) - 1]
        return (px.to(next(mod.parameters()).dtype),) + tuple(a[1:])

    def keep(name):
        def f(mod, a, o):
            cap["calls"][-1][name] = _first(o).detach().clone()
        return f

    def ids_hook(mod, a, o):
        if cap["ids"] is None:
            cap["ids"] = a[0][0].tolist()

    def emb_hook(mod, a, kw):
        if cap["emb"] is None:
            cap["emb"] = (a[0] if a else kw["hidden_states"]).detach().clone()

    def layer_hook(l):
        def f(mod, a, o):
            if l not in cap["layers"]:
                cap["layers"][l] = _first(o).detach().clone()
        return f

    def head_hook(mod, a, o):
        if cap["logits"] is None:
            cap["logits"] = o[0, -1].detach().float().clone()

    hs += [mm.sam_model.register_forward_pre_hook(sam_pre), mm.vision_model.register_forward_pre_hook(vision_pre),
           mm.sam_model.register_forward_hook(keep("sam")), mm.vision_model.register_forward_hook(keep("clip")),
           mm.projector.register_forward_hook(keep("projector")), mm.embed_tokens.register_forward_hook(ids_hook),
           mm.layers[0].register_forward_pre_hook(emb_hook, with_kwargs=True),
           m.lm_head.register_forward_hook(head_hook)]
    hs += [layer.register_forward_hook(layer_hook(l)) for l, layer in enumerate(mm.layers)]
    try:
        with torch.no_grad():
            res = m.infer(tok, image_file=files if len(files) > 1 else files[0], crop_mode=not base, return_dict=True)
    finally:
        for h in hs:
            h.remove()
    return cap, res


def _save_dump(cap, res, out: Path, gen):
    import torch
    from safetensors.torch import save_file
    t = {"embeds": cap["emb"].contiguous(), "logits": cap["logits"][None].contiguous()}
    for i, c in enumerate(cap["calls"]):
        for k in ("pixels", "sam", "clip", "projector"):
            t[f"{k}{i}"] = c[k].contiguous()
    for l, v in cap["layers"].items():
        t[f"layer{l}"] = v.contiguous()
    save_file(t, f"{out}.safetensors")
    Path(f"{out}.json").write_text(json.dumps({"prompt": cap["ids"], "gen": gen, "text": res.get("text", "")}))


def dump(model: Path, out: Path, files: list[str], base: bool) -> None:
    import gc
    import torch
    m, tok = _load(model, torch.bfloat16)
    cap, res = _infer_hooked(m, tok, files, base)
    gen = [int(x) for x in res.get("token_ids", [])]
    _save_dump(cap, res, out, gen)
    print(f"{out}: {len(cap['calls'])} encoder calls, prompt {len(cap['ids'])}, reply {len(gen)}", flush=True)
    pixels = [c["pixels"] for c in cap["calls"]]
    del m, cap
    gc.collect()
    m, tok = _load(model, torch.float32)
    cap, res = _infer_hooked(m, tok, files, base, pixels=pixels)
    gen = [int(x) for x in res.get("token_ids", [])]
    _save_dump(cap, res, Path(f"{out}-f32"), gen)
    print(f"{out}-f32: {len(cap['calls'])} encoder calls, reply {len(gen)}", flush=True)


def _cast(x, dtype):
    import torch
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(dtype, copy=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_cast(v, dtype) for v in x)
    if isinstance(x, dict):
        return {k: _cast(v, dtype) for k, v in x.items()}
    return x


def blocks(model: Path, dump_path: Path, out: Path, call: int, view: int) -> None:
    import gc
    import torch
    from safetensors.torch import load_file, save_file
    px = load_file(dump_path)[f"pixels{call}"][view:view + 1]
    m, _ = _load(model, torch.float32)
    mm = m.model
    towers = {"sam": list(mm.sam_model.blocks), "clip": list(mm.vision_model.transformer.layers)}
    seen = {t: [] for t in towers}
    hs = []
    for t, mods in towers.items():
        for b in mods:
            def pre(mod, a, kw, t=t):
                seen[t].append({"args": _cast(a, torch.float32), "kw": kw})
            def post(mod, a, kw, o, t=t):
                seen[t][-1]["out"] = _first(o).detach().clone()
            hs += [b.register_forward_pre_hook(pre, with_kwargs=True), b.register_forward_hook(post, with_kwargs=True)]
    with torch.no_grad():
        p = px.float()
        feat = mm.sam_model(p)
        cl = mm.vision_model(p, feat)
        # the projector takes CLIP's rows (without the class row) beside SAM's
        proj = mm.projector(torch.cat((cl[:, 1:], feat.flatten(2).permute(0, 2, 1)), dim=-1))
    for h in hs:
        h.remove()
    del m
    gc.collect()
    m, _ = _load(model, torch.bfloat16)
    mm = m.model
    towers = {"sam": list(mm.sam_model.blocks), "clip": list(mm.vision_model.transformer.layers)}
    t = {}
    with torch.no_grad():
        for name, mods in towers.items():
            ins, f32s, bfs = [], [], []
            for b, s in zip(mods, seen[name]):
                x = s["args"][0]
                ins.append(x.to(torch.bfloat16).reshape(-1, x.shape[-1]))
                f32s.append(s["out"].reshape(-1, x.shape[-1]))
                y = b(*_cast(s["args"], torch.bfloat16), **_cast(s["kw"], torch.bfloat16))
                bfs.append(_first(y).reshape(-1, x.shape[-1]))
            t[f"{name}_in"] = torch.stack(ins).contiguous()
            t[f"{name}_f32"] = torch.stack(f32s).float().contiguous()
            t[f"{name}_bf16"] = torch.stack(bfs).to(torch.bfloat16).contiguous()
        bf = torch.bfloat16
        rows = lambda x: x.flatten(2).permute(0, 2, 1)[0]  # SAM's [1, 1024, side, side] as rows
        pb = px.to(bf)
        # the neck and its convolutions: SAM in bf16 with block 11's output replaced by float32's, rounded
        last = seen["sam"][-1]["out"]
        h = mm.sam_model.blocks[-1].register_forward_hook(lambda mod, a, o: last.to(bf))
        t["neck_in"] = last.to(bf).reshape(-1, last.shape[-1]).contiguous()
        t["neck_f32"] = rows(feat).float().contiguous()
        t["neck_bf16"] = rows(mm.sam_model(pb)).to(bf).contiguous()
        h.remove()
        # CLIP's embedding and pre-norm: CLIP in bf16 on float32's SAM output, rounded, up to layer 0's input
        got = {}
        def front(mod, a, kw):
            got["x"] = a[0] if a else kw["hidden_states"]
            raise _Stop()
        h = mm.vision_model.transformer.layers[0].register_forward_pre_hook(front, with_kwargs=True)
        try:
            mm.vision_model(pb, feat.to(bf))
        except _Stop:
            pass
        h.remove()
        x0 = seen["clip"][0]["args"][0]
        t["front_in"] = rows(feat).to(bf).contiguous()
        t["front_f32"] = x0.reshape(-1, x0.shape[-1]).float().contiguous()
        t["front_bf16"] = got["x"].reshape(-1, x0.shape[-1]).to(bf).contiguous()
        # the projector on float32's CLIP and SAM rows, rounded
        t["project_cx"] = cl[0].to(bf).contiguous()
        t["project_m3"] = rows(feat).to(bf).contiguous()
        t["project_f32"] = proj[0].float().contiguous()
        fused = torch.cat((cl[:, 1:].to(bf), feat.flatten(2).permute(0, 2, 1).to(bf)), dim=-1)
        t["project_bf16"] = mm.projector(fused)[0].to(bf).contiguous()
    t["pixels"] = px.contiguous()
    save_file(t, f"{out}.safetensors")
    shapes = ", ".join(f"{k} {tuple(v.shape)}" for k, v in t.items())
    print(f"{out}: view {view} of call {call} ({px.shape[-1]}), {shapes}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("pages", help="write the check pages")
    pg.add_argument("dir", type=Path)
    pg.add_argument("model", type=Path)
    cp = sub.add_parser("capture", help="the reference's transcripts of CASE=PAGE[,PAGE...]")
    cp.add_argument("model", type=Path)
    cp.add_argument("out", type=Path)
    cp.add_argument("cases", nargs="+")
    cp.add_argument("--base", action="store_true", help="the global view alone (crop_mode=False)")
    dp = sub.add_parser("dump", help="the stage dumps of PAGE[,PAGE...] (OUT and OUT-f32)")
    dp.add_argument("model", type=Path)
    dp.add_argument("out", type=Path)
    dp.add_argument("pages")
    dp.add_argument("--base", action="store_true", help="the global view alone (crop_mode=False)")
    bp = sub.add_parser("blocks", help="the block cases of one view of a dump")
    bp.add_argument("model", type=Path)
    bp.add_argument("dump", type=Path)
    bp.add_argument("out", type=Path)
    bp.add_argument("--call", type=int, default=0)
    bp.add_argument("--view", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "pages":
        pages(a.dir, a.model.expanduser())
    elif a.cmd == "capture":
        capture(a.model.expanduser(), a.out, a.cases, a.base)
    elif a.cmd == "dump":
        dump(a.model.expanduser(), a.out, a.pages.split(","), a.base)
    else:
        blocks(a.model.expanduser(), a.dump, a.out, a.call, a.view)
    return 0


if __name__ == "__main__":
    sys.exit(main())
