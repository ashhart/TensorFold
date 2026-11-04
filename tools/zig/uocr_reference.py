#!/usr/bin/env python3
"""Unlimited-OCR-2's reference: the release's own `infer` on the CPU in torch bf16, greedy (see --help)."""

USAGE = """Unlimited-OCR-2's reference: the release's own `infer` (modeling_unlimitedocr.py) on the CPU in torch bf16, greedy,
the transcripts the native engine is compared with.

Pinned: baidu/Unlimited-OCR-2 at 0480966938291445591f69f08a13b041c073a96a, transformers 4.52.1 (the model card's),
torch 2.15.0 (the card tests 2.13.0 on CUDA; this runs on the CPU), Pillow 13.0.0, PyMuPDF 1.27.2.2.

  pages DIR MODEL          writes the check pages to DIR: two synthetic report pages (A4 at 150 dpi, macOS's Arial)
                           and pages 1, 3 and 10 of the paper the release ships (Unlimited-OCR-2.pdf) at 150 dpi
  capture MODEL OUT CASE=PAGE[,PAGE...] ... [--base]
                           runs `infer` on each case's pages in order (gundam mode, the publisher's server default;
                           --base for the global view alone) and writes OUT/<case>.json: the prompt ids, the
                           generated ids and the text

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


def capture(model: Path, out: Path, cases: list[str], base: bool) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.Tensor.cuda = lambda self, *a, **k: self  # the release's code moves masks with .cuda(); this runs on the CPU
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModel.from_pretrained(model, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16).eval()
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
    a = p.parse_args(argv)
    if a.cmd == "pages":
        pages(a.dir, a.model.expanduser())
    else:
        capture(a.model.expanduser(), a.out, a.cases, a.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
