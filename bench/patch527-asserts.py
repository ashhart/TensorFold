#!/usr/bin/env python3
"""ROUND-527: add the 20 missing `static_assert(sizeof(T) == N)` lines to the DeepSeek V4.1 Metal sources.

Each size comes from the Metal compiler's own record-layout dump (tools/zig/dsv41_metal_struct_sizes.py),
not from hand arithmetic.  The assert is inserted on its own line straight after the struct's closing
`};`, in the file that defines the struct.  Idempotent: a struct that already asserts its size is left alone.
"""
import os
import sys

ASSERTS = {
    "ops.metal": [
        ("Dense", 40, "Dense must mirror ops.zig's Dense"),
        ("Activation", 24, "Activation must mirror ops.zig's Activation"),
        ("Copy", 16, "Copy must mirror ops.zig's Copy"),
        ("Combine", 32, "Combine must mirror ops.zig's Combine"),
    ],
    "index_exl3.metal": [
        ("Dsv41Pick", 8, "Dsv41Pick must keep its 8-byte (float score, int position) record"),
    ],
    "engram_exl3.metal": [
        ("EngramHistory", 16, "EngramHistory must mirror engram.zig's History"),
    ],
    "cand.metal": [
        ("TfCandArgs", 12, "TfCandArgs must mirror metal/cand.zig's Args"),
    ],
    "fused.metal": [
        ("TfFPostArgs", 8, "TfFPostArgs must mirror metal/fused.zig's PostArgs"),
        ("TfFQuantArgs", 28, "TfFQuantArgs must mirror metal/glue.zig's Strides"),
    ],
    "glue_tmp.metal": [
        ("TfGCandArgs", 12, "TfGCandArgs must mirror metal/glue.zig's GCandArgs"),
        ("TfGEmbedArgs", 12, "TfGEmbedArgs must mirror metal/glue.zig's GEmbedArgs"),
        ("TfGFp4Args", 24, "TfGFp4Args must mirror metal/glue.zig's GFp4Args"),
        ("TfGFp4RtArgs", 12, "TfGFp4RtArgs must mirror metal/glue.zig's GFp4RtArgs"),
        ("TfGHcArgs2", 8, "TfGHcArgs2 must mirror metal/glue.zig's GHcArgs2"),
        ("TfGMeanArgs", 12, "TfGMeanArgs must mirror metal/glue.zig's GMeanArgs"),
        ("TfGMvArgs", 8, "TfGMvArgs must keep its 8-byte (k, n) pair (glue_tmp's own mv kernel)"),
        ("TfGPoolArgs", 8, "TfGPoolArgs must mirror metal/glue.zig's GPoolArgs"),
        ("TfGRopeArgs", 24, "TfGRopeArgs must mirror metal/glue.zig's GRopeArgs"),
        ("TfGScatterArgs", 12, "TfGScatterArgs must mirror metal/glue.zig's GScatterArgs"),
    ],
    "compress.metal": [
        ("TfCPoolArgs", 8, "TfCPoolArgs must mirror metal/compress.zig's PoolArgs"),
        ("TfCRtArgs", 12, "TfCRtArgs must mirror metal/compress.zig's RtArgs"),
        ("TfCPackArgs", 24, "TfCPackArgs must mirror metal/compress.zig's PackArgs"),
    ],
    "fold.metal": [
        ("TfOPostArgs", 20, "TfOPostArgs must mirror metal/fold.zig's PostArgs"),
    ],
    "topk512.metal": [
        ("TfTopk512Args", 4, "TfTopk512Args must mirror metal/glue.zig's TopkArgs"),
    ],
}


def insert_assert(text, name, size, message):
    """Insert the assert right after `struct name { ... };`, or report that it is already asserted."""
    marker = f"struct {name} {{"
    at = text.find(marker)
    while at != -1:
        # A definition, not a use: after the closing brace of the body there is a `;`.
        depth, i = 1, at + len(marker)
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        end = text.find(";", i)
        if end == -1:
            raise SystemExit(f"struct {name}: no closing `;`")
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        after = text[line_end:]
        if after.lstrip().startswith(f"static_assert(sizeof({name})"):
            return text, False
        line = f'\nstatic_assert(sizeof({name}) == {size}, "{message}");'
        return text[:line_end] + line + after, True
    raise SystemExit(f"struct {name}: definition not found")


def main(directory):
    changed = []
    for source, table in ASSERTS.items():
        path = os.path.join(directory, source)
        text = open(path, encoding="utf-8").read()
        for name, size, message in table:
            text, did = insert_assert(text, name, size, message)
            if did:
                changed.append(f"{source}:{name}")
        open(path, "w", encoding="utf-8").write(text)
    print(f"asserts added: {len(changed)}")
    for line in changed:
        print("  " + line)
    return 0


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(root, "zig", "kernels", "metal", "deepseek_v41")))
