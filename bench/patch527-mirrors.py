#!/usr/bin/env python3
"""ROUND-527 part 2: the Zig side of the struct sweep.

The R-526 audit's 20 unchecked kernel arguments were mostly positional `[N]u32` payloads built at the callsite
from a semantic Zig struct (ops.Dense, ops.Combine, glue.Strides ...).  A tuple cannot be checked: reorder the
Zig struct's fields and the payload silently changes meaning while every size stays put -- the R-468 bug class
exactly.  This patch gives each payload a NAMED `extern struct` whose fields are the Metal struct's fields in
the Metal struct's order, builds the payload with NAMED initialisers, and registers the pair in
`mirror_checked.zig`, so `index_mirror.check` compares names, types, array lengths, offsets and total at
comptime.

No arithmetic, no dispatch, no kernel text changes: every payload's bytes are the same as the tuple's.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = os.path.join(ROOT, "zig", "src", "families", "deepseek_v41")


def edit(path, pairs):
    full = os.path.join(FAMILY, path)
    text = open(full, encoding="utf-8").read()
    for old, new, count in pairs:
        have = text.count(old)
        if have != count:
            raise SystemExit(f"{path}: expected {count} of\n---\n{old}\n---\nfound {have}")
        text = text.replace(old, new)
    open(full, "w", encoding="utf-8").write(text)
    print(f"edited {path}")


# ---- ops.zig: the four payloads that had named structs and still went out as tuples --------------------
DENSE_ARGS = """
/// ROUND-527: the kernel's `Dense` record, field for field (it was a positional payload until this round).
pub const DenseArgs = extern struct { rows: u32, k: u32, n: u32, groups: u32, x_stride: u32, w_stride: u32, y_stride: u32, x_dtype: u32, w_dtype: u32, y_dtype: u32 };
"""
ACTIVATION_ARGS = """
/// ROUND-527: the kernel's `Activation` record, field for field (`weighted`/`input_f32` as the kernel's uints).
pub const ActivationArgs = extern struct { rows: u32, width: u32, input_stride: u32, output_stride: u32, weighted: u32, input_f32: u32 };
"""
COMBINE_ARGS = """
/// ROUND-527: the kernel's `Combine` record, field for field (`input_f32` as the kernel's uint).
pub const CombineArgs = extern struct { rows: u32, width: u32, routed_row_stride: u32, slot_stride: u32, shared_stride: u32, y_stride: u32, input_f32: u32, id_stride: u32 };
"""
COPY_ARGS = """
/// ROUND-527: the kernel's `Copy` record, field for field.
pub const CopyArgs = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };
"""
edit("ops.zig", [
    ("    y_dtype: Dtype = .bf16,\n};",
     "    y_dtype: Dtype = .bf16,\n};\n" + DENSE_ARGS, 1),
    ("    input_f32: bool = false,\n};",
     "    input_f32: bool = false,\n};\n" + ACTIVATION_ARGS, 1),
    ("pub const Copy = struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };",
     "pub const Copy = struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };\n" + COPY_ARGS, 1),
    ("    id_stride: u32 = 6,\n};",
     "    id_stride: u32 = 6,\n};\n" + COMBINE_ARGS, 1),
    ("e.setValue([10]u32{ d.rows, d.k, d.n, d.groups, d.x_stride, d.w_stride, d.y_stride, @backingInt(d.x_dtype), @backingInt(d.w_dtype), @backingInt(d.y_dtype) }, 3);",
     "e.setValue(DenseArgs{ .rows = d.rows, .k = d.k, .n = d.n, .groups = d.groups, .x_stride = d.x_stride, .w_stride = d.w_stride, .y_stride = d.y_stride, .x_dtype = @backingInt(d.x_dtype), .w_dtype = @backingInt(d.w_dtype), .y_dtype = @backingInt(d.y_dtype) }, 3);", 2),
    ("e.setValue([6]u32{ d.rows, d.width, d.input_stride, d.output_stride, @intFromBool(d.weighted), @intFromBool(d.input_f32) }, 4);",
     "e.setValue(ActivationArgs{ .rows = d.rows, .width = d.width, .input_stride = d.input_stride, .output_stride = d.output_stride, .weighted = @intFromBool(d.weighted), .input_f32 = @intFromBool(d.input_f32) }, 4);", 1),
    ("e.setValue([4]u32{ d.rows, d.width, d.x_stride, d.y_stride }, 2);",
     "e.setValue(CopyArgs{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride }, 2);", 1),
    ("e.setValue([8]u32{ d.rows, d.width, d.routed_row_stride, d.slot_stride, d.shared_stride, d.y_stride, @intFromBool(d.input_f32), d.id_stride }, 4);",
     "e.setValue(CombineArgs{ .rows = d.rows, .width = d.width, .routed_row_stride = d.routed_row_stride, .slot_stride = d.slot_stride, .shared_stride = d.shared_stride, .y_stride = d.y_stride, .input_f32 = @intFromBool(d.input_f32), .id_stride = d.id_stride }, 4);", 1),
])

# ---- metal/glue.zig: glue_tmp's live argument records ---------------------------------------------------
GLUE_ARGS = """
/// ROUND-527: the argument records of glue_tmp's kernels, one named extern struct each so the mirror check
/// compares them field by field (they were positional `[N]u32` payloads until this round).
pub const GRopeArgs = extern struct { pairs: u32, heads: u32, head_stride: u32, row_stride: u32, off: u32, positions: u32 };
pub const GCandArgs = extern struct { keys: u32, blocks: u32, topk: u32 };
pub const GHcArgs2 = extern struct { dim: u32, rows: u32 };
pub const GScatterArgs = extern struct { words: u32, items: u32, rep: u32 };
pub const GEmbedArgs = extern struct { dim: u32, rows: u32, vocab: u32 };
pub const GMeanArgs = extern struct { dim: u32, rows: u32, stride: u32 };
pub const TopkArgs = extern struct { width: u32 };
"""
edit("metal/glue.zig", [
    ("pub const RouteArgs = extern struct { rows: i32, topk: i32, scale: f32, eps: f32 };",
     "pub const RouteArgs = extern struct { rows: i32, topk: i32, scale: f32, eps: f32 };\n" + GLUE_ARGS, 1),
    ("pub const Strides = struct { k: u32, nb: u32, na: u32 = 1, sa: u32 = 0, sb: u32, ta: u32 = 0, tb: u32 };",
     "pub const Strides = extern struct { k: u32, nb: u32, na: u32, sa: u32, sb: u32, ta: u32, tb: u32 };", 1),
    ("e.setValue([6]u32{ head / 2, heads, head, heads * head, head - 64, positions }, 4);",
     "e.setValue(GRopeArgs{ .pairs = head / 2, .heads = heads, .head_stride = head, .row_stride = heads * head, .off = head - 64, .positions = positions }, 4);", 1),
    ("e.setValue([3]u32{ keys, blocks, topk }, 4);",
     "e.setValue(GCandArgs{ .keys = keys, .blocks = blocks, .topk = topk }, 4);", 1),
    ("e.setValue([2]u32{ dim, rows }, 3);",
     "e.setValue(GHcArgs2{ .dim = dim, .rows = rows }, 3);", 1),
    ("e.setValue([2]u32{ dim, rows }, 5);",
     "e.setValue(GHcArgs2{ .dim = dim, .rows = rows }, 5);", 1),
    ("e.setValue([3]u32{ bytes / 4, items, rep }, 3);",
     "e.setValue(GScatterArgs{ .words = bytes / 4, .items = items, .rep = rep }, 3);", 1),
    ("e.setValue([3]u32{ dim, rows, vocab }, 4);",
     "e.setValue(GEmbedArgs{ .dim = dim, .rows = rows, .vocab = vocab }, 4);", 1),
    ("e.setValue([3]u32{ dim, rows, stride }, 2);",
     "e.setValue(GMeanArgs{ .dim = dim, .rows = rows, .stride = stride }, 2);", 1),
    ("e.setValue([1]u32{width}, 0);",
     "e.setValue(TopkArgs{ .width = width }, 0);", 1),
])

# ---- metal/cand.zig -------------------------------------------------------------------------------------
edit("metal/cand.zig", [
    ("/// Candidate blocks of scores [rows][keys] (vis[row] keys count): mask bytes, 1 = candidate; keyed u32 scratch.",
     "/// ROUND-527: the kernel's `TfCandArgs` record, field for field (a positional payload until this round).\npub const Args = extern struct { keys: u32, blocks: u32, topk: u32 };\n\n/// Candidate blocks of scores [rows][keys] (vis[row] keys count): mask bytes, 1 = candidate; keyed u32 scratch.", 1),
    ("e.setValue([3]u32{ keys, blocks, @min(topk, max_topk) }, 0);",
     "e.setValue(Args{ .keys = keys, .blocks = blocks, .topk = @min(topk, max_topk) }, 0);", 1),
])

# ---- metal/fused.zig ------------------------------------------------------------------------------------
edit("metal/fused.zig", [
    ("pub const SwiArgs = extern struct { n: u32, limit: f32 };",
     "pub const SwiArgs = extern struct { n: u32, limit: f32 };\n/// ROUND-527: the kernel's `TfFPostArgs` record, field for field.\npub const PostArgs = extern struct { dim: u32, rows: u32 };\n/// ROUND-527: the kernel's `TfFQuantArgs` record, field for field (glue.Strides' fields in the kernel's order).\npub const QuantArgs = extern struct { k: u32, nb: u32, na: u32, sa: u32, sb: u32, ta: u32, tb: u32 };", 1),
    ("e.setValue([2]u32{ dim, rows }, 7);",
     "e.setValue(PostArgs{ .dim = dim, .rows = rows }, 7);", 1),
    ("e.setValue([7]u32{ s.k, s.nb, s.na, s.sa, s.sb, s.ta, s.tb }, 2);",
     "e.setValue(QuantArgs{ .k = s.k, .nb = s.nb, .na = s.na, .sa = s.sa, .sb = s.sb, .ta = s.ta, .tb = s.tb }, 2);", 1),
])

# ---- metal/compress.zig ----------------------------------------------------------------------------------
edit("metal/compress.zig", [
    ("pub const threads = 32;",
     """pub const threads = 32;

/// ROUND-527: compress.metal's three argument records, field for field (positional payloads until this round).
pub const PoolArgs = extern struct { rows: u32, dim: u32 };
pub const RtArgs = extern struct { blocks: u32, in_stride: u32, out_row: u32 };
pub const PackArgs = extern struct { blocks: u32, in_stride: u32, code_row: u32, scale_row: u32, scale_off: u32, use_dest: u32 };""", 1),
    ("e.setValue([2]u32{ rows, dim }, 5);",
     "e.setValue(PoolArgs{ .rows = rows, .dim = dim }, 5);", 1),
    ("e.setValue([3]u32{ dim / 16, dim, dim }, 3);",
     "e.setValue(RtArgs{ .blocks = dim / 16, .in_stride = dim, .out_row = dim }, 3);", 1),
    ("e.setValue([6]u32{ k.blocks, k.in_stride, k.code_row, k.scale_row, k.scale_off, @intFromBool(k.dest != null) }, 4);",
     "e.setValue(PackArgs{ .blocks = k.blocks, .in_stride = k.in_stride, .code_row = k.code_row, .scale_row = k.scale_row, .scale_off = k.scale_off, .use_dest = @intFromBool(k.dest != null) }, 4);", 1),
])

# ---- metal/fold.zig ---------------------------------------------------------------------------------------
edit("metal/fold.zig", [
    ("pub const Routed = struct { out: Ref, where: Ref, weights: Ref, ids: Ref, sum: Ref, topk: u32, per_group: u32 };",
     "pub const Routed = struct { out: Ref, where: Ref, weights: Ref, ids: Ref, sum: Ref, topk: u32, per_group: u32 };\n/// ROUND-527: the kernel's `TfOPostArgs` record, field for field.\npub const PostArgs = extern struct { dim: u32, rows: u32, topk: u32, per_group: u32, groups: u32 };", 1),
    ("e.setValue([5]u32{ dim, rows, r.topk, r.per_group, 8 }, 10);",
     "e.setValue(PostArgs{ .dim = dim, .rows = rows, .topk = r.topk, .per_group = r.per_group, .groups = 8 }, 10);", 1),
])

# ---- the one metal message that names the Zig type the wire record lives in ------------------------------
edit("../../../kernels/metal/deepseek_v41/fused.metal", [
    ('"TfFQuantArgs must mirror metal/glue.zig\'s Strides"', '"TfFQuantArgs must mirror metal/fused.zig\'s QuantArgs"', 1),
])

# ---- engram.zig: the 16-byte history record --------------------------------------------------------------
edit("engram.zig", [
    ("pub const history_bytes = 16;",
     """pub const history_bytes = 16;

/// ROUND-527: engram_exl3.metal's `EngramHistory`, field for field: three token slots and the count.
pub const History = extern struct { tail: [3]i32, available: u32 };
comptime {
    if (@sizeOf(History) != history_bytes) @compileError("engram.History must stay 16 bytes");
}""", 1),
])

# ---- the registry ----------------------------------------------------------------------------------------
edit("mirror_checked.zig", [
    ('const attn = @import("metal/csa2_attn.zig");',
     'const attn = @import("metal/csa2_attn.zig");\nconst cand = @import("metal/cand.zig");\nconst compress = @import("metal/compress.zig");', 1),
    ('        mirror.check(fused.SwiArgs, sources.dsv41.fused, "TfFSwiArgs");',
     '''        mirror.check(fused.SwiArgs, sources.dsv41.fused, "TfFSwiArgs");

        // ROUND-527: the payloads that used to be positional literals -- one named extern struct a twin, so
        // a reorder or a same-size retype on either side is a build error, not a silent argument shift.
        mirror.check(ops.DenseArgs, sources.deepseek_v41_ops, "Dense");
        mirror.check(ops.ActivationArgs, sources.deepseek_v41_ops, "Activation");
        mirror.check(ops.CopyArgs, sources.deepseek_v41_ops, "Copy");
        mirror.check(ops.CombineArgs, sources.deepseek_v41_ops, "Combine");
        mirror.check(glue.GRopeArgs, sources.dsv41.glue_tmp, "TfGRopeArgs");
        mirror.check(glue.GCandArgs, sources.dsv41.glue_tmp, "TfGCandArgs");
        mirror.check(glue.GHcArgs2, sources.dsv41.glue_tmp, "TfGHcArgs2");
        mirror.check(glue.GScatterArgs, sources.dsv41.glue_tmp, "TfGScatterArgs");
        mirror.check(glue.GEmbedArgs, sources.dsv41.glue_tmp, "TfGEmbedArgs");
        mirror.check(glue.GMeanArgs, sources.dsv41.glue_tmp, "TfGMeanArgs");
        mirror.check(glue.TopkArgs, sources.dsv41.topk512, "TfTopk512Args");
        mirror.check(cand.Args, sources.dsv41.cand, "TfCandArgs");
        mirror.check(fused.PostArgs, sources.dsv41.fused, "TfFPostArgs");
        mirror.check(fused.QuantArgs, sources.dsv41.fused, "TfFQuantArgs");
        mirror.check(compress.PoolArgs, sources.dsv41.compress, "TfCPoolArgs");
        mirror.check(compress.RtArgs, sources.dsv41.compress, "TfCRtArgs");
        mirror.check(compress.PackArgs, sources.dsv41.compress, "TfCPackArgs");
        mirror.check(fold.PostArgs, sources.dsv41.fold, "TfOPostArgs");
        mirror.check(engram.History, sources.deepseek_v41_engram, "EngramHistory");''', 1),
])
print("round 527 part 2 applied")
