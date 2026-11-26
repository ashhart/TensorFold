//! Hy4-flash row, argmax, TP sharding, BF16 KV write, RoPE, and ViT CUDA launchers (`hy4_rows.cu`).
const std = @import("std");
const cuda = @import("cuda");

pub const route = @import("route_cuda.zig");
pub const dfly2 = @import("dfly2_glue_cuda.zig");

const launch = cuda.launch;
const Stream = cuda.Stream;

pub const threads = 256;
pub const warps = threads / 32;
pub const Tile = enum { wide, narrow };
pub const tile_rows = [_]usize{ 16, 4 };
pub const tile_cols = [_]usize{ 4, 2 };

pub const Rows = struct {
    mod: cuda.Module,
    wide_fn: cuda.Function,
    narrow_fn: cuda.Function,
    argmax_fn: cuda.Function,
    slice_bf16_fn: cuda.Function,
    interleave2_bf16_fn: cuda.Function,
    slice_f32_fn: cuda.Function,
    comb_swiglu_bf16_fn: cuda.Function,
    kv_write_fn: cuda.Function,
    q_rope_fn: cuda.Function,
    f32_to_bf16_fn: cuda.Function,
    interleave_bf16_fn: cuda.Function,
    kv_write_ex_fn: cuda.Function,
    kv_write_paged_fn: cuda.Function,
    q_rope_mrope_fn: cuda.Function,
    splice_streams_fn: cuda.Function,
    vit_qkv_fn: cuda.Function,
    vit_merge_gather_fn: cuda.Function,
    vit_append_newline_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Rows {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_rows);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .wide_fn = try mod.function("hy4_rows_wide"),
            .narrow_fn = try mod.function("hy4_rows_narrow"),
            .argmax_fn = try mod.function("hy4_rows_argmax"),
            .slice_bf16_fn = try mod.function("hy4_rows_slice_bf16"),
            .interleave2_bf16_fn = try mod.function("hy4_rows_interleave2_bf16"),
            .slice_f32_fn = try mod.function("hy4_rows_slice_f32"),
            .comb_swiglu_bf16_fn = try mod.function("hy4_rows_comb_swiglu_bf16"),
            .kv_write_fn = try mod.function("hy4_rows_kv_write"),
            .q_rope_fn = try mod.function("hy4_rows_q_rope"),
            .f32_to_bf16_fn = try mod.function("hy4_rows_f32_to_bf16"),
            .interleave_bf16_fn = try mod.function("hy4_rows_interleave_bf16"),
            .kv_write_ex_fn = try mod.function("hy4_rows_kv_write_ex"),
            .kv_write_paged_fn = try mod.function("hy4_rows_kv_write_paged"),
            .q_rope_mrope_fn = try mod.function("hy4_rows_q_rope_mrope"),
            .splice_streams_fn = try mod.function("hy4_rows_splice_streams"),
            .vit_qkv_fn = try mod.function("hy4_rows_vit_qkv"),
            .vit_merge_gather_fn = try mod.function("hy4_rows_vit_merge_gather"),
            .vit_append_newline_fn = try mod.function("hy4_rows_vit_append_newline"),
        };
    }

    pub fn unload(r: *Rows) void {
        r.mod.unload();
        r.* = undefined;
    }

    pub fn sliceBf16(r: *const Rows, s: Stream, src: u64, src_stride: usize, dst: u64, dst_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        if (src_stride % 8 != 0 or dst_cols % 8 != 0) return error.BadStepShape;
        const total = rows_n * (dst_cols / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        a.add(src);
        a.add(@as(i32, @intCast(src_stride / 8)));
        a.add(dst);
        a.add(@as(i32, @intCast(dst_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.slice_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn interleave2Bf16(r: *const Rows, s: Stream, src: u64, dst: u64, half_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        if (half_cols % 8 != 0) return error.BadStepShape;
        const total = rows_n * 2 * (half_cols / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(half_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.interleave2_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn f32ToBf16(r: *const Rows, s: Stream, src: u64, dst: u64, total: usize, nonfinite: u64) !void {
        if (total == 0) return;
        const n4 = (total + 3) / 4;
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(total)));
        a.add(nonfinite);
        try launch.launch(r.f32_to_bf16_fn, .{ .grid = .{ .x = @intCast((n4 + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn interleaveBf16(r: *const Rows, s: Stream, src: u64, dst: u64, local_cols: usize, rows_n: usize, world: usize, nonfinite: u64) !void {
        if (rows_n == 0 or world == 0) return;
        if (local_cols % 8 != 0) return error.BadStepShape;
        const total = rows_n * world * (local_cols / 8);
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(local_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        a.add(@as(i32, @intCast(world)));
        a.add(nonfinite);
        try launch.launch(r.interleave_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn sliceF32(r: *const Rows, s: Stream, src: u64, src_stride: usize, col_off: usize, dst: u64, dst_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        if (src_stride % 4 != 0 or col_off % 4 != 0 or dst_cols % 4 != 0) return error.BadStepShape;
        const total = rows_n * (dst_cols / 4);
        if (total == 0) return;
        var a: launch.Args = .{};
        a.add(src);
        a.add(@as(i32, @intCast(src_stride / 4)));
        a.add(@as(i32, @intCast(col_off / 4)));
        a.add(dst);
        a.add(@as(i32, @intCast(dst_cols / 4)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.slice_f32_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn combSwigluBf16(r: *const Rows, s: Stream, comb_f32: u64, comb_stride: usize, my_col: usize, peer_f32: u64, gu: u64, gu_stride: usize, gate_off: usize, up_off: usize, out: u64, half: usize, sw: usize, rows_n: usize, rank: u32) !void {
        if (rows_n == 0) return;
        const total = rows_n * (half + sw);
        var a: launch.Args = .{};
        a.add(comb_f32);
        a.add(@as(i32, @intCast(comb_stride)));
        a.add(@as(i32, @intCast(my_col)));
        a.add(peer_f32);
        a.add(gu);
        a.add(@as(i32, @intCast(gu_stride)));
        a.add(@as(i32, @intCast(gate_off)));
        a.add(@as(i32, @intCast(up_off)));
        a.add(out);
        a.add(@as(i32, @intCast(half)));
        a.add(@as(i32, @intCast(sw)));
        a.add(@as(i32, @intCast(rows_n)));
        a.add(@as(i32, @intCast(rank)));
        try launch.launch(r.comb_swiglu_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn argmax(r: *const Rows, s: Stream, logits: u64, vocab_local: usize, vocab_lo: usize, m: usize, out: u64) !void {
        if (m == 0) return;
        var a: launch.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(vocab_local)));
        a.add(@as(i32, @intCast(vocab_lo)));
        a.add(@as(i32, @intCast(m)));
        a.add(out);
        try launch.launch(r.argmax_fn, .{ .grid = .{ .x = @intCast(m), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    pub fn kvWrite(r: *const Rows, s: Stream, qa: u64, qa_stride: usize, q_lora: usize, kv_norm_w: u64, freqs: u64, pos: u64, caches: u64, single_cache: u64, ring: usize, rows_n: usize, eps: f32) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(pos);
        a.add(caches);
        a.add(single_cache);
        a.add(@as(i32, @intCast(ring)));
        a.add(eps);
        try launch.launch(r.kv_write_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn kvWriteEx(r: *const Rows, s: Stream, qa: u64, qa_stride: usize, q_lora: usize, kv_norm_w: u64, freqs: u64, slot_pos: u64, rot_pos: u64, pos3: u64, ld3: usize, caches: u64, single_cache: u64, ring: usize, rows_n: usize, eps: f32) !void {
        if (rows_n == 0) return;
        if (rot_pos == 0 and pos3 == 0) {
            return r.kvWrite(s, qa, qa_stride, q_lora, kv_norm_w, freqs, slot_pos, caches, single_cache, ring, rows_n, eps);
        }
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(slot_pos);
        a.add(rot_pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(caches);
        a.add(single_cache);
        a.add(@as(i32, @intCast(ring)));
        a.add(eps);
        try launch.launch(r.kv_write_ex_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, s, &a);
    }

    pub fn kvWritePaged(r: *const Rows, s: Stream, qa: u64, qa_stride: usize, q_lora: usize, kv_norm_w: u64, freqs: u64, slot_pos: u64, rot_pos: u64, pos3: u64, ld3: usize, page_table: u64, row_stream: u64, pt_stride: usize, arena_base: u64, rows_n: usize, eps: f32) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(slot_pos);
        a.add(rot_pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(page_table);
        a.add(row_stream);
        a.add(@as(i32, @intCast(pt_stride)));
        a.add(arena_base);
        a.add(eps);
        try launch.launch(r.kv_write_paged_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, s, &a);
    }

    pub fn qRope(r: *const Rows, s: Stream, x: u64, xs: usize, xh: usize, out: u64, os: usize, oh: usize, pos: u64, freqs: u64, rows_n: usize, heads: usize) !void {
        if (rows_n == 0 or heads == 0) return;
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(xs)));
        a.add(@as(i64, @intCast(xh)));
        a.add(out);
        a.add(@as(i64, @intCast(os)));
        a.add(@as(i64, @intCast(oh)));
        a.add(pos);
        a.add(freqs);
        try launch.launch(r.q_rope_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn qRopeMrope(r: *const Rows, s: Stream, x: u64, xs: usize, xh: usize, out: u64, os: usize, oh: usize, pos: u64, pos3: u64, ld3: usize, freqs: u64, rows_n: usize, heads: usize) !void {
        if (rows_n == 0 or heads == 0) return;
        if (pos3 == 0) return r.qRope(s, x, xs, xh, out, os, oh, pos, freqs, rows_n, heads);
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(xs)));
        a.add(@as(i64, @intCast(xh)));
        a.add(out);
        a.add(@as(i64, @intCast(os)));
        a.add(@as(i64, @intCast(oh)));
        a.add(pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(freqs);
        try launch.launch(r.q_rope_mrope_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn spliceStreams(r: *const Rows, s: Stream, feat: u64, src_idx: u64, streams: u64, hidden: usize, rows_n: usize) !void {
        if (rows_n == 0 or hidden == 0) return;
        if (hidden % 8 != 0) return error.BadStepShape;
        const hu4 = hidden / 8;
        var a: launch.Args = .{};
        a.add(feat);
        a.add(src_idx);
        a.add(streams);
        a.add(@as(i32, @intCast(hu4)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.splice_streams_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast((hu4 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn vitQkv(r: *const Rows, s: Stream, acc: u64, bias: u64, q: u64, k: u64, v: u64, rows_n: usize, heads: usize, hd: usize) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        inline for (.{ acc, bias, q, k, v }) |p| a.add(p);
        inline for (.{ rows_n, heads, hd }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_qkv_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn vitMergeGather(r: *const Rows, s: Stream, src: u64, dst: u64, gh: usize, gw: usize, d: usize) !void {
        if (gh % 2 != 0 or gw % 2 != 0 or d % 8 != 0) return error.BadStepShape;
        const total = (gh / 2) * (gw / 2) * 4 * (d / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        inline for (.{ gh, gw, d / 8 }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_merge_gather_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn vitAppendNewline(r: *const Rows, s: Stream, merged: u64, newline: u64, out: u64, mh: usize, mw: usize, out_d: usize) !void {
        if (out_d % 8 != 0) return error.BadStepShape;
        const total = mh * (mw + 1) * (out_d / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        inline for (.{ merged, newline, out }) |p| a.add(p);
        inline for (.{ mh, mw, out_d / 8 }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_append_newline_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Pick wide while it fills twice 48 SMs' 2 CTAs, else narrow.
    pub fn pick(m: usize, n: usize) Tile {
        const ctas = gridX(.wide, n) * ((m + tile_rows[0] - 1) / tile_rows[0]);
        return if (ctas >= 192) .wide else .narrow;
    }

    pub fn gridX(t: Tile, n: usize) usize {
        const per = warps * tile_cols[@backingInt(t)];
        return (n + per - 1) / per;
    }

    pub fn rows(r: *const Rows, dense: *const cuda.dense_rows.DenseRows, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        if (m <= 4) return dense.rows(s, x, x_stride, w, out, out_stride, m, n, k);
        const t = pick(m, n);
        const f = if (t == .wide) r.wide_fn else r.narrow_fn;
        return launchTiled(f, t, 0, s, x, x_stride, w, out, out_stride, m, n, k);
    }
};

/// Launch any `hy4_rows_wide` or `hy4_rows_narrow` entry point (production or planted control).
pub fn launchTiled(f: cuda.Function, t: Tile, repeat: i32, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
    if (k % 8 != 0 or x % 16 != 0 or w % 16 != 0 or x_stride % 8 != 0) return error.Misaligned;
    if (m == 0 or n == 0) return;
    var a: launch.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(out_stride)));
    a.add(@as(i32, @intCast(m)));
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(k)));
    a.add(repeat);
    const rt = tile_rows[@backingInt(t)];
    const grid: launch.Dim3 = .{ .x = @intCast(Rows.gridX(t, n)), .y = @intCast((m + rt - 1) / rt), .z = 1 };
    try launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, &a);
}

test "vector launchers reject unsupported remainder widths" {
    const r: Rows = undefined;
    const s: Stream = undefined;
    try std.testing.expectError(error.BadStepShape, r.sliceBf16(s, 0, 64, 0, 9, 2));
    try std.testing.expectError(error.BadStepShape, r.sliceBf16(s, 0, 65, 0, 8, 2));
    try std.testing.expectError(error.BadStepShape, r.interleave2Bf16(s, 0, 0, 10, 2));
    try std.testing.expectError(error.BadStepShape, r.interleaveBf16(s, 0, 0, 10, 2, 2, 0));
    try std.testing.expectError(error.BadStepShape, r.sliceF32(s, 0, 32, 0, 0, 5, 2));
    try std.testing.expectError(error.BadStepShape, r.sliceF32(s, 0, 32, 2, 0, 8, 2));
    try std.testing.expectError(error.BadStepShape, r.sliceF32(s, 0, 33, 0, 0, 8, 2));
    try std.testing.expectError(error.BadStepShape, r.spliceStreams(s, 0, 0, 0, 12, 2));
    try std.testing.expectError(error.BadStepShape, r.vitMergeGather(s, 0, 0, 3, 4, 16));
    try std.testing.expectError(error.BadStepShape, r.vitMergeGather(s, 0, 0, 4, 3, 16));
    try std.testing.expectError(error.BadStepShape, r.vitMergeGather(s, 0, 0, 4, 4, 12));
    try std.testing.expectError(error.BadStepShape, r.vitAppendNewline(s, 0, 0, 0, 2, 2, 12));
}

test "rows_cuda: pick routes vocab slices and prompt routers wide, decode routers narrow" {
    try std.testing.expectEqual(Tile.wide, Rows.pick(16, 60416));
    try std.testing.expectEqual(Tile.wide, Rows.pick(5, 30208));
    try std.testing.expectEqual(Tile.narrow, Rows.pick(16, 512));
    try std.testing.expectEqual(Tile.narrow, Rows.pick(32, 512));
    try std.testing.expectEqual(Tile.wide, Rows.pick(2048, 512));
    std.testing.refAllDecls(Rows);
}
