//! mla.cu: MLA over a 512-wide latent with 64 roped columns and 128-wide heads (Ling-3.1), on GLM's chunk bodies.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Args = launch_.Args;

/// The widths the kernels are built for: the latent, the nope and value heads, the roped columns.
pub const latent_width = 512;
pub const head_width = 128;
pub const rope_width = 64;
/// Heads a chunk block takes (the r16 form).
pub const heads_tile = 16;
/// Absorb and expand: rows from prompt_rows on go prompt_rb a block (as GLM's latent), fewer rows in one block.
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// A model's MLA shape against the one the kernels are built for, refused before anything is allocated or launched.
pub fn geometry(kv_lora: usize, nope: usize, v_dim: usize, rope_dim: usize) error{MlaGeometry}!void {
    if (kv_lora != latent_width or nope != head_width or v_dim != head_width or rope_dim != rope_width) return error.MlaGeometry;
}

/// Online-softmax partials a chunk writes: o [nch][rows][heads][512] fp32, m and l [nch][rows][heads] fp32.
pub const Partials = struct { po: u64, pm: u64, pl: u64 };

/// The 64 roped columns of `heads` heads a row: in at x + r x_row + h x_head, out likewise (bf16 elements).
pub const RopeRows = struct { x: u64, x_row: usize, x_head: usize, out: u64, o_row: usize, o_head: usize };

/// wide_smem<hb, 64>(): GLM's chunk tiles with q and key rows 576 wide (the roped columns after the latent).
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 576 + 2 * 32 * 576 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// The rope's inv_freq: theta^(-2i / 64) for i < 32, from float64 rounded to fp32.
pub fn ropeFreqs(theta: f64) [rope_width / 2]f32 {
    var f: [rope_width / 2]f32 = undefined;
    for (&f, 0..) |*x, i| x.* = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / rope_width));
    return f;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

pub const Mla = struct {
    mod: Module,
    chunks_fn: Function,
    chunks_split_fn: Function,
    merge_part_fn: Function,
    absorb_q_fn: Function,
    expand_v_fn: Function,
    rope_fn: Function,
    gate_fn: Function,

    pub fn load(d: *const Driver) !Mla {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.mla);
        errdefer mod.unload();
        var m: Mla = .{ .mod = mod, .chunks_fn = undefined, .chunks_split_fn = undefined, .merge_part_fn = undefined, .absorb_q_fn = undefined, .expand_v_fn = undefined, .rope_fn = undefined, .gate_fn = undefined };
        inline for (.{ "chunks", "chunks_split", "merge_part", "absorb_q", "expand_v", "rope", "gate" }, .{ "tf_mla_chunks_r16", "tf_mla_chunks_split_r16", "tf_mla_merge_part", "tf_mla_absorb_q_128", "tf_mla_expand_v_128", "tf_mla_rope", "tf_mla_gate" }) |field, name|
            @field(m, field ++ "_fn") = try mod.function(name);
        try m.chunks_fn.allowDynamicShared(chunksSmem(heads_tile));
        try m.chunks_split_fn.allowDynamicShared(chunksSmem(heads_tile));
        return m;
    }

    pub fn unload(m: *Mla) void {
        m.mod.unload();
    }

    /// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
    fn rowBlocks(f: Function, s: Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize, extra: anytype) !void {
        const rb = if (rows >= prompt_rows) prompt_rb else rows;
        var a: Args = .{};
        inline for (.{ in, w, out, i64of(rows), i64of(heads), i64of(rb) } ++ extra) |v| a.add(v);
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The absorb: q_nope the first d = 128 of each head's `q_stride` columns, out rows `out_stride` apart per head.
    pub fn absorbQ(m: *const Mla, s: Stream, q: u64, q_stride: usize, wk: u64, out: u64, out_stride: usize, rows: usize, heads: usize, d: usize, lw: usize) !void {
        if (d != head_width or lw != latent_width or q_stride < d or out_stride < lw) return error.LatentShape;
        if (rows == 0) return;
        try rowBlocks(m.absorb_q_fn, s, q, wk, out, rows, heads, lw / 32, .{ i64of(q_stride), i64of(out_stride) });
    }

    /// out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw], 128-wide v heads.
    pub fn expandV(m: *const Mla, s: Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
        if (lw != latent_width or dv != head_width) return error.LatentShape;
        if (rows == 0) return;
        try rowBlocks(m.expand_v_fn, s, ol, wv, out, rows, heads, dv / 16, .{});
    }

    /// The GPT-J rope at device i32 pos + r, fp32 inv_freq [32], cos and sin times `scale` (YaRN's; 1: plain).
    pub fn rope(m: *const Mla, s: Stream, v: RopeRows, rows: usize, heads: usize, pos: u64, inv_freq: u64, scale: f32) !void {
        if (rows == 0 or heads == 0) return;
        var a: Args = .{};
        inline for (.{ v.x, i64of(v.x_row), i64of(v.x_head), v.out, i64of(v.o_row), i64of(v.o_head), pos, inv_freq, scale }) |x| a.add(x);
        try launch_.launch(m.rope_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = rope_width / 2 } }, s, &a);
    }

    /// The chunk tiles with roped keys: qa and cache rows lw + 64 wide, 16-head blocks, nch chunks' partials into `p`.
    pub fn chunks(m: *const Mla, s: Stream, qa: u64, cache: u64, pos: u64, p: Partials, rows: usize, heads: usize, nch: usize, scale: f32) !void {
        if (rows == 0 or nch == 0) return;
        var a: Args = .{};
        inline for (.{ qa, cache, pos, @as(u64, 0), @as(u64, 0), p.po, p.pm, p.pl, i64of(rows), i64of(heads), @as(i64, 0), scale, @as(c_int, 0), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
        try launch_.launch(m.chunks_fn, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, heads_tile), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = chunksSmem(heads_tile) }, s, &a);
    }

    /// The chunk tiles over a split cache: this rank's `nch` chunks (chunk z * world + rank of the positions) into `p`.
    pub fn chunksSplit(m: *const Mla, s: Stream, qa: u64, cache: u64, pos: u64, p: Partials, rows: usize, heads: usize, nch: usize, scale: f32, world: usize, rank: usize) !void {
        if (nch == 0 or rows == 0) return;
        if (world == 0 or rank >= world) return error.SplitRank;
        var a: Args = .{};
        inline for (.{ qa, cache, pos, p.po, p.pm, p.pl, i64of(rows), i64of(heads), scale, cint(world), cint(rank) }) |v| a.add(v);
        try launch_.launch(m.chunks_split_fn, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, heads_tile), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = chunksSmem(heads_tile) }, s, &a);
    }

    /// `nch` chunks' partials merged into one chunk's (o, m, l in fp32) at `out`: nch 0 writes the empty one.
    pub fn mergePart(m: *const Mla, s: Stream, p: Partials, out: Partials, rows: usize, heads: usize, nch: usize) !void {
        if (rows == 0) return;
        var a: Args = .{};
        inline for (.{ p.po, p.pm, p.pl, out.po, out.pm, out.pl, i64of(rows), i64of(heads), cint(nch) }) |v| a.add(v);
        try launch_.launch(m.merge_part_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// o [rows][heads][dv] bf16 *= bf16(sigmoid(g [rows][gs] at column h)): the head-wise output gate, in place.
    pub fn gate(m: *const Mla, s: Stream, o: u64, g: u64, gs: usize, rows: usize, heads: usize, dv: usize) !void {
        if (dv == 0 or dv > 1024 or gs < heads) return error.GateShape;
        if (rows == 0 or heads == 0) return;
        var a: Args = .{};
        inline for (.{ o, i64of(heads), g, i64of(gs) }) |v| a.add(v);
        try launch_.launch(m.gate_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = @intCast(dv) } }, s, &a);
    }
};

test "mla: the roped chunk tiles fit the 99 KiB a GB10 block may take, and other widths are refused" {
    try std.testing.expect(chunksSmem(heads_tile) <= 99 * 1024);
    try std.testing.expectEqual(@as(f32, 1.0), ropeFreqs(6e6)[0]);
    try geometry(512, 128, 128, 64);
    for ([_][4]usize{ .{ 256, 128, 128, 64 }, .{ 512, 192, 128, 64 }, .{ 512, 128, 256, 64 }, .{ 512, 128, 128, 32 } }) |x|
        try std.testing.expectError(error.MlaGeometry, geometry(x[0], x[1], x[2], x[3]));
}
