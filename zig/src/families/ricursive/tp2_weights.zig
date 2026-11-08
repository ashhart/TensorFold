//! Ripple's EXL3 pack on CUDA: trellis projections as exl3.Weight, norms as fp32, a rank's share at TP cuts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");

const Weight = cuda.exl3.Weight;
const mul1_magic: u32 = 0x83DCD12D;

/// A rank's share: q/k/v/g, gate/up and the head by output columns; o and down by input rows (their sums added once).
pub const Cut = struct {
    rank: usize = 0,
    world: usize = 1,

    /// [lo, hi) of `n` for this rank, whole 128-wide blocks.
    pub fn span(c: Cut, n: usize) !struct { lo: usize, hi: usize } {
        if (c.world == 0 or c.rank >= c.world or n % (128 * c.world) != 0) return error.RicursiveCut;
        const part = n / c.world;
        return .{ .lo = c.rank * part, .hi = (c.rank + 1) * part };
    }
};

pub const Layer = struct {
    global: bool,
    input_norm: u64,
    post_attn: u64,
    pre_ffn: u64,
    post_ffn: u64,
    /// q_norm then k_norm, fp32 [2, 128].
    qk_norm: u64,
    /// sigmoid(value_res_lambda): v_proj's input is lerp(seed, normed, blend).
    blend: f32,
    q: Weight,
    k: Weight,
    v: Weight,
    g: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    embed: u64 = 0,
    final_norm: u64 = 0,
    head: Weight = undefined,
    layers: []Layer = &.{},
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        w.gpa.free(w.layers);
    }

    fn put(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try w.bufs.ensureUnusedCapacity(w.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        w.bufs.appendAssumeCapacity(b);
        w.bytes += bytes.len;
        return b.ptr;
    }
};

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// A bf16 vector [n] widened to fp32 on the device (norm weights: y = x * w, no offset).
fn f32Vec(ck: *core.Checkpoint, key: []const u8, n: usize, out: []f32) !void {
    const t = try ck.expect(key, .bf16, &.{n});
    for (out[0..n], 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
}

fn norm(w: *Weights, d: *const cuda.Driver, ck: *core.Checkpoint, key: []const u8, n: usize) !u64 {
    const v = try w.gpa.alloc(f32, n);
    defer w.gpa.free(v);
    try f32Vec(ck, key, n, v);
    return w.put(d, std.mem.sliceAsBytes(v));
}

pub const Side = enum { columns, rows };

/// An n-by-k EXL3 projection `base` (trellis [k/16][n/16][8 * halves] i16, mul1, fp16 suh [k] and svh [n]), cut.
pub fn exl3(w: *Weights, d: *const cuda.Driver, ck: *core.Checkpoint, base: []const u8, n: usize, k: usize, side: Side, cut: Cut) !Weight {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try name(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) return error.UnexpectedTensor;
    const halves = t.dim(2) / 8;
    const marker = try ck.get(try name(&buf, "{s}.mul1", .{base}));
    if (marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedRicursiveQuantization;
    const suh = try ck.get(try name(&buf, "{s}.suh", .{base}));
    const svh = try ck.get(try name(&buf, "{s}.svh", .{base}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedTensor;
    const tile = t.dim(2) * 2; // bytes of one 16x16 tile
    var rk = k;
    var rn = n;
    var trellis: u64 = 0;
    var su: u64 = 0;
    var sv: u64 = 0;
    if (cut.world == 1) {
        trellis = try w.put(d, t.bytes);
        su = try w.put(d, suh.bytes);
        sv = try w.put(d, svh.bytes);
    } else if (side == .rows) { // input rows: a contiguous run of k/16 tile rows and suh's matching span
        const sp = try cut.span(k);
        rk = sp.hi - sp.lo;
        const row = (n / 16) * tile;
        trellis = try w.put(d, t.bytes[sp.lo / 16 * row .. sp.hi / 16 * row]);
        su = try w.put(d, suh.bytes[sp.lo * 2 .. sp.hi * 2]);
        sv = try w.put(d, svh.bytes);
    } else { // output columns: each tile row's span of n/16 tiles, gathered on the host
        const sp = try cut.span(n);
        rn = sp.hi - sp.lo;
        const row = (n / 16) * tile;
        const part = (rn / 16) * tile;
        const g = try w.gpa.alloc(u8, (k / 16) * part);
        defer w.gpa.free(g);
        for (0..k / 16) |r| @memcpy(g[r * part ..][0..part], t.bytes[r * row + sp.lo / 16 * tile ..][0..part]);
        trellis = try w.put(d, g);
        su = try w.put(d, suh.bytes);
        sv = try w.put(d, svh.bytes[sp.lo * 2 .. sp.hi * 2]);
    }
    const st = cuda.exl3.strides(rk, rn, halves, false);
    const out: Weight = .{ .trellis = trellis, .suh = su, .svh = sv, .k = @intCast(rk), .n = @intCast(rn), .halves = @intCast(halves), .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    if (cuda.exl3.variantIndex(out.halves, out.codebook, cuda.exl3.planOf(out).wk) == null) return error.UnsupportedRicursiveQuantization;
    return out;
}

/// The whole pack (or a rank's share) onto the device.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *core.Checkpoint, c: config.Config, cut: Cut) !Weights {
    if (!c.exl3) return error.UnsupportedRicursivePrecision;
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    const h = c.hidden;
    const t = try ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, h });
    w.embed = try w.put(d, t.bytes); // every rank holds the whole table (the value residual's seed is the embedding)
    w.final_norm = try norm(&w, d, ck, "model.norm.weight", h);
    w.head = try exl3(&w, d, ck, "lm_head", c.vocab, h, .columns, cut);
    w.layers = try gpa.alloc(Layer, c.layers);
    var buf: [200]u8 = undefined;
    var pb: [64]u8 = undefined;
    for (w.layers, 0..) |*l, i| {
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        l.global = c.global[i];
        l.input_norm = try norm(&w, d, ck, try name(&buf, "{s}input_layernorm.weight", .{p}), h);
        l.post_attn = try norm(&w, d, ck, try name(&buf, "{s}post_attention_layernorm.weight", .{p}), h);
        l.pre_ffn = try norm(&w, d, ck, try name(&buf, "{s}pre_feedforward_layernorm.weight", .{p}), h);
        l.post_ffn = try norm(&w, d, ck, try name(&buf, "{s}post_feedforward_layernorm.weight", .{p}), h);
        var qk: [2 * config.head_dim]f32 = undefined;
        try f32Vec(ck, try name(&buf, "{s}self_attn.q_norm.weight", .{p}), config.head_dim, qk[0..config.head_dim]);
        try f32Vec(ck, try name(&buf, "{s}self_attn.k_norm.weight", .{p}), config.head_dim, qk[config.head_dim..]);
        l.qk_norm = try w.put(d, std.mem.sliceAsBytes(&qk));
        const lam = try ck.get(try name(&buf, "{s}self_attn.value_res_lambda", .{p}));
        if (lam.dtype != .f32 or lam.numel() != 1) return error.UnexpectedTensor;
        const x: f32 = @bitCast(std.mem.readInt(u32, lam.bytes[0..4], .little));
        l.blend = 1.0 / (1.0 + @exp(-x)); // torch.sigmoid(lambda).item(), as the reference folds it once
        const qw = c.queryWidth();
        const kw = c.kvWidth();
        l.q = try exl3(&w, d, ck, try name(&buf, "{s}self_attn.q_proj", .{p}), qw, h, .columns, cut);
        l.k = try exl3(&w, d, ck, try name(&buf, "{s}self_attn.k_proj", .{p}), kw, h, .columns, cut);
        l.v = try exl3(&w, d, ck, try name(&buf, "{s}self_attn.v_proj", .{p}), kw, h, .columns, cut);
        l.g = try exl3(&w, d, ck, try name(&buf, "{s}self_attn.g_proj", .{p}), qw, h, .columns, cut);
        l.o = try exl3(&w, d, ck, try name(&buf, "{s}self_attn.o_proj", .{p}), h, qw, .rows, cut);
        l.gate = try exl3(&w, d, ck, try name(&buf, "{s}mlp.gate_proj", .{p}), c.intermediate, h, .columns, cut);
        l.up = try exl3(&w, d, ck, try name(&buf, "{s}mlp.up_proj", .{p}), c.intermediate, h, .columns, cut);
        l.down = try exl3(&w, d, ck, try name(&buf, "{s}mlp.down_proj", .{p}), h, c.intermediate, .rows, cut);
    }
    return w;
}

test "ricursive cut: halves on 128-wide blocks, refused off them" {
    const c: Cut = .{ .rank = 1, .world = 2 };
    const s = try c.span(10240);
    try std.testing.expectEqual(@as(usize, 5120), s.lo);
    try std.testing.expectEqual(@as(usize, 10240), s.hi);
    try std.testing.expectEqual(@as(usize, 50176), (try c.span(100352)).lo);
    try std.testing.expectError(error.RicursiveCut, c.span(1000));
    try std.testing.expectError(error.RicursiveCut, (Cut{ .rank = 2, .world = 2 }).span(1024));
}
