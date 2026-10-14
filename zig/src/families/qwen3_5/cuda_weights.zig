//! The checkpoint's weights onto the device as qlinear tiles: 4-bit on affine-4, others AffineN (7 on 8), unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Geometry = geometry.Geometry;
const Slot = geometry.Slot;
const Engine = engine.Engine;
const Proj = engine.Proj;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;
const servedBits = cuda.qlinear.servedBits;
const rowBits = cuda.qlinear.rowBits;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const head_dim = geometry.head_dim;

/// The packed tiles' bytes of an n-by-k projection of `bits` codes (qlinear's words, scales and biases).
fn packedBytes(n: usize, k: usize, bits: u8) usize {
    if (servedBits(bits) == 4) {
        const l = Affine4.layout(n, k);
        return aligned(l.words) + 2 * aligned(l.scales);
    }
    const l = AffineN.layout(n, k, servedBits(bits));
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// The checkpoint's own bytes of an n-by-k projection of `bits` codes: words, then its scales and biases.
fn rawBytes(n: usize, k: usize, bits: u8) usize {
    return n * k * bits / 8 + 2 * (n * k / 64 * 2);
}

/// A slot's output and input widths.
fn shape(g: Geometry, slot: Slot) [2]usize {
    return switch (slot) {
        .gate, .up => .{ g.intermediate, g.hidden },
        .down => .{ g.hidden, g.intermediate },
        .qkv => .{ g.convWidth(), g.hidden },
        .z => .{ g.valueWidth(), g.hidden },
        .a, .b => .{ g.v_heads, g.hidden },
        .out => .{ g.hidden, g.valueWidth() },
        .q => .{ 2 * g.attnWidth(), g.hidden },
        .k, .v => .{ g.kv_heads * head_dim, g.hidden },
        .o => .{ g.hidden, g.attnWidth() },
    };
}

/// The slots layer `i` loads: the MLP's and its mixer's.
fn slots(g: Geometry, i: usize) []const Slot {
    return if (g.linear[i]) &.{ .gate, .up, .down, .qkv, .z, .a, .b, .out } else &.{ .gate, .up, .down, .q, .k, .v, .o };
}

/// The token embedding as the embed kernels read it: MLX rows at the width it is served (4 or 8 bits).
fn embedBytes(g: Geometry) usize {
    return aligned(g.vocab * g.hidden * rowBits(g.bits.embed) / 8) + 2 * aligned(g.vocab * g.hidden / 64 * 2);
}

/// What `load` takes from the arena: every packed projection and vector, each allocation aligned.
pub fn total(g: Geometry) usize {
    var t: usize = embedBytes(g);
    t += packedBytes(g.vocab, g.hidden, g.bits.head); // the head, tied or not
    t += aligned(g.hidden * 2);
    for (0..g.layers) |i| {
        t += 2 * aligned(g.hidden * 2);
        for (slots(g, i)) |slot| {
            const nk = shape(g, slot);
            t += packedBytes(nk[0], nk[1], g.bits.of(i, slot));
        }
        if (g.linear[i]) {
            t += aligned(g.convWidth() * geometry.taps * 2) + aligned(geometry.dk * 2) + aligned(g.v_heads * 4) + aligned(g.v_heads * 2);
        } else {
            t += 2 * aligned(head_dim * 2);
        }
    }
    return t;
}

/// The largest projection's bytes as stored: the staging buffer the load packs each one from (and widens through).
pub fn stageBytes(g: Geometry) usize {
    var most = rawBytes(g.vocab, g.hidden, g.bits.head);
    if (rowBits(g.bits.embed) != g.bits.embed) most = @max(most, rawBytes(g.vocab, g.hidden, g.bits.embed));
    for (0..g.layers) |i| for (slots(g, i)) |slot| {
        const nk = shape(g, slot);
        most = @max(most, rawBytes(nk[0], nk[1], g.bits.of(i, slot)));
    };
    return most;
}

fn upload(e: *Engine, bytes: []const u8) !u64 {
    const p = try e.alloc(bytes.len);
    try e.arena.upload(p - e.arena.ptr, bytes);
    return p;
}

fn put(e: *Engine, t: core.checkpoint.Tensor) !u64 {
    return upload(e, t.bytes);
}

/// A vector the kernels read as f32, stored as f32 (the 2B) or bf16 (the 27B's A_log).
fn f32vec(e: *Engine, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [160]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    if (t.is(.f32, &.{n})) return put(e, t);
    if (!t.is(.bf16, &.{n})) return error.UnexpectedTensor;
    const words = try e.gpa.alloc(u32, n);
    defer e.gpa.free(words);
    for (words, std.mem.bytesAsSlice(u16, t.bytes)) |*w, h| w.* = @as(u32, h) << 16;
    return upload(e, std.mem.sliceAsBytes(words));
}

fn tensor(e: *Engine, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape_: []const usize) !u64 {
    var buf: [160]u8 = undefined;
    return put(e, try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape_));
}

/// An MLX affine projection packed into qlinear's tiles from staging; 4-bit takes the affine-4 pack, others AffineN's.
fn linear(e: *Engine, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize, bits: u8) !Proj {
    var buf: [160]u8 = undefined;
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    const parts = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k * bits / 32 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (parts, &raw) |p, *r| {
        const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] });
        try e.stage.upload(at, t.bytes);
        r.* = e.stage.ptr + at;
        at += t.bytes.len;
    }
    const served = servedBits(bits);
    const out: Proj = if (served == 4) blk: {
        const l = Affine4.layout(n, k);
        const q: Affine4 = .{ .w = try e.alloc(l.words), .s = try e.alloc(l.scales), .b = try e.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
        try e.affine.pack(e.stream, raw[0], raw[1], raw[2], q);
        break :blk .{ .a4 = q };
    } else blk: {
        const l = AffineN.layout(n, k, served);
        const q: AffineN = .{ .w = try e.alloc(l.words), .s = try e.alloc(l.scales), .b = try e.alloc(l.scales), .n = n, .k = k, .npad = l.npad, .bits = served };
        try e.wide.packWords(e.stream, raw[0], bits, n, k, q.w);
        try e.wide.packTable(e.stream, raw[1], n, k, q.npad, q.s);
        try e.wide.packTable(e.stream, raw[2], n, k, q.npad, q.b);
        break :blk .{ .an = q };
    };
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return out;
}

/// The token embedding's words for the embed kernels: as stored at 4 or 8 bits, else widened through staging.
fn embedWords(e: *Engine, ck: *core.Checkpoint, m: []const u8) !u64 {
    const g = e.g;
    const bits = g.bits.embed;
    const wide = rowBits(bits);
    var buf: [160]u8 = undefined;
    const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{m}), .u32, &.{ g.vocab, g.hidden * bits / 32 });
    if (bits == wide) return put(e, t);
    try e.stage.upload(0, t.bytes);
    const out = try e.alloc(g.vocab * g.hidden * wide / 8);
    try e.wide.widen(e.stream, e.stage.ptr, bits, g.vocab, g.hidden, out, wide);
    try e.stream.synchronize();
    return out;
}

pub fn load(e: *Engine, ck: *core.Checkpoint) !void {
    const g = e.g;
    const m = "language_model.model.";
    e.embed_bits = rowBits(g.bits.embed);
    e.embed_w = try embedWords(e, ck, m);
    e.embed_s = try tensor(e, ck, "{s}embed_tokens.scales", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.embed_b = try tensor(e, ck, "{s}embed_tokens.biases", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.head = if (g.tied) try linear(e, ck, m, "embed_tokens", g.vocab, g.hidden, g.bits.head) else try linear(e, ck, "language_model.", "lm_head", g.vocab, g.hidden, g.bits.head);
    e.final_norm = try tensor(e, ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    const conv = g.convWidth();
    for (e.layers, 0..) |*l, i| {
        var pbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        var ps: std.enums.EnumArray(Slot, Proj) = .initUndefined();
        for (slots(g, i)) |slot| {
            const nk = shape(g, slot);
            ps.set(slot, try linear(e, ck, p, geometry.slot_modules.get(slot), nk[0], nk[1], g.bits.of(i, slot)));
        }
        l.* = .{
            .input_norm = try tensor(e, ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .post_norm = try tensor(e, ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .gate = ps.get(.gate),
            .up = ps.get(.up),
            .down = ps.get(.down),
            .mixer = undefined,
        };
        if (g.linear[i]) {
            l.mixer = .{ .delta = .{
                .qkv = ps.get(.qkv),
                .z = ps.get(.z),
                .a = ps.get(.a),
                .b = ps.get(.b),
                .out = ps.get(.out),
                .conv = try tensor(e, ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ conv, geometry.taps, 1 }),
                .norm = try tensor(e, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}),
                .a_log = try f32vec(e, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                .dt_bias = try tensor(e, ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{g.v_heads}),
            } };
        } else {
            l.mixer = .{ .attention = .{
                .q = ps.get(.q),
                .k = ps.get(.k),
                .v = ps.get(.v),
                .o = ps.get(.o),
                .q_norm = try tensor(e, ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim}),
                .k_norm = try tensor(e, ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim}),
            } };
        }
    }
}

test "a uniform 4-bit checkpoint keeps its sizes; other modules take their own width" {
    var g: Geometry = undefined;
    g = .{ .hidden = 2048, .intermediate = 6144, .layers = 2, .vocab = 248320, .k_heads = 16, .v_heads = 16, .q_heads = 8, .kv_heads = 2, .eps = 1e-6, .tied = true, .context = 4096, .linear = @splat(false) };
    g.linear[0] = true;
    const uniform = total(g);
    try std.testing.expectEqual(rawBytes(g.vocab, g.hidden, 4), stageBytes(g));
    g.bits.proj[1][@backingInt(Slot.down)] = 8;
    const nk = shape(g, .down);
    try std.testing.expectEqual(uniform - packedBytes(nk[0], nk[1], 4) + packedBytes(nk[0], nk[1], 8), total(g));
    g.bits.proj[1][@backingInt(Slot.down)] = 3; // on 3-bit tiles: 3 bits a weight
    try std.testing.expectEqual(uniform - packedBytes(nk[0], nk[1], 4) + packedBytes(nk[0], nk[1], 3), total(g));
    try std.testing.expect(packedBytes(nk[0], nk[1], 3) < packedBytes(nk[0], nk[1], 4));
    g.bits.proj[1][@backingInt(Slot.down)] = 4;
    g.bits.embed = 6;
    g.bits.head = 6;
    try std.testing.expect(total(g) > uniform);
    try std.testing.expectEqual(rawBytes(g.vocab, g.hidden, 6), stageBytes(g));
}
