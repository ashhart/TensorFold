//! The checkpoint's weights onto the device: affine-4 projections packed as qlinear tiles, out of the engine's arena.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Geometry = geometry.Geometry;
const Engine = engine.Engine;
const Affine4 = cuda.qlinear.Affine4;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const head_dim = geometry.head_dim;

/// The packed tiles' bytes of an n-by-k affine-4 projection (qlinear's words, scales and biases).
pub fn packedBytes(n: usize, k: usize) usize {
    const l = Affine4.layout(n, k);
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// The checkpoint's own bytes of an n-by-k affine-4 projection: words, then its scales and biases.
fn rawBytes(n: usize, k: usize) usize {
    return n * k / 8 * 4 + 2 * (n * k / 64 * 2);
}

/// A layer's routed experts' device bytes: the pair table, the down table and the router rows.
pub fn routedBytes(experts: usize, intermediate: usize, hidden: usize) usize {
    const ex = cuda.experts.Affine4Experts;
    return aligned(ex.pairBytes(experts + 1, intermediate, hidden)) + aligned(ex.bytes(experts + 1, hidden, intermediate)) + aligned((experts + 1) * hidden * 2);
}

/// What `load` takes from the arena: every packed projection and vector, each allocation aligned.
pub fn total(g: Geometry) usize {
    var t: usize = 0;
    t += aligned(g.vocab * g.hidden / 8 * 4) + 2 * aligned(g.vocab * g.hidden / 64 * 2); // the embedding as stored
    t += packedBytes(g.vocab, g.hidden); // the head, tied or not
    t += aligned(g.hidden * 2);
    for (g.linear[0..g.layers]) |lin| {
        t += 2 * aligned(g.hidden * 2);
        if (g.moe()) t += routedBytes(g.experts, g.intermediate, g.hidden) else t += 2 * packedBytes(g.intermediate, g.hidden) + packedBytes(g.hidden, g.intermediate);
        if (lin) {
            t += packedBytes(g.convWidth(), g.hidden) + packedBytes(g.valueWidth(), g.hidden) + 2 * packedBytes(g.v_heads, g.hidden) + packedBytes(g.hidden, g.valueWidth());
            t += aligned(g.convWidth() * geometry.taps * 2) + aligned(geometry.dk * 2) + aligned(g.v_heads * 4) + aligned(g.v_heads * 2);
        } else {
            t += packedBytes(2 * g.attnWidth(), g.hidden) + 2 * packedBytes(g.kv_heads * head_dim, g.hidden) + packedBytes(g.hidden, g.attnWidth());
            t += 2 * aligned(head_dim * 2);
        }
    }
    return t;
}

/// The largest projection's bytes as stored: the staging buffer the load packs each one from.
pub fn stageBytes(g: Geometry) usize {
    var most: usize = 0;
    for ([_][2]usize{ .{ g.vocab, g.hidden }, .{ g.intermediate, g.hidden }, .{ g.hidden, g.intermediate }, .{ g.convWidth(), g.hidden }, .{ g.valueWidth(), g.hidden }, .{ 2 * g.attnWidth(), g.hidden } }) |nk| most = @max(most, rawBytes(nk[0], nk[1]));
    if (g.moe()) most = @max(most, @max(rawBytes(g.experts * g.intermediate, g.hidden), rawBytes(g.experts * g.hidden, g.intermediate)));
    return most;
}

/// `into` is the engine or a drafter's weights: anything with `alloc(bytes)` out of its `arena`.
pub fn upload(into: anytype, bytes: []const u8) !u64 {
    const p = try into.alloc(bytes.len);
    try into.arena.upload(p - into.arena.ptr, bytes);
    return p;
}

fn put(into: anytype, t: core.checkpoint.Tensor) !u64 {
    return upload(into, t.bytes);
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

pub fn tensor(into: anytype, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
    var buf: [160]u8 = undefined;
    return put(into, try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape));
}

/// Raw words, scales and biases staged in order; each part's device pointer.
fn stage(e: *Engine, parts: [3][]const u8) ![3]u64 {
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    for (parts, &raw) |p, *r| {
        if (at + p.len > e.stage.len) return error.StageTooSmall;
        try e.stage.upload(at, p);
        r.* = e.stage.ptr + at;
        at += p.len;
    }
    return raw;
}

fn packStaged(e: *Engine, into: anytype, raw: [3]u64, n: usize, k: usize) !Affine4 {
    const l = Affine4.layout(n, k);
    const q: Affine4 = .{ .w = try into.alloc(l.words), .s = try into.alloc(l.scales), .b = try into.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try e.affine.pack(e.stream, raw[0], raw[1], raw[2], q);
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return q;
}

/// An MLX affine-4 projection packed into qlinear's tiles from the staging buffer, which holds it as stored.
pub fn linear(e: *Engine, into: anytype, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize) !Affine4 {
    var buf: [160]u8 = undefined;
    var parts: [3][]const u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (specs, &parts) |p, *part| part.* = (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] })).bytes;
    return packStaged(e, into, try stage(e, parts), n, k);
}

/// Inputs [from, from + len) of an n-by-k projection as their own packed projection (the MTP's fc halves).
pub fn linearSlice(e: *Engine, into: anytype, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize, from: usize, len: usize) !Affine4 {
    var buf: [160]u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize, usize }{ .{ "weight", .u32, k / 8, 4 }, .{ "scales", .bf16, k / 64, 2 }, .{ "biases", .bf16, k / 64, 2 } };
    var parts: [3][]u8 = undefined;
    var taken: usize = 0;
    errdefer for (parts[0..taken]) |p| e.gpa.free(p);
    for (specs, &parts) |p, *part| {
        const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] });
        const per = k / p[2]; // inputs a column holds
        const cols = len / per;
        part.* = try e.gpa.alloc(u8, n * cols * p[3]);
        taken += 1;
        for (0..n) |r| @memcpy(part.*[r * cols * p[3] ..][0 .. cols * p[3]], t.bytes[(r * p[2] + from / per) * p[3] ..][0 .. cols * p[3]]);
    }
    defer for (parts) |p| e.gpa.free(p);
    return packStaged(e, into, try stage(e, .{ parts[0], parts[1], parts[2] }), n, len);
}

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// f32 to bf16, round to nearest even (finite values).
fn f32bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// 8-bit affine rows (four codes a word, a scale and bias a group of 64) dequantized to bf16: the router rows.
fn dequant8(dst: []u16, words: []align(1) const u32, scales: []align(1) const u16, biases: []align(1) const u16, k: usize) void {
    const groups = k / 64;
    for (0..dst.len / k) |r| {
        for (0..k) |i| {
            const code: f32 = @floatFromInt((words[r * (k / 4) + i / 4] >> @intCast(8 * (i % 4))) & 0xff);
            dst[r * k + i] = f32bf16(code * bf16f(scales[r * groups + i / 64]) + bf16f(biases[r * groups + i / 64]));
        }
    }
}

/// The router's rows as bf16 [experts + 1][hidden]: the routed experts' gate, then the shared expert's gate row.
fn routerRows(e: *Engine, into: anytype, ck: *core.Checkpoint, prefix: []const u8) !u64 {
    const g = e.g;
    const rows = try e.gpa.alloc(u16, (g.experts + 1) * g.hidden);
    defer e.gpa.free(rows);
    var buf: [160]u8 = undefined;
    const names = [_]struct { []const u8, usize, usize }{ .{ "mlp.gate", 0, g.experts }, .{ "mlp.shared_expert_gate", g.experts, 1 } };
    for (names) |n| {
        const w = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, n[0] }), .u32, &.{ n[2], g.hidden / 4 });
        const s = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.scales", .{ prefix, n[0] }), .bf16, &.{ n[2], g.hidden / 64 });
        const b = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.biases", .{ prefix, n[0] }), .bf16, &.{ n[2], g.hidden / 64 });
        dequant8(rows[n[1] * g.hidden ..][0 .. n[2] * g.hidden], std.mem.bytesAsSlice(u32, w.bytes), std.mem.bytesAsSlice(u16, s.bytes), std.mem.bytesAsSlice(u16, b.bytes), g.hidden);
    }
    return upload(into, std.mem.sliceAsBytes(rows));
}

/// `count` stacked experts' projection (one at count 0) staged as stored and packed into table `out` as the mi-th of m.
fn expertTable(e: *Engine, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, count: usize, n: usize, k: usize, out: u64, m: usize, mi: usize) !void {
    var buf: [160]u8 = undefined;
    var parts: [3][]const u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (specs, &parts) |p, *part| {
        const stacked: []const usize = &.{ count, n, p[2] };
        const one: []const usize = &.{ n, p[2] };
        part.* = (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], if (count == 0) one else stacked)).bytes;
    }
    const raw = try stage(e, parts);
    try e.experts.packInto(e.stream, raw[0], raw[1], raw[2], out, @max(count, 1), n, k, m, mi);
    try e.stream.synchronize(); // the next table's staging overwrites this one's
}

/// A layer's routed experts: gate and up interleaved as one pair table, down apart, the shared expert last in each.
pub fn routed(e: *Engine, into: anytype, ck: *core.Checkpoint, p: []const u8) !engine.Mlp {
    const g = e.g;
    const ex = cuda.experts.Affine4Experts;
    const ni = g.intermediate;
    const pair = try into.alloc(ex.pairBytes(g.experts + 1, ni, g.hidden));
    const down = try into.alloc(ex.bytes(g.experts + 1, g.hidden, ni));
    try expertTable(e, ck, p, "mlp.switch_mlp.gate_proj", g.experts, ni, g.hidden, pair, 2, 0);
    try expertTable(e, ck, p, "mlp.switch_mlp.up_proj", g.experts, ni, g.hidden, pair, 2, 1);
    try expertTable(e, ck, p, "mlp.switch_mlp.down_proj", g.experts, g.hidden, ni, down, 1, 0);
    const shared_pair = pair + ex.pairBytes(g.experts, ni, g.hidden);
    try expertTable(e, ck, p, "mlp.shared_expert.gate_proj", 0, ni, g.hidden, shared_pair, 2, 0);
    try expertTable(e, ck, p, "mlp.shared_expert.up_proj", 0, ni, g.hidden, shared_pair, 2, 1);
    try expertTable(e, ck, p, "mlp.shared_expert.down_proj", 0, g.hidden, ni, down + ex.bytes(g.experts, g.hidden, ni), 1, 0);
    const layer: cuda.experts.Layer = .{ .format = .affine4, .up = pair, .down = down, .width = ni, .dims = g.hidden, .experts = g.experts + 1, .swiglu = true };
    return .{ .moe = .{ .router = try routerRows(e, into, ck, p), .experts = layer } };
}

pub fn load(e: *Engine, ck: *core.Checkpoint) !void {
    const g = e.g;
    const m = "language_model.model.";
    e.embed_w = try tensor(e, ck, "{s}embed_tokens.weight", .{m}, .u32, &.{ g.vocab, g.hidden / 8 });
    e.embed_s = try tensor(e, ck, "{s}embed_tokens.scales", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.embed_b = try tensor(e, ck, "{s}embed_tokens.biases", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.head = if (g.tied) try linear(e, e, ck, m, "embed_tokens", g.vocab, g.hidden) else try linear(e, e, ck, "language_model.", "lm_head", g.vocab, g.hidden);
    e.final_norm = try tensor(e, ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    const conv = g.convWidth();
    const vw = g.valueWidth();
    for (e.layers, 0..) |*l, i| {
        var pbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        l.* = .{
            .input_norm = try tensor(e, ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .post_norm = try tensor(e, ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .mlp = if (g.moe()) try routed(e, e, ck, p) else .{ .dense = .{
                .gate = try linear(e, e, ck, p, "mlp.gate_proj", g.intermediate, g.hidden),
                .up = try linear(e, e, ck, p, "mlp.up_proj", g.intermediate, g.hidden),
                .down = try linear(e, e, ck, p, "mlp.down_proj", g.hidden, g.intermediate),
            } },
            .mixer = undefined,
        };
        if (g.linear[i]) {
            l.mixer = .{ .delta = .{
                .qkv = try linear(e, e, ck, p, "linear_attn.in_proj_qkv", conv, g.hidden),
                .z = try linear(e, e, ck, p, "linear_attn.in_proj_z", vw, g.hidden),
                .a = try linear(e, e, ck, p, "linear_attn.in_proj_a", g.v_heads, g.hidden),
                .b = try linear(e, e, ck, p, "linear_attn.in_proj_b", g.v_heads, g.hidden),
                .out = try linear(e, e, ck, p, "linear_attn.out_proj", g.hidden, vw),
                .conv = try tensor(e, ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ conv, geometry.taps, 1 }),
                .norm = try tensor(e, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}),
                .a_log = try f32vec(e, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                .dt_bias = try tensor(e, ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{g.v_heads}),
            } };
        } else {
            l.mixer = .{ .attention = .{
                .q = try linear(e, e, ck, p, "self_attn.q_proj", 2 * g.attnWidth(), g.hidden),
                .k = try linear(e, e, ck, p, "self_attn.k_proj", g.kv_heads * head_dim, g.hidden),
                .v = try linear(e, e, ck, p, "self_attn.v_proj", g.kv_heads * head_dim, g.hidden),
                .o = try linear(e, e, ck, p, "self_attn.o_proj", g.hidden, g.attnWidth()),
                .q_norm = try tensor(e, ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim}),
                .k_norm = try tensor(e, ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim}),
            } };
        }
    }
}
