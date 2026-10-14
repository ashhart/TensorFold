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
fn packedBytes(n: usize, k: usize) usize {
    const l = Affine4.layout(n, k);
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// The checkpoint's own bytes of an n-by-k affine-4 projection: words, then its scales and biases.
fn rawBytes(n: usize, k: usize) usize {
    return n * k / 8 * 4 + 2 * (n * k / 64 * 2);
}

/// What `load` takes from the arena: every packed projection and vector, each allocation aligned.
pub fn total(g: Geometry) usize {
    var t: usize = 0;
    t += aligned(g.vocab * g.hidden / 8 * 4) + 2 * aligned(g.vocab * g.hidden / 64 * 2); // the embedding as stored
    t += packedBytes(g.vocab, g.hidden); // the head, tied or not
    t += aligned(g.hidden * 2);
    for (g.linear[0..g.layers]) |lin| {
        t += 2 * aligned(g.hidden * 2) + 2 * packedBytes(g.intermediate, g.hidden) + packedBytes(g.hidden, g.intermediate);
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

fn tensor(e: *Engine, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
    var buf: [160]u8 = undefined;
    return put(e, try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape));
}

/// An MLX affine-4 projection packed into qlinear's tiles from the staging buffer, which holds it as stored.
fn linear(e: *Engine, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize) !Affine4 {
    var buf: [160]u8 = undefined;
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    const parts = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (parts, &raw) |p, *r| {
        const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] });
        try e.stage.upload(at, t.bytes);
        r.* = e.stage.ptr + at;
        at += t.bytes.len;
    }
    const l = Affine4.layout(n, k);
    const q: Affine4 = .{ .w = try e.alloc(l.words), .s = try e.alloc(l.scales), .b = try e.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try e.affine.pack(e.stream, raw[0], raw[1], raw[2], q);
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return q;
}

pub fn load(e: *Engine, ck: *core.Checkpoint) !void {
    const g = e.g;
    const m = "language_model.model.";
    e.embed_w = try tensor(e, ck, "{s}embed_tokens.weight", .{m}, .u32, &.{ g.vocab, g.hidden / 8 });
    e.embed_s = try tensor(e, ck, "{s}embed_tokens.scales", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.embed_b = try tensor(e, ck, "{s}embed_tokens.biases", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.head = if (g.tied) try linear(e, ck, m, "embed_tokens", g.vocab, g.hidden) else try linear(e, ck, "language_model.", "lm_head", g.vocab, g.hidden);
    e.final_norm = try tensor(e, ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    const conv = g.convWidth();
    const vw = g.valueWidth();
    for (e.layers, 0..) |*l, i| {
        var pbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        l.* = .{
            .input_norm = try tensor(e, ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .post_norm = try tensor(e, ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .gate = try linear(e, ck, p, "mlp.gate_proj", g.intermediate, g.hidden),
            .up = try linear(e, ck, p, "mlp.up_proj", g.intermediate, g.hidden),
            .down = try linear(e, ck, p, "mlp.down_proj", g.hidden, g.intermediate),
            .mixer = undefined,
        };
        if (g.linear[i]) {
            l.mixer = .{ .delta = .{
                .qkv = try linear(e, ck, p, "linear_attn.in_proj_qkv", conv, g.hidden),
                .z = try linear(e, ck, p, "linear_attn.in_proj_z", vw, g.hidden),
                .a = try linear(e, ck, p, "linear_attn.in_proj_a", g.v_heads, g.hidden),
                .b = try linear(e, ck, p, "linear_attn.in_proj_b", g.v_heads, g.hidden),
                .out = try linear(e, ck, p, "linear_attn.out_proj", g.hidden, vw),
                .conv = try tensor(e, ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ conv, geometry.taps, 1 }),
                .norm = try tensor(e, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}),
                .a_log = try f32vec(e, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                .dt_bias = try tensor(e, ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{g.v_heads}),
            } };
        } else {
            l.mixer = .{ .attention = .{
                .q = try linear(e, ck, p, "self_attn.q_proj", 2 * g.attnWidth(), g.hidden),
                .k = try linear(e, ck, p, "self_attn.k_proj", g.kv_heads * head_dim, g.hidden),
                .v = try linear(e, ck, p, "self_attn.v_proj", g.kv_heads * head_dim, g.hidden),
                .o = try linear(e, ck, p, "self_attn.o_proj", g.hidden, g.attnWidth()),
                .q_norm = try tensor(e, ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim}),
                .k_norm = try tensor(e, ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim}),
            } };
        }
    }
}
