//! The audio tower's weights by name and its schedule for every backend; `r` binds the kernels, `up` the uploads.
const std = @import("std");
const cfg = @import("config.zig");
const chunks = @import("chunks.zig");
const mel = @import("mel.zig");
const names = @import("names.zig");
const tower_ref = @import("tower_ref.zig");

pub const Layer = struct { ln1: u64, ln1_b: u64, qkv: u64, qkv_b: u64, out: u64, out_b: u64, ln2: u64, ln2_b: u64, fc1: u64, fc1_b: u64, fc2: u64, fc2_b: u64 };

/// A GELU after a Linear, or none.
pub const Act = enum { none, gelu };

/// The tower's tensors as device addresses, each put by the backend's `up` (q, k and v as one [3 d, d] matrix).
pub const Weights = struct {
    conv: [3]u64,
    conv_b: [3]u64,
    conv_out: u64,
    positions: u64, // the per-chunk sinusoid rows (bf16 as the package casts them), tiled over `group` chunks
    layers: []Layer,
    ln_post: u64,
    ln_post_b: u64,
    proj1: u64,
    proj1_b: u64,
    proj2: u64,
    proj2_b: u64,
    group: usize,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }

    /// `src.expect(name, shape)` gives a bf16 tensor's bytes; `up.put(bytes)` and `up.taps(bytes, co, k)` place them.
    pub fn load(gpa: std.mem.Allocator, a: cfg.Audio, src: anytype, up: anytype, group: usize) !Weights {
        const t = names.tower;
        const k = cfg.conv_taps;
        var w: Weights = undefined;
        w.group = group;
        w.conv[0] = try up.taps(try src.expect(t ++ "conv2d1.weight", &.{ a.conv, 1, k, k }), a.conv, 9);
        w.conv[1] = try up.taps(try src.expect(t ++ "conv2d2.weight", &.{ a.conv, a.conv, k, k }), a.conv, 9 * a.conv);
        w.conv[2] = try up.taps(try src.expect(t ++ "conv2d3.weight", &.{ a.conv, a.conv, k, k }), a.conv, 9 * a.conv);
        inline for (.{ "conv2d1.bias", "conv2d2.bias", "conv2d3.bias" }, 0..) |n, i| w.conv_b[i] = try up.put(try src.expect(t ++ n, &.{a.conv}));
        w.conv_out = try up.put(try src.expect(t ++ "conv_out.weight", &.{ a.d, a.conv * a.convBins() }));
        w.positions = try positions(gpa, a, up, group);
        w.layers = try gpa.alloc(Layer, a.layers);
        errdefer gpa.free(w.layers);
        var nb: [4][160]u8 = undefined;
        for (w.layers, 0..) |*l, i| {
            const p = struct {
                fn f(b: *[160]u8, li: usize, n: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, names.tower ++ "layers.{d}.{s}", .{ li, n });
                }
            }.f;
            const sq = &[_]usize{ a.d, a.d };
            const v = &[_]usize{a.d};
            const qkv = [3][]const u8{ try src.expect(try p(&nb[0], i, "self_attn.q_proj.weight"), sq), try src.expect(try p(&nb[1], i, "self_attn.k_proj.weight"), sq), try src.expect(try p(&nb[2], i, "self_attn.v_proj.weight"), sq) };
            l.qkv = try joined(gpa, up, &qkv);
            const qkv_b = [3][]const u8{ try src.expect(try p(&nb[0], i, "self_attn.q_proj.bias"), v), try src.expect(try p(&nb[1], i, "self_attn.k_proj.bias"), v), try src.expect(try p(&nb[2], i, "self_attn.v_proj.bias"), v) };
            l.qkv_b = try joined(gpa, up, &qkv_b);
            const one = .{
                .{ &l.out, "self_attn.out_proj.weight", sq },       .{ &l.out_b, "self_attn.out_proj.bias", v },
                .{ &l.ln1, "self_attn_layer_norm.weight", v },      .{ &l.ln1_b, "self_attn_layer_norm.bias", v },
                .{ &l.ln2, "final_layer_norm.weight", v },          .{ &l.ln2_b, "final_layer_norm.bias", v },
                .{ &l.fc1, "fc1.weight", &[_]usize{ a.ffn, a.d } }, .{ &l.fc1_b, "fc1.bias", &[_]usize{a.ffn} },
                .{ &l.fc2, "fc2.weight", &[_]usize{ a.d, a.ffn } }, .{ &l.fc2_b, "fc2.bias", v },
            };
            inline for (one) |x| x[0].* = try up.put(try src.expect(try p(&nb[3], i, x[1]), x[2]));
        }
        w.ln_post = try up.put(try src.expect(t ++ "ln_post.weight", &.{a.d}));
        w.ln_post_b = try up.put(try src.expect(t ++ "ln_post.bias", &.{a.d}));
        w.proj1 = try up.put(try src.expect(t ++ "proj1.weight", &.{ a.d, a.d }));
        w.proj1_b = try up.put(try src.expect(t ++ "proj1.bias", &.{a.d}));
        w.proj2 = try up.put(try src.expect(t ++ "proj2.weight", &.{ a.out, a.d }));
        w.proj2_b = try up.put(try src.expect(t ++ "proj2.bias", &.{a.out}));
        return w;
    }
};

/// Tensors put as one: their rows back to back.
pub fn joined(gpa: std.mem.Allocator, up: anytype, parts: []const []const u8) !u64 {
    const all = try std.mem.concat(gpa, u8, parts);
    defer gpa.free(all);
    return up.put(all);
}

/// The 13-row sinusoid table at torch's points, bf16, tiled over `group` chunks.
fn positions(gpa: std.mem.Allocator, a: cfg.Audio, up: anytype, group: usize) !u64 {
    const crow = chunks.rows(a.chunkFrames());
    const table = try tower_ref.positions(gpa, .torch, crow, a.d);
    defer gpa.free(table);
    const tiled = try gpa.alloc(u16, group * table.len);
    defer gpa.free(tiled);
    for (0..group) |g| for (table, 0..) |v, i| {
        tiled[g * table.len + i] = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(v)))) >> 16);
    };
    return up.put(std.mem.sliceAsBytes(tiled));
}

/// f32 to bf16 bits, nearest even (torch's cast of the features to the model's dtype).
pub fn bf16Bits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// The schedule's bf16 buffers, each from the backend's `al.alloc(bytes)`.
pub const Buffers = struct {
    in: u64,
    c1: u64,
    c2: u64,
    x: u64,
    h: u64,
    qkv: u64,
    a: u64,
    wide: u64,
    max_rows: usize,

    /// For clips of up to `max_rows` rows, `group` chunks a convolution pass.
    pub fn alloc(a: cfg.Audio, group: usize, max_rows: usize, al: anytype) !Buffers {
        const cf = a.chunkFrames();
        const p1 = group * ((a.mels - 1) / 2 + 1) * ((cf - 1) / 2 + 1);
        const rows = max_rows + chunks.rows(cf);
        return .{
            .in = try al.alloc(2 * group * a.mels * cf),
            .c1 = try al.alloc(2 * p1 * a.conv),
            .c2 = try al.alloc(2 * (p1 / 4 + 1) * a.conv),
            .x = try al.alloc(2 * rows * a.d),
            .h = try al.alloc(2 * rows * @max(a.d, a.out)),
            .qkv = try al.alloc(2 * rows * 3 * a.d),
            .a = try al.alloc(2 * rows * a.d),
            .wide = try al.alloc(2 * @max(rows * a.ffn, group * chunks.rows(cf) * a.conv * a.convBins())), // also conv_out's input
            .max_rows = max_rows,
        };
    }
};

/// The bf16 input of chunks [first, first + n) of `feats`: [n][mels][cf], zero past the clip.
fn input(a: cfg.Audio, feats: mel.Features, cf: usize, first: usize, n: usize, host: []u16) []u16 {
    const out = host[0 .. n * a.mels * cf];
    @memset(out, 0);
    for (0..n) |i| for (0..a.mels) |m| for (0..cf) |j| {
        const fr = (first + i) * cf + j;
        if (fr < feats.frames) out[(i * a.mels + m) * cf + j] = bf16Bits(feats.at(m, fr));
    };
    return out;
}

/// The convolutions over a group of chunks into rows of b.x from first * 13 on, positions added; each chunk is
/// convolved at the plan's chunk length (a lone short chunk at its own, as the package pads to the longest chunk).
fn front(r: anytype, a: cfg.Audio, w: *const Weights, b: Buffers, feats: mel.Features, cf: usize, first: usize, n: usize, host: []u16) !void {
    try r.upload(b.in, std.mem.sliceAsBytes(input(a, feats, cf, first, n, host)));
    const f1 = (a.mels - 1) / 2 + 1;
    const t1 = (cf - 1) / 2 + 1;
    const f2 = (f1 - 1) / 2 + 1;
    const t2 = (t1 - 1) / 2 + 1;
    const f3 = (f2 - 1) / 2 + 1;
    const t3 = (t2 - 1) / 2 + 1;
    // each conv: bf16(taps . w^T + b), dumped as the package's hook sees it, then the GELU in place
    try r.conv(w.conv[0], w.conv_b[0], b.in, b.c1, n, a.mels, cf, 1, a.conv);
    try r.dump("conv1", b.c1, n * f1 * t1, a.conv, .{ n, f1, t1 });
    try r.gelu(b.c1, b.c1, n * f1 * t1 * a.conv);
    try r.conv(w.conv[1], w.conv_b[1], b.c1, b.c2, n, f1, t1, a.conv, a.conv);
    try r.dump("conv2", b.c2, n * f2 * t2, a.conv, .{ n, f2, t2 });
    try r.gelu(b.c2, b.c2, n * f2 * t2 * a.conv);
    try r.conv(w.conv[2], w.conv_b[2], b.c2, b.c1, n, f2, t2, a.conv, a.conv);
    try r.dump("conv3", b.c1, n * f3 * t3, a.conv, .{ n, f3, t3 });
    try r.gelu(b.c1, b.c1, n * f3 * t3 * a.conv);
    try r.flatten(b.c1, b.wide, n, f3, t3, a.conv);
    try r.linear(b.wide, w.conv_out, null, b.x + first * t3 * a.d * 2, n * t3, a.d, a.conv * f3, .none, w.positions);
}

/// The tower over `feats` into `out` [rows, a.out] bf16; the plan's rows returned.
pub fn encode(r: anytype, gpa: std.mem.Allocator, a: cfg.Audio, w: *const Weights, b: Buffers, feats: mel.Features, out: u64) !usize {
    const plan = try chunks.Plan.init(a, feats.frames);
    if (plan.rows > b.max_rows) return error.AudioTooLong;
    const host = try gpa.alloc(u16, w.group * a.mels * a.chunkFrames());
    defer gpa.free(host);
    var first: usize = 0;
    while (first < plan.chunks) : (first += w.group) {
        try front(r, a, w, b, feats, plan.chunk_frames, first, @min(w.group, plan.chunks - first), host);
        try r.sync(); // the host input is refilled for the next group
    }
    const rows = plan.rows;
    const d = a.d;
    try r.dump("embed", b.x, rows, d, null);
    var name: [16]u8 = undefined;
    for (w.layers, 0..) |l, i| {
        try r.layernorm(b.x, l.ln1, l.ln1_b, b.h, rows, d);
        try r.linear(b.h, l.qkv, l.qkv_b, b.qkv, rows, 3 * d, d, .none, null);
        for (0..plan.windows()) |wi| {
            const win = plan.window(wi);
            try r.attention(b.qkv + win.start * 3 * d * 2, b.a + win.start * d * 2, win.len, a.heads, a.headDim());
        }
        try r.linear(b.a, l.out, l.out_b, b.x, rows, d, d, .none, b.x);
        try r.layernorm(b.x, l.ln2, l.ln2_b, b.h, rows, d);
        try r.linear(b.h, l.fc1, l.fc1_b, b.wide, rows, a.ffn, d, .gelu, null);
        try r.linear(b.wide, l.fc2, l.fc2_b, b.x, rows, d, a.ffn, .none, b.x);
        try r.dump(try std.fmt.bufPrint(&name, "layer{d}", .{i}), b.x, rows, d, null);
    }
    try r.layernorm(b.x, w.ln_post, w.ln_post_b, b.h, rows, d);
    try r.dump("ln_post", b.h, rows, d, null);
    try r.linear(b.h, w.proj1, w.proj1_b, b.a, rows, d, d, .none, null);
    try r.dump("proj1", b.a, rows, d, null);
    try r.gelu(b.a, b.h, rows * d);
    try r.linear(b.h, w.proj2, w.proj2_b, out, rows, a.out, d, .none, null);
    try r.dump("audio_rows", out, rows, a.out, null);
    return rows;
}

test "bf16Bits: nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Bits(1.0 + 1.0 / 256.0));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16Bits(1.0 + 3.0 / 256.0));
}
