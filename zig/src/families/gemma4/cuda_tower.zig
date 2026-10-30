//! Gemma 4 31B's image tower on CUDA (#1048 step 12): 16-px patches through 27 encoder layers, pooling, projection.
const std = @import("std");
const cuda = @import("cuda");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const image = @import("image.zig");

const vit = cuda.vit;
const W = wts.tower_width;
const H = wts.tower_heads;
const D = wts.tower_head_dim;
const eps: f32 = 1e-6;
const rope_theta: f32 = 100;
/// Patches an image has at most: the largest budget's tokens, each a 3 x 3 block.
pub const max_patches = image.max_budget * image.pool * image.pool;

pub const Tower = struct {
    gpa: std.mem.Allocator,
    w: *const wts.Tower,
    k: *const kern.Kernels,
    vit: vit.Kernels,
    attn: vit.Attention,
    gemms: vit.Gemms,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    s: Scratch = undefined,

    const Scratch = struct { patches: u64, x: [2]u64, xn: u64, hs: u64, qkv: u64, att: u64, o: u64, gate: u64, up: u64, act: u64, e: [2]u64, ids: u64, cos: u64, sin: u64, pooled: u64, pn: u64, acc: u64 };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Tower, k: *const kern.Kernels) !Tower {
        var t: Tower = .{ .gpa = gpa, .w = w, .k = k, .vit = try vit.Kernels.load(d), .attn = undefined, .gemms = undefined };
        errdefer t.vit.unload();
        t.attn = try vit.Attention.load(d);
        errdefer t.attn.unload();
        t.gemms = try vit.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        errdefer t.buffers.deinit(gpa);
        errdefer for (t.buffers.items) |*b| b.free();
        const p = max_patches;
        const n = image.max_budget;
        t.s = .{
            .patches = try t.alloc(d, p * wts.patch_width * 2),
            .x = .{ try t.alloc(d, p * W * 2), try t.alloc(d, p * W * 2) },
            .xn = try t.alloc(d, p * W * 2),
            .hs = try t.alloc(d, p * W * 2),
            .qkv = try t.alloc(d, p * 3 * W * 2),
            .att = try t.alloc(d, p * W * 2),
            .o = try t.alloc(d, p * W * 2),
            .gate = try t.alloc(d, p * wts.tower_inter * 2),
            .up = try t.alloc(d, p * wts.tower_inter * 2),
            .act = try t.alloc(d, p * wts.tower_inter * 2),
            .e = .{ try t.alloc(d, p * W * 2), try t.alloc(d, p * W * 2) },
            .ids = try t.alloc(d, 2 * p * 4),
            .cos = try t.alloc(d, p * D * 2),
            .sin = try t.alloc(d, p * D * 2),
            .pooled = try t.alloc(d, n * W * 2),
            .pn = try t.alloc(d, n * W * 2),
            .acc = try t.alloc(d, p * W * 4),
        };
        return t;
    }

    /// The device bytes init allocates for its scratch.
    pub fn workspaceBytes() u64 {
        const p: u64 = max_patches;
        const n: u64 = image.max_budget;
        return p * wts.patch_width * 2 + 11 * p * W * 2 + 3 * p * wts.tower_inter * 2 + 2 * p * 4 + 2 * p * D * 2 + 2 * n * W * 2 + p * W * 4;
    }

    fn alloc(t: *Tower, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try t.buffers.append(t.gpa, b);
        return b.ptr;
    }

    pub fn deinit(t: *Tower) void {
        for (t.buffers.items) |*b| b.free();
        t.buffers.deinit(t.gpa);
        t.gemms.deinit();
        t.attn.unload();
        t.vit.unload();
    }

    /// Patches (image.prepare's, 16 px) to `out` [gh gw / 9, hidden] bf16, queued on `s` (the host rows copied first).
    pub fn encode(t: *Tower, d: *const cuda.Driver, s: cuda.Stream, pixels: []const f32, gh: u32, gw: u32, out: u64) !void {
        const p: usize = @as(usize, gh) * gw;
        if (gh == 0 or gw == 0 or gh % image.pool != 0 or gw % image.pool != 0 or p > max_patches) return error.GemmaImageGrid;
        if (pixels.len != p * wts.patch_width or gw > wts.tower_table_rows or gh > wts.tower_table_rows) return error.GemmaImageGrid;
        const x = t.s;
        // host rows: bf16 2 (p - 0.5), the RoPE's cos and sin, each patch's two position rows
        const host = try t.gpa.alloc(u16, p * wts.patch_width + 2 * p * D);
        defer t.gpa.free(host);
        for (pixels, host[0 .. p * wts.patch_width]) |v, *h| h.* = vit.bf16(2 * (v - 0.5));
        rope(host[p * wts.patch_width ..][0 .. p * D], host[p * wts.patch_width + p * D ..][0 .. p * D], gw, p);
        const ids = try t.gpa.alloc(i32, 2 * p);
        defer t.gpa.free(ids);
        for (0..p) |i| {
            ids[i] = @intCast(i % gw);
            ids[p + i] = @intCast(wts.tower_table_rows + i / gw);
        }
        const up = struct {
            fn to(dv: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
                try (cuda.DeviceBuffer{ .d = dv, .ptr = ptr, .len = bytes.len, .borrowed = true }).upload(0, bytes);
            }
        }.to;
        try up(d, x.patches, std.mem.sliceAsBytes(host[0 .. p * wts.patch_width]));
        try up(d, x.cos, std.mem.sliceAsBytes(host[p * wts.patch_width ..][0 .. p * D]));
        try up(d, x.sin, std.mem.sliceAsBytes(host[p * wts.patch_width + p * D ..][0 .. p * D]));
        try up(d, x.ids, std.mem.sliceAsBytes(ids));
        const w = t.w;
        try t.gemms.run(s, x.patches, w.input_proj, x.x[0], p, W, wts.patch_width, .bf16);
        try t.k.embed.bf16(s, x.ids, w.table, x.e[0], p, W);
        try t.k.embed.bf16(s, x.ids + p * 4, w.table, x.e[1], p, W);
        try t.vit.pointwise(s, x.e[0], x.e[1], x.e[1], p * W, .none);
        try t.vit.pointwise(s, x.x[0], x.e[1], x.x[0], p * W, .none);
        try t.k.norms.addRmsH(s, x.x[0], null, w.layers[0].in_norm, .bf16, x.hs, x.xn, null, p, W, eps);
        var cur: usize = 0;
        for (&w.layers, 0..) |*l, i| {
            try t.gemms.run(s, x.xn, l.qkv, x.qkv, p, 3 * W, W, .bf16);
            try t.vit.g4Qkv(s, x.qkv, l.q_norm, l.k_norm, x.cos, x.sin, p, H, D, eps);
            try t.attn.full(s, x.qkv, x.att, p, H, D, 1.0);
            try t.gemms.run(s, x.att, l.o, x.o, p, W, W, .bf16);
            // x + bf16(rms(o) post_attn), then its rms by pre_ff
            try t.k.norms.addRmsTail(s, x.o, x.x[cur], l.pre_ff, x.x[1 - cur], x.xn, null, p, W, eps, .{ .branch = l.post_attn });
            cur = 1 - cur;
            try t.gemms.run(s, x.xn, l.gate, x.gate, p, wts.tower_inter, W, .bf16);
            try t.gemms.run(s, x.xn, l.up, x.up, p, wts.tower_inter, W, .bf16);
            try t.vit.g4GeluMul(s, x.gate, x.up, x.act, p * wts.tower_inter);
            try t.gemms.run(s, x.act, l.down, x.o, p, W, wts.tower_inter, .bf16);
            const next = if (i + 1 < w.layers.len) w.layers[i + 1].in_norm else l.in_norm; // the last one's is unread
            try t.k.norms.addRmsTail(s, x.o, x.x[cur], next, x.x[1 - cur], x.xn, null, p, W, eps, .{ .branch = l.post_ff });
            cur = 1 - cur;
        }
        const n = p / (image.pool * image.pool);
        try t.vit.g4Pool(s, x.x[cur], w.std_bias, w.std_scale, x.pooled, gh, gw, W);
        try t.k.norms.rms(s, x.pooled, .bf16, null, x.pn, .bf16, n, W, eps);
        switch (w.proj) {
            .a4 => |q| try t.k.affine.prompt(s, x.pn, q, out, n),
            .an => |q| try t.k.wide.prompt(s, x.pn, q, out, n),
        }
    }
};

/// Gemma4VisionRotaryEmbedding gw wide (Metal's vision.rope): dims 0-35 by x, 36-71 by y, bf16 cos and sin.
fn rope(cs: []u16, sn: []u16, gw: usize, p: usize) void {
    const half = D / 2;
    var inv: [half / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, half));
    for (0..p) |q| for (0..D) |dd| {
        const axis: f32 = @floatFromInt(if (dd < half) q % gw else q / gw);
        const angle = axis * inv[(dd % half) % (half / 2)];
        cs[q * D + dd] = vit.bf16(@cos(angle));
        sn[q * D + dd] = vit.bf16(@sin(angle));
    };
}

test "the RoPE rows: position 0 is cos 1, sin 0; the x half follows the column, the y half the row" {
    var cs: [2 * 3 * D]u16 = undefined;
    var sn: [2 * 3 * D]u16 = undefined;
    rope(&cs, &sn, 3, 6);
    try std.testing.expectEqual(vit.bf16(1), cs[0]);
    try std.testing.expectEqual(vit.bf16(0), sn[D - 1]);
    try std.testing.expectEqual(cs[1 * D + 0], cs[4 * D + 0]); // patches 1 and 4 share the column
    try std.testing.expectEqual(cs[3 * D + D / 2], cs[4 * D + D / 2]); // patches 3 and 4 share the row
}
