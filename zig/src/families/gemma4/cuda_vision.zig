//! Gemma 4 12B's images on CUDA (#1048 step 12): the processor, and the embedder at torch's per-op bf16 rounding.
const std = @import("std");
const cuda = @import("cuda");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const image = @import("cuda_image.zig");
const tw = @import("cuda_tower.zig");

const vit = cuda.vit;

/// nn.LayerNorm's default and Gemma4RMSNorm's.
const ln_eps: f32 = 1e-5;
const rms_eps: f32 = 1e-6;

/// --vision's limits: tokens an image takes at most and images a request.
pub const Limits = struct { image_tokens: u32, max_images: u32 };

/// What the server's image path reads: the processor, the image token and the markers it wraps an image's rows in.
pub const Vision = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    marks: struct { open: u32, image: u32, close: u32 } = .{ .open = 255999, .image = 258880, .close = 258882 },
    mark_ids: [3]u32 = .{ 255999, 258880, 258882 },
    limits: Limits,
    /// The 12B's embedder on merged 48-px patches, or the 31B's tower on 16-px ones.
    e: union(enum) { embedder: Embedder, tower: tw.Tower },

    pub fn prepare(v: *const Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !image.Prepared {
        return image.prepare(v.io, a, bytes, @min(max_tokens, v.limits.image_tokens), if (v.e == .tower) .patches else .merged);
    }

    /// An image's prepared rows to `out` [tokens, hidden] bf16, queued on `s`.
    pub fn encode(v: *Vision, d: *const cuda.Driver, s: cuda.Stream, pixels: []const f32, gh: u32, gw: u32, out: u64) !void {
        return switch (v.e) {
            inline else => |*e| e.encode(d, s, pixels, gh, gw, out),
        };
    }

    pub fn deinit(v: *Vision) void {
        switch (v.e) {
            inline else => |*e| e.deinit(),
        }
    }

    pub const explain = image.decoderProblem;
    /// Gemma 4's templates write an image for parts of type "image" (as the Metal family offers them).
    pub const part_type = "image";
};

/// The embedder's kernels, cuBLASLt plans and scratch for an image of up to image.max_budget rows.
pub const Embedder = struct {
    w: *const wts.Vision,
    k: *const kern.Kernels,
    vit: vit.Kernels,
    gemms: vit.Gemms,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,
    // x: [T, merged] patches and patch_ln1's; y: [T, width] rows of each step; acc: f32 rows for an unfused bias
    x: [2]u64 = undefined,
    y: [6]u64 = undefined,
    acc: u64 = 0,
    ids: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Vision, k: *const kern.Kernels) !Embedder {
        var e: Embedder = .{ .w = w, .k = k, .vit = try vit.Kernels.load(d), .gemms = undefined, .gpa = gpa };
        errdefer e.vit.unload();
        e.gemms = try vit.Gemms.init(gpa, d);
        errdefer e.gemms.deinit();
        errdefer e.buffers.deinit(gpa);
        errdefer for (e.buffers.items) |*b| b.free();
        const t = image.max_budget;
        for (&e.x) |*p| p.* = try e.alloc(d, t * wts.merged_width * 2);
        for (&e.y) |*p| p.* = try e.alloc(d, t * wts.vision_width * 2);
        e.acc = try e.alloc(d, t * wts.vision_width * 4);
        e.ids = try e.alloc(d, 2 * t * 4);
        return e;
    }

    /// The device bytes init allocates for its scratch.
    pub fn workspaceBytes() u64 {
        const t: u64 = image.max_budget;
        return 2 * t * wts.merged_width * 2 + 6 * t * wts.vision_width * 2 + t * wts.vision_width * 4 + 2 * t * 4;
    }

    fn alloc(e: *Embedder, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try e.buffers.append(e.gpa, b);
        return b.ptr;
    }

    pub fn deinit(e: *Embedder) void {
        for (e.buffers.items) |*b| b.free();
        e.buffers.deinit(e.gpa);
        e.gemms.deinit();
        e.vit.unload();
    }

    /// Merged patches (image.prepare's) to `out` [gh gw / 9, hidden] bf16, queued on `s` (the pixels copied first).
    pub fn encode(e: *Embedder, d: *const cuda.Driver, s: cuda.Stream, pixels: []const f32, gh: u32, gw: u32, out: u64) !void {
        if (gh == 0 or gw == 0 or gh % image.pool != 0 or gw % image.pool != 0) return error.GemmaImageGrid;
        const t: usize = gh * gw / (image.pool * image.pool);
        const cols: usize = gw / image.pool;
        if (t > image.max_budget or pixels.len != t * wts.merged_width) return error.GemmaImageGrid;
        if (cols > wts.table_rows or gh / image.pool > wts.table_rows) return error.GemmaImageGrid;
        const mw = wts.merged_width;
        const vw = wts.vision_width;
        const host = try e.gpa.alloc(u16, t * mw + 2 * t * 2);
        defer e.gpa.free(host);
        for (pixels, host[0 .. t * mw]) |p, *h| h.* = vit.bf16(p);
        // each row's x and y position rows of the [1120, 2, width] table, as rows of a [2240, width] one
        const ids = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(host[t * mw ..]));
        for (0..t) |r| {
            ids[r] = @intCast(2 * (r % cols));
            ids[t + r] = @intCast(2 * (r / cols) + 1);
        }
        try (cuda.DeviceBuffer{ .d = d, .ptr = e.x[0], .len = t * mw * 2, .borrowed = true }).upload(0, std.mem.sliceAsBytes(host[0 .. t * mw]));
        try (cuda.DeviceBuffer{ .d = d, .ptr = e.ids, .len = 2 * t * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(ids));
        const w = e.w;
        try e.vit.layerNormWide(s, e.x[0], w.ln1_w, w.ln1_b, e.x[1], t, mw, ln_eps);
        try vit.linear(&e.gemms, &e.vit, s, e.acc, e.x[1], w.dense_w, w.dense_b, e.y[0], t, vw, mw); // bf16(x w^T + b)
        try e.vit.layerNormWide(s, e.y[0], w.ln2_w, w.ln2_b, e.y[1], t, vw, ln_eps);
        // positions: bf16(x + bf16(table[px][0] + table[py][1])), then pos_norm
        try e.k.embed.bf16(s, e.ids, w.table, e.y[2], t, vw);
        try e.k.embed.bf16(s, e.ids + t * 4, w.table, e.y[3], t, vw);
        try e.vit.pointwise(s, e.y[2], e.y[3], e.y[3], t * vw, .none);
        try e.vit.pointwise(s, e.y[1], e.y[3], e.y[3], t * vw, .none);
        try e.vit.layerNormWide(s, e.y[3], w.pos_w, w.pos_b, e.y[4], t, vw, ln_eps);
        try e.k.norms.rms(s, e.y[4], .bf16, null, e.y[5], .bf16, t, vw, rms_eps);
        switch (w.proj) {
            .q => |p| switch (p) {
                .a4 => |q| try e.k.affine.prompt(s, e.y[5], q, out, t),
                .an => |q| try e.k.wide.prompt(s, e.y[5], q, out, t),
            },
            .bf16 => |m| try e.gemms.run(s, e.y[5], m, out, t, vw, vw, .bf16),
        }
    }
};

test "the embedder's sizes: 3 x 3 merged 16-px patches, the 12B's width" {
    try std.testing.expectEqual(@as(usize, wts.merged_width), image.merged_width);
    try std.testing.expectEqual(@as(u32, 280), image.max_budget);
}
