//! The RADIO tower and the omni projector on CUDA: an image's patches to its rows in the language model's width.
const std = @import("std");
const cuda = @import("cuda");
const Image = @import("lanes").stream.Image;
const Config = @import("config.zig").Config;
const Weights = @import("weights.zig").Weights;
const rows_ = @import("rows.zig");
const Allocator = std.mem.Allocator;
const vit = cuda.vit;

pub const Tower = struct {
    gpa: Allocator,
    c: Config,
    w: Weights,
    k: vit.Kernels,
    at: vit.Attention,
    g: vit.Gemms,
    norms: cuda.norms.Norms,
    scratch: cuda.DeviceBuffer,
    b: Buffers,
    bytes: u64, // on the device: weights, scratch and the Linears' workspace

    const Buffers = struct { px: u64, h: u64, x: u64, wide: u64, a: u64, acc: u64 };

    /// The tower from `dir` (its weights as stored) and scratch for the largest image the processor makes.
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config) !Tower {
        if (c.hidden > vit.norm_most or c.dim() * c.heads != c.hidden) return error.VisionShape;
        var w = try Weights.load(gpa, io, d, dir, &c);
        errdefer w.deinit(gpa);
        var k = try vit.Kernels.load(d);
        errdefer k.unload();
        var at = try vit.Attention.load(d);
        errdefer at.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        var norms = try cuda.norms.Norms.load(d);
        errdefer norms.unload();
        const n: usize = c.max_patches;
        const rows = n + c.prefix;
        const hd = c.hidden;
        const wide = @max(3 * hd, c.ffn, c.projector_hidden / (Config.shuffle * Config.shuffle));
        // bf16: patches, the residual, normed and Linear outputs, wide outputs, attention; f32: products, float patches
        const sizes = [_]usize{ 2 * n * c.patchWidth(), 2 * rows * hd, 2 * rows * hd, 2 * rows * wide, 2 * rows * hd, 4 * rows * @max(wide, c.patchWidth()) };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var ptrs: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &ptrs) |sz, *p| {
            p.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        return .{
            .gpa = gpa,
            .c = c,
            .w = w,
            .k = k,
            .at = at,
            .g = g,
            .norms = norms,
            .scratch = scratch,
            .b = .{ .px = ptrs[0], .h = ptrs[1], .x = ptrs[2], .wide = ptrs[3], .a = ptrs[4], .acc = ptrs[5] },
            .bytes = w.bytes + total + g.workspace.len,
        };
    }

    pub fn deinit(t: *Tower) void {
        t.scratch.free();
        t.norms.unload();
        t.g.deinit();
        t.at.unload();
        t.k.unload();
        t.w.deinit(t.gpa);
    }

    /// Each span's image through the tower and projector into its rows of `out` (bf16 [tokens, text hidden]).
    pub fn encode(t: *Tower, s: cuda.Stream, images: []const Image, spans: []const rows_.Span, out: *const rows_.Rows) !void {
        if (out.hidden != t.c.text_hidden) return error.RowShape;
        for (spans) |sp| {
            const img = images[sp.image];
            if (img.tokens != sp.n or sp.row + sp.n > out.limit) return error.ImageRows;
            try t.picture(s, img, out.at(sp));
        }
    }

    /// One image's rows at `out`: the tower's features, then the projector.
    pub fn picture(t: *Tower, s: cuda.Stream, img: Image, out: u64) !void {
        try t.project(s, try t.features(s, img), img.gh, img.gw, out);
    }

    /// The tower: an image's patch rows after its last norm, bf16 [gh * gw, hidden] in scratch until the next call.
    pub fn features(t: *Tower, s: cuda.Stream, img: Image) !u64 {
        const c = t.c;
        const b = t.b;
        const k = &t.k;
        const g = &t.g;
        const w = &t.w;
        const gh: usize = img.gh;
        const gw: usize = img.gw;
        const n = std.math.mul(usize, gh, gw) catch return error.ImageShape;
        if (n == 0 or n > c.max_patches or gh % 2 != 0 or gw % 2 != 0 or n / 4 != img.tokens or c.patch == 0) return error.ImageShape;
        if (c.video_temporal != 0 and c.video_temporal != 2) return error.ImageShape;
        const image_width = std.math.mul(usize, 3, std.math.mul(usize, c.patch, c.patch) catch return error.ImageShape) catch return error.ImageShape;
        const video_width = std.math.mul(usize, c.video_temporal, image_width) catch return error.ImageShape;
        const video_values = std.math.mul(usize, n, video_width) catch return error.ImageShape;
        const video = c.video_temporal == 2 and img.pixels.len == video_values;
        const pw = if (video) video_width else image_width;
        const values = std.math.mul(usize, n, pw) catch return error.ImageShape;
        const room = std.math.mul(usize, c.max_patches, image_width) catch return error.ImageShape;
        if (img.pixels.len != values or values > room) return error.ImageShape;
        const hd = c.hidden;
        // the float patches land in the f32 products' room (the patch Linear has no bias), cast to bf16 on the GPU
        try t.scratch.uploadAsync(b.acc - t.scratch.ptr, std.mem.sliceAsBytes(img.pixels), s.handle);
        try k.cast(s, b.acc, b.px, n * pw);
        // the cls and register rows, then each patch's projection plus its position row
        const rows = n + c.prefix;
        const feats = b.h + 2 * c.prefix * hd;
        try t.scratch.copyFrom(b.h - t.scratch.ptr, w.cls, 2 * c.prefix * hd, s.handle);
        try g.run(s, b.px, if (video) w.video_patch else w.patch, b.x, n, hd, pw, .bf16);
        try k.positionAdd(s, b.x, w.table, feats, gh, gw, c.side, hd);
        const dim = c.dim();
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
        for (w.layers) |l| {
            try k.layerNorm(s, b.h, l.norm1, l.norm1_bias, b.x, rows, hd, c.eps);
            try vit.linear(g, k, s, b.acc, b.x, l.qkv, l.qkv_bias, b.wide, rows, 3 * hd, hd);
            try t.at.full(s, b.wide, b.a, rows, c.heads, dim, scale);
            try vit.linear(g, k, s, b.acc, b.a, l.proj, l.proj_bias, b.x, rows, hd, hd);
            try k.pointwise(s, b.x, b.h, b.h, rows * hd, .none);
            try k.layerNorm(s, b.h, l.norm2, l.norm2_bias, b.x, rows, hd, c.eps);
            try vit.linear(g, k, s, b.acc, b.x, l.fc1, l.fc1_bias, b.wide, rows, c.ffn, hd);
            try k.pointwise(s, b.wide, null, b.wide, rows * c.ffn, .gelu);
            try vit.linear(g, k, s, b.acc, b.wide, l.fc2, l.fc2_bias, b.x, rows, hd, c.ffn);
            try k.pointwise(s, b.x, b.h, b.h, rows * hd, .none);
        }
        if (!c.final_norm) return feats; // the patch rows: cls and registers dropped
        try k.layerNorm(s, feats, w.final, w.final_bias, b.x, n, hd, c.eps);
        return b.x;
    }

    /// The projector: features [gh * gw, hidden] shuffled 2 x 2, RMSNorm, then the relu^2 MLP into `out`.
    pub fn project(t: *Tower, s: cuda.Stream, feats: u64, gh: usize, gw: usize, out: u64) !void {
        const c = t.c;
        const b = t.b;
        const tokens = gh * gw / 4;
        const m = 4 * c.hidden;
        try t.k.unshuffle(s, feats, b.a, gh, gw, c.hidden);
        try t.norms.rms(s, b.a, .bf16, t.w.rms, b.x, .bf16, tokens, m, Config.rms_eps);
        try t.g.run(s, b.x, t.w.lin1, b.wide, tokens, c.projector_hidden, m, .bf16);
        try t.k.pointwise(s, b.wide, null, b.wide, tokens * c.projector_hidden, .relu2);
        try t.g.run(s, b.wide, t.w.lin2, out, tokens, c.text_hidden, c.projector_hidden, .bf16);
    }
};
