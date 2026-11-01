//! Idol's image encoder on CUDA: FLUX.2's VAE encoder in fp32, a 1024 x 1024 picture to the condition's 4,101 rows.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("vae_config.zig");
const ks = @import("cuda_vae_kernels.zig");
const wts = @import("cuda_vae_weights.zig");
const gm = @import("cuda_gemm.zig");
const model = @import("cuda_model.zig");

pub const size = cfg.size;
pub const latent = cfg.latent;
pub const pad = cfg.pad;
pub const rows = cfg.rows;
pub const channels = cfg.channels;
pub const Stages = cfg.Stages;
pub const Verdict = cfg.Verdict;
pub const gate = cfg.gate;

pub const Timing = struct { wall: f64, gpu: f64 };

/// The widest picture tensor: the first stage's 1024 x 1024 x 128.
const most = size * size * cfg.widths[0];
/// The mid block's positions, and the query rows a softmax pass takes (its scores fill one picture tensor at most).
const mid = latent * latent;
pub const query_chunk = 4096;

/// The scratch's parts in bytes, as Encoder's views: picture, noise, four picture tensors, moments, rows, GroupNorm's.
const parts = [_]usize{ size * size * 3 * 4, mid * cfg.latent_channels * 4, most * 4, most * 4, most * 4, most * 4, mid * 2 * cfg.latent_channels * 4, rows * channels * 4, 2 * ks.groups * 4, ks.groups * ks.chunks(size * size) * 4 };

pub fn scratchBytes() usize {
    var total: usize = 0;
    for (parts) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: ks.Kernels,
    gemms: gm.Gemms,
    w: wts.Weights,
    scratch: cuda.DeviceBuffer,
    marks: [2]cuda.Event,
    image: u64 = 0,
    noise: u64 = 0,
    t: [4]u64 = @splat(0),
    moments: u64 = 0,
    out: u64 = 0,
    stats: u64 = 0,
    part: u64 = 0,

    /// The vae/ folder `dir` (FLUX.2-klein-4B's, or FLUX.2-dev's rounded to bf16) on `ctx`'s GPU, if admitted.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Encoder {
        const sm = try ctx.capability();
        if (!model.admitted(sm)) {
            var buf: [256]u8 = undefined;
            const name = ctx.name(&buf) catch "this GPU";
            std.log.err("idol vae: {s} (sm_{d}) is not a GPU this family admits; it runs on sm_120 and sm_121 alone", .{ name, sm });
            return error.UncheckedGpu;
        }
        const d = ctx.d;
        const v = try gpa.create(Encoder);
        errdefer gpa.destroy(v);
        v.* = .{ .gpa = gpa, .d = d, .stream = undefined, .k = undefined, .gemms = undefined, .w = undefined, .scratch = undefined, .marks = undefined };
        v.stream = try cuda.Stream.init(d, false);
        errdefer v.stream.deinit();
        v.k = try ks.Kernels.load(d);
        errdefer v.k.unload();
        v.gemms = try gm.Gemms.init(gpa, d);
        errdefer v.gemms.deinit();
        v.w = try wts.load(gpa, io, d, dir);
        errdefer v.w.deinit();
        v.scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes());
        errdefer v.scratch.free();
        v.marks[0] = try cuda.Event.init(d, true);
        errdefer v.marks[0].deinit();
        v.marks[1] = try cuda.Event.init(d, true);
        errdefer v.marks[1].deinit();
        var at: u64 = v.scratch.ptr;
        const views = [_]*u64{ &v.image, &v.noise, &v.t[0], &v.t[1], &v.t[2], &v.t[3], &v.moments, &v.out, &v.stats, &v.part };
        for (views, parts) |view, n| {
            view.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        return v;
    }

    pub fn deinit(v: *Encoder) void {
        for (&v.marks) |*m| m.deinit();
        v.scratch.free();
        v.w.deinit();
        v.gemms.deinit();
        v.k.unload();
        v.stream.deinit();
        v.gpa.destroy(v);
    }

    fn upload(v: *const Encoder, at: u64, bytes: []const u8) !void {
        try v.d.check(v.d.api.cuMemcpyHtoDAsync_v2(at, bytes.ptr, bytes.len, v.stream.handle), "cuMemcpyHtoDAsync");
    }

    fn download(v: *const Encoder, out: []u8, at: u64) !void {
        try v.d.check(v.d.api.cuMemcpyDtoHAsync_v2(out.ptr, at, out.len, v.stream.handle), "cuMemcpyDtoHAsync");
        try v.stream.synchronize();
    }

    /// The condition rows [4101, 128] of `image` [1024, 1024, 3] in [0, 1] with the posterior's `noise` [128, 128, 32].
    pub fn encode(v: *Encoder, image: []const f32, noise: []const f32, out: []f32, stages: Stages) !Timing {
        if (image.len != size * size * 3) return error.PictureSize;
        if (noise.len != latent * latent * cfg.latent_channels or out.len != rows * channels) return error.LatentRows;
        const started = model.seconds();
        try v.upload(v.image, std.mem.sliceAsBytes(image));
        try v.upload(v.noise, std.mem.sliceAsBytes(noise));
        try v.marks[0].record(v.stream);
        const s = v.stream;
        var R: usize = size;
        try ks.convIn(&v.k, s, v.image, v.w.conv_in.w, v.w.conv_in.b, v.t[0], R, R);
        var x: usize = 0;
        for (v.w.stages, 0..) |st, i| {
            for (st.resnets) |r| x = try v.resnet(r, x, R);
            if (st.down) |d| {
                const y = (x + 1) % v.t.len;
                try ks.conv(&v.k, s, v.t[x], d.w, d.b, null, v.t[y], ks.down(R, d.cin, d.cout, d.ldw));
                x = y;
                R /= 2;
            }
            if (stages.down[i]) |dst| try v.download(std.mem.sliceAsBytes(dst), v.t[x]);
        }
        x = try v.resnet(v.w.mid[0], x, R);
        x = try v.attention(x, R * R);
        x = try v.resnet(v.w.mid[1], x, R);
        if (stages.mid) |dst| try v.download(std.mem.sliceAsBytes(dst), v.t[x]);
        const n = (x + 1) % v.t.len;
        const o = (x + 2) % v.t.len;
        try v.norm(v.t[x], v.t[n], R * R, cfg.widths[3], v.w.norm_out, true);
        try v.conv(v.w.conv_out, v.t[n], v.t[o], R, null);
        try v.conv(v.w.quant, v.t[o], v.moments, R, null);
        try ks.tokens(&v.k, s, v.moments, v.noise, v.w.bn_mean, v.w.bn_std, v.out, R, R, pad);
        try v.marks[1].record(s);
        try v.download(std.mem.sliceAsBytes(out), v.out);
        if (stages.moments) |dst| try v.download(std.mem.sliceAsBytes(dst), v.moments);
        const gpu = @as(f64, try cuda.Event.elapsedMs(v.marks[0], v.marks[1])) / 1000;
        return .{ .wall = model.seconds() - started, .gpu = gpu };
    }

    /// One residual block on t[x] at R x R, its output in another of t; returns that one's index.
    fn resnet(v: *const Encoder, r: wts.Resnet, x: usize, R: usize) !usize {
        const n = (x + 1) % v.t.len;
        const h = (x + 2) % v.t.len;
        const sc = (x + 3) % v.t.len;
        try v.norm(v.t[x], v.t[n], R * R, r.conv1.cin, r.norm1, true);
        try v.conv(r.conv1, v.t[n], v.t[h], R, null);
        try v.norm(v.t[h], v.t[n], R * R, r.conv1.cout, r.norm2, true);
        const skip = if (r.shortcut) |c| blk: {
            try v.conv(c, v.t[x], v.t[sc], R, null);
            break :blk v.t[sc];
        } else v.t[x];
        try v.conv(r.conv2, v.t[n], v.t[h], R, skip);
        return h;
    }

    /// The mid block's self-attention on t[x] (`n` positions): GroupNorm, q, k and v, one head of 512, out + t[x].
    fn attention(v: *Encoder, x: usize, n: usize) !usize {
        // cuBLASLt in true fp32 (CUBLAS_COMPUTE_32F, fp32 operands); v as W_v xn^T, so P V is cuBLASLt's X W^T too
        const W = cfg.width;
        const s = v.stream;
        const xn = v.t[(x + 1) % v.t.len];
        const qkv = v.t[(x + 2) % v.t.len];
        const scores = v.t[(x + 3) % v.t.len];
        const q = qkv;
        const k = qkv + n * W * 4;
        const vt = qkv + 2 * n * W * 4;
        const att = qkv + 3 * n * W * 4;
        try v.norm(v.t[x], xn, n, W, v.w.attn_norm, false);
        try v.gemms.run(s, .f32, xn, v.w.q.w, q, n, W, W);
        try ks.bias(&v.k, s, q, v.w.q.b, null, q, n, W, false);
        try v.gemms.run(s, .f32, xn, v.w.k.w, k, n, W, W);
        try ks.bias(&v.k, s, k, v.w.k.b, null, k, n, W, false);
        try v.gemms.run(s, .f32, v.w.v.w, xn, vt, W, n, W);
        try ks.bias(&v.k, s, vt, v.w.v.b, null, vt, W, n, true);
        const scale = 1.0 / @sqrt(@as(f32, W));
        var r0: usize = 0;
        while (r0 < n) : (r0 += query_chunk) {
            const m = @min(query_chunk, n - r0);
            try v.gemms.run(s, .f32, q + r0 * W * 4, k, scores, m, n, W);
            try ks.softmax(&v.k, s, scores, m, n, scale);
            try v.gemms.run(s, .f32, scores, vt, att + r0 * W * 4, m, W, n);
        }
        // the output projection's sums into q's rows (q is spent), then its bias and the residual into xn's
        try v.gemms.run(s, .f32, att, v.w.out.w, q, n, W, W);
        try ks.bias(&v.k, s, q, v.w.out.b, v.t[x], xn, n, W, false);
        return (x + 1) % v.t.len;
    }

    /// y = GroupNorm(x) with `nm` (then SiLU when `silu`), x and y [pixels, channels], statistics in two passes.
    fn norm(v: *const Encoder, x: u64, y: u64, n: usize, ch: usize, nm: wts.Norm, silu: bool) !void {
        for ([_]bool{ false, true }) |centered| {
            try ks.groupPart(&v.k, v.stream, x, v.stats, v.part, n, ch, centered);
            try ks.groupFinish(&v.k, v.stream, v.part, v.stats, n, ch, centered, cfg.eps);
        }
        try ks.norm(&v.k, v.stream, x, v.stats, nm.g, nm.b, y, n, ch, silu);
    }

    /// y = conv(x) + b (+ r) at R x R, a same-size 3x3 or a 1x1.
    fn conv(v: *const Encoder, c: wts.Conv, x: u64, y: u64, R: usize, r: ?u64) !void {
        const shape = if (c.taps == 9) ks.same(R, c.cin, c.cout, c.ldw) else ks.one(R, c.cin, c.cout, c.ldw);
        try ks.conv(&v.k, v.stream, x, c.w, c.b, r, y, shape);
    }
};

test "scratch: four picture tensors, and the mid block's attention fits two of them" {
    try std.testing.expect(scratchBytes() < 2400 << 20);
    // q, k, v^T and the attention's output in one picture tensor; a chunk's scores in another
    try std.testing.expect(4 * mid * cfg.width <= most);
    try std.testing.expect(query_chunk * mid <= most);
    try std.testing.expect(mid % query_chunk == 0 and mid <= ks.softmax_max);
    // GroupNorm's chunk sums at the widest stage, and every kernel index as c_int
    try std.testing.expectEqual(@as(usize, 4096), ks.chunks(size * size));
    try std.testing.expect(most < std.math.maxInt(c_int));
}

test "admission: the encoder takes the denoiser's GPUs, sm_120 and sm_121 alone" {
    try std.testing.expect(model.admitted(121) and model.admitted(120));
    for ([_]u32{ 80, 86, 89, 90, 100 }) |sm| try std.testing.expect(!model.admitted(sm));
}
