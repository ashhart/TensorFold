//! Idol's denoiser on CUDA: the image condition refined once, then latent, condition and camera rows each step.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const host = @import("host.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const gm = @import("cuda_gemm.zig");

const H = cfg.hidden;
const F = cfg.ffn;
const L = cfg.joint_rows;
const R = cfg.latent_rows;
const C = cfg.cond_rows;

pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

pub fn seconds() f64 {
    const c = @import("sys").c;
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The GPUs this family admits: GB10 (sm_121) and the RTX 50 / PRO Blackwell cards (sm_120), refused others by name.
pub fn admitted(sm: u32) bool {
    return sm == 120 or sm == 121;
}

/// A refined image condition: the condition refiner's rows [4101, 1024] fp16 on the device.
pub const Condition = struct {
    rows: cuda.DeviceBuffer,

    pub fn deinit(c: *Condition) void {
        c.rows.free();
    }
};

/// The scratch buffer's parts in bytes, in the order of Model's views.
const parts = [_]usize{
    L * H * 2,              L * H * 2,                 L * 3 * H * 2,               L * 3 * H * 2,               L * H * 2,
    L * H * 2,              L * F * 2,                 L * F * 4,                   L * H * 4,                   L * 2 * cfg.repo_hidden * 4,
    L * cfg.heads * 32 * 4, L * cfg.heads * 32 * 4,    R * cfg.latent_channels * 4, R * cfg.latent_channels * 4, cfg.modulated * host.vectors * H * 2,
    2 * H * 4,              C * cfg.cond_channels * 4, C * cfg.cond2_channels * 4,  H * 2,                       R * H * 2,
};

pub fn scratchBytes() usize {
    var total: usize = 0;
    for (parts) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: ks.Kernels,
    gemms: gm.Gemms,
    w: *wts.Weights,
    scratch: cuda.DeviceBuffer,
    /// A pass's start and end on the stream, for its GPU time.
    marks: [2]cuda.Event,
    // fp16 rows: the stream, a norm's output, a qkv projection, attention's packed heads, its output, a projection's
    x: u64 = 0,
    xn: u64 = 0,
    qkv: u64 = 0,
    qkh: u64 = 0,
    att: u64 = 0,
    y: u64 = 0,
    /// the MLP's GELU rows, and a block linear's fp32 sums before its bias
    u: u64 = 0,
    acc: u64 = 0,
    /// fp32 rows: RePo's norm (and the condition's two projections), its gate | content sums, cos and sin
    ln: u64 = 0,
    gc: u64 = 0,
    cos: u64 = 0,
    sin: u64 = 0,
    lat: u64 = 0,
    vel: u64 = 0,
    mods: u64 = 0,
    head: u64 = 0,
    f1: u64 = 0,
    f2: u64 = 0,
    ones: u64 = 0,
    pe: u64 = 0,
    /// The model time `mods` and `head` hold, if any, and their host copies.
    time: ?f32 = null,
    mods_host: host.Mods = undefined,
    head_host: [2][H]f32 = undefined,
    /// The stream's rows downloaded for `stream`, and the camera row each forward reads back.
    rows_host: []f16 = &.{},

    /// The denoiser of snapshot folder `dir` on `ctx`'s GPU, if it is one this family admits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Model {
        const sm = try ctx.capability();
        if (!admitted(sm)) {
            var buf: [256]u8 = undefined;
            const name = ctx.name(&buf) catch "this GPU";
            std.log.err("idol: {s} (sm_{d}) is not a GPU this family admits; it runs on sm_120 and sm_121 alone", .{ name, sm });
            return error.UncheckedGpu;
        }
        const d = ctx.d;
        const t = try gpa.create(Model);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .stream = undefined, .k = undefined, .gemms = undefined, .w = undefined, .scratch = undefined, .marks = undefined };
        t.rows_host = try gpa.alloc(f16, L * H);
        errdefer gpa.free(t.rows_host);
        t.stream = try cuda.Stream.init(d, false);
        errdefer t.stream.deinit();
        t.k = try ks.Kernels.load(d);
        errdefer t.k.unload();
        t.gemms = try gm.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        t.w = try wts.load(gpa, io, d, dir);
        errdefer {
            t.w.deinit();
            gpa.destroy(t.w);
        }
        t.scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes());
        errdefer t.scratch.free();
        t.marks[0] = try cuda.Event.init(d, true);
        errdefer t.marks[0].deinit();
        t.marks[1] = try cuda.Event.init(d, true);
        errdefer t.marks[1].deinit();
        var at: usize = t.scratch.ptr;
        const views = [_]*u64{ &t.x, &t.xn, &t.qkv, &t.qkh, &t.att, &t.y, &t.u, &t.acc, &t.ln, &t.gc, &t.cos, &t.sin, &t.lat, &t.vel, &t.mods, &t.head, &t.f1, &t.f2, &t.ones, &t.pe };
        for (views, parts) |view, n| {
            view.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        const ones = try gpa.alloc(f16, H);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        try t.upload(t.ones, std.mem.sliceAsBytes(ones));
        const pe = try gpa.alloc(f16, R * H);
        defer gpa.free(pe);
        host.positions(t.w.host.pos, pe);
        try t.upload(t.pe, std.mem.sliceAsBytes(pe));
        return t;
    }

    pub fn deinit(t: *Model) void {
        for (&t.marks) |*m| m.deinit();
        t.scratch.free();
        t.w.deinit();
        t.gpa.destroy(t.w);
        t.gemms.deinit();
        t.k.unload();
        t.stream.deinit();
        t.gpa.free(t.rows_host);
        t.gpa.destroy(t);
    }

    fn upload(t: *const Model, at: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoDAsync_v2(at, bytes.ptr, bytes.len, t.stream.handle), "cuMemcpyHtoDAsync");
        try t.stream.synchronize();
    }

    fn download(t: *const Model, out: []u8, at: u64) !void {
        try t.d.check(t.d.api.cuMemcpyDtoHAsync_v2(out.ptr, at, out.len, t.stream.handle), "cuMemcpyDtoHAsync");
        try t.stream.synchronize();
    }

    fn copy(t: *const Model, to: u64, from: u64, n: usize) !void {
        try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(to, from, n, t.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// The modulated block `i`'s vector `part` (0 shift, 1 one plus scale, 2 gate; 3 to 5 the MLP's).
    fn vec(t: *const Model, i: usize, part: usize) u64 {
        return t.mods + ((i * host.vectors) + part) * H * 2;
    }

    /// Every modulated block's vectors and the output's at model time `time`, unless they are already there.
    fn modulate(t: *Model, time: f32) !void {
        if (t.time == time) return;
        t.time = null;
        host.modulation(t.w.host, time, &t.mods_host, &t.head_host);
        try t.upload(t.mods, std.mem.sliceAsBytes(&t.mods_host));
        try t.upload(t.head, std.mem.sliceAsBytes(&t.head_host));
        t.time = time;
    }

    /// The image condition: DINOv3 rows [4101, 1280] and FLUX.2 VAE rows [4101, 128] (zeros: the negative branch).
    pub fn condition(t: *Model, feature1: []const f32, feature2: []const f32) !Condition {
        if (feature1.len != C * cfg.cond_channels or feature2.len != C * cfg.cond2_channels) return error.ConditionRows;
        var c: Condition = .{ .rows = try cuda.DeviceBuffer.alloc(t.d, C * H * 2) };
        errdefer c.rows.free();
        try t.upload(t.f1, std.mem.sliceAsBytes(feature1));
        try t.upload(t.f2, std.mem.sliceAsBytes(feature2));
        const a = t.ln;
        const b = t.ln + C * H * 4;
        try t.gemms.run(t.stream, .f32, t.f1, t.w.cond, a, C, H, cfg.cond_channels);
        try t.gemms.run(t.stream, .f32, t.f2, t.w.cond2, b, C, H, cfg.cond2_channels);
        try ks.cond(&t.k, t.stream, a, t.w.cond_b, b, t.w.cond2_b, t.x, C, H);
        for (t.w.context, t.w.context_repo) |blk, r| try t.block(blk, r, C, null);
        try t.copy(c.rows.ptr, t.x, C * H * 2);
        try t.stream.synchronize();
        return c;
    }

    /// The velocity of latent [8192, 16] and camera [5] at model time `time` (1000 times t) under condition `c`.
    pub fn forward(t: *Model, c: *const Condition, latent: []const f32, camera: *const [cfg.camera_channels]f32, time: f32, velocity: []f32, camera_velocity: *[cfg.camera_channels]f32) !Timing {
        if (latent.len != R * cfg.latent_channels or velocity.len != latent.len) return error.LatentRows;
        const started = seconds();
        try t.modulate(time);
        try t.upload(t.lat, std.mem.sliceAsBytes(latent));
        const camera_row = t.rows_host[0..H];
        host.cameraToken(t.w.host, camera, camera_row);
        try t.upload(t.x + (L - 1) * H * 2, std.mem.sliceAsBytes(camera_row));
        const spent = seconds() - started;
        try t.marks[0].record(t.stream);
        try t.copy(t.x + R * H * 2, c.rows.ptr, C * H * 2);
        try ks.embed(&t.k, t.stream, t.lat, t.w.input, t.w.input_b, t.pe, t.x, R);
        for (t.w.noise, t.w.noise_repo, 0..) |blk, r, i| try t.block(blk, r, R, i);
        for (t.w.blocks, t.w.repo, 0..) |blk, r, i| try t.block(blk, r, L, cfg.refiners + i);
        try ks.norm32(&t.k, t.stream, t.x, t.head + H * 4, t.head, t.ln, R, cfg.default_eps, .modulated);
        try t.gemms.run(t.stream, .f32, t.ln, t.w.out, t.vel, R, cfg.latent_channels, H);
        try ks.bias32(&t.k, t.stream, t.vel, t.w.out_b, R, cfg.latent_channels);
        try t.marks[1].record(t.stream);
        try t.download(std.mem.sliceAsBytes(velocity), t.vel);
        try t.download(std.mem.sliceAsBytes(camera_row), t.x + (L - 1) * H * 2);
        host.cameraVelocity(t.w.host, camera_row, &t.head_host, camera_velocity);
        const gpu = @as(f64, try cuda.Event.elapsedMs(t.marks[0], t.marks[1])) / 1000;
        return .{ .wall = seconds() - started, .gpu = gpu, .host = spent };
    }

    /// Modulated block `index` alone at model time `time` on `rows` (the noise refiner's 8192, else 12294) given rows.
    pub fn blockAt(t: *Model, index: usize, rows: []const f16, time: f32) !void {
        const n: usize = if (index < cfg.refiners) R else L;
        if (index >= cfg.modulated or rows.len != n * H) return error.BlockRows;
        try t.modulate(time);
        try t.upload(t.x, std.mem.sliceAsBytes(rows));
        if (index < cfg.refiners) {
            try t.block(t.w.noise[index], t.w.noise_repo[index], n, index);
        } else try t.block(t.w.blocks[index - cfg.refiners], t.w.repo[index - cfg.refiners], n, index);
        try t.stream.synchronize();
    }

    /// The residual stream's first `rows` rows as the last pass left them, fp16 (valid until the next call).
    pub fn residual(t: *Model, rows: usize) ![]const f16 {
        const out = t.rows_host[0 .. rows * H];
        try t.download(std.mem.sliceAsBytes(out), t.x);
        return out;
    }

    /// y fp16 [rows, n] = fp16(x w^T + b) from fp32 sums, then fp16(GELU) when `gelu`.
    fn linear(t: *Model, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, k: usize, gelu: bool) !void {
        try t.gemms.run(t.stream, .f16, x, w, t.acc, rows, n, k);
        try ks.bias(&t.k, t.stream, t.acc, b, y, rows, n, gelu);
    }

    /// One block over the stream's first `rows` rows: modulated block `mod`, or the condition refiner's (null).
    fn block(t: *Model, b: wts.Block, r: wts.Repo, rows: usize, mod: ?usize) !void {
        const k = &t.k;
        const s = t.stream;
        // RePo: angles from the block's input rows, its projections in fp32
        try ks.norm32(k, s, t.x, r.norm, r.norm_b, t.ln, rows, cfg.default_eps, .affine);
        try t.gemms.run(s, .f32, t.ln, r.maps, t.gc, rows, 2 * cfg.repo_hidden, H);
        try ks.repo(k, s, t.gc, r.final, r.freqs, t.cos, t.sin, rows);
        if (mod) |m| {
            try ks.norm(k, s, t.x, 0, 0, t.vec(m, 1), t.vec(m, 0), t.xn, rows, cfg.block_eps, .modulated);
        } else try ks.norm(k, s, t.x, b.norm1, b.norm1_b, t.ones, t.ones, t.xn, rows, cfg.block_eps, .affine);
        try t.linear(t.xn, b.qkv, b.qkv_b, t.qkv, rows, 3 * H, H, false);
        try ks.qk(k, s, t.qkv, b.q_norm, b.k_norm, t.cos, t.sin, t.qkh, rows);
        try ks.attention(k, s, t.qkh, t.att, rows, cfg.heads, 1.0 / @sqrt(@as(f32, cfg.head_dim)));
        try t.linear(t.att, b.out, b.out_b, t.y, rows, H, H, false);
        try ks.gateAdd(k, s, t.x, t.y, if (mod) |m| t.vec(m, 2) else t.ones, rows, H);
        if (mod) |m| {
            try ks.norm(k, s, t.x, 0, 0, t.vec(m, 4), t.vec(m, 3), t.xn, rows, cfg.block_eps, .modulated);
        } else try ks.norm(k, s, t.x, b.norm2, b.norm2_b, t.ones, t.ones, t.xn, rows, cfg.block_eps, .affine);
        try t.linear(t.xn, b.up, b.up_b, t.u, rows, F, H, true);
        try t.linear(t.u, b.down, b.down_b, t.y, rows, H, F, false);
        try ks.gateAdd(k, s, t.x, t.y, if (mod) |m| t.vec(m, 5) else t.ones, rows, H);
    }
};

test "scratch for the joint rows: every part 256-aligned, the condition's two projections fit RePo's norm rows" {
    try std.testing.expectEqual(@as(usize, 12294), L);
    try std.testing.expect(2 * C * H * 4 <= parts[8]);
    try std.testing.expect(scratchBytes() < 1 << 30);
    // the kernels index rows and offsets as c_int
    try std.testing.expect(L * F < std.math.maxInt(c_int));
}

test "admission: sm_120 and sm_121 alone" {
    try std.testing.expect(admitted(121) and admitted(120));
    for ([_]u32{ 80, 86, 89, 90, 100, 103, 122 }) |sm| try std.testing.expect(!admitted(sm));
}
