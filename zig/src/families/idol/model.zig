//! Idol's denoiser on Metal: the image condition refined once, then latent, condition and camera rows each step.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const host = @import("host.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const Ref = ks.Ref;

const H = cfg.hidden;
const F = cfg.ffn;
const L = cfg.joint_rows;
const R = cfg.latent_rows;
const C = cfg.cond_rows;

pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

/// A refined image condition: the condition refiner's rows [4101, 1024] fp16.
pub const Condition = struct {
    rows: mtl.Buffer,

    pub fn deinit(c: *Condition) void {
        c.rows.deinit();
    }
};

/// The scratch buffer's parts in bytes, in the order of Model's views.
const parts = [_]usize{
    L * H * 2, L * H * 2,                 L * 3 * H * 2,               L * H * 2,                          L * H * 2,                          L * H * 2,                   L * H * 2,                   L * H * 2,
    L * F * 2, L * H * 4,                 L * 2 * cfg.repo_hidden * 4, L * cfg.heads * cfg.repo_pairs * 4, L * cfg.heads * cfg.repo_pairs * 4, R * cfg.latent_channels * 4, R * cfg.latent_channels * 4, cfg.modulated * host.vectors * H * 2,
    2 * H * 4, C * cfg.cond_channels * 4, C * cfg.cond2_channels * 4,  H * 2,                              R * H * 2,
};

pub fn scratchBytes() usize {
    var total: usize = 0;
    for (parts) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    scratch: mtl.Buffer,
    // fp16 rows: the stream, a norm's output, packed qkv, heads [16, rows, 64], attention rows, a projection's output
    x: Ref,
    xn: Ref,
    qkv: Ref,
    q: Ref,
    k_: Ref,
    v: Ref,
    att: Ref,
    y: Ref,
    /// the MLP's GELU rows
    u: Ref,
    /// fp32 rows: RePo's norm (and the condition's two projections), its gate | content sums, cos and sin
    ln: Ref,
    gc: Ref,
    cos: Ref,
    sin: Ref,
    lat: Ref,
    vel: Ref,
    mods: Ref,
    head: Ref,
    f1: Ref,
    f2: Ref,
    ones: Ref,
    pe: Ref,
    /// The model time `mods` and `head` hold, if any.
    time: ?f32 = null,

    /// The denoiser of snapshot folder `dir`.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !*Model {
        // qualified on the M5's tensor units alone: the projections and attention take another path before them
        if (!device.tensorUnits()) {
            std.log.err("idol: {s} has no tensor units; this family runs on the M5's alone, the only chip it was checked on", .{device.name()});
            return error.TensorUnitsRequired;
        }
        const t = try gpa.create(Model);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.device = device;
        t.time = null;
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.w = try wts.load(gpa, io, device, dir);
        errdefer t.w.deinit();
        t.scratch = try device.buffer(scratchBytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &t.x, &t.xn, &t.qkv, &t.q, &t.k_, &t.v, &t.att, &t.y, &t.u, &t.ln, &t.gc, &t.cos, &t.sin, &t.lat, &t.vel, &t.mods, &t.head, &t.f1, &t.f2, &t.ones, &t.pe };
        for (views, parts) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        @memset(halves(t.ones, H), 1.0);
        host.positions(t.w.host.pos, halves(t.pe, R * H));
        return t;
    }

    pub fn deinit(t: *Model) void {
        t.scratch.deinit();
        t.w.deinit();
        t.k.deinit();
        t.queue.deinit();
        t.gpa.destroy(t);
    }

    fn floats(r: Ref, n: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn halves(r: Ref, n: usize) []f16 {
        return @as([*]f16, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    /// The modulated block `i`'s vector `part` (0 shift, 1 one plus scale, 2 gate; 3 to 5 the MLP's).
    fn vec(t: *const Model, i: usize, part: usize) Ref {
        return t.mods.at(((i * host.vectors) + part) * H * 2);
    }

    /// The image condition: DINOv3 rows [4101, 1280] and FLUX.2 VAE rows [4101, 128] (zeros: the negative branch).
    pub fn condition(t: *Model, feature1: []const f32, feature2: []const f32) !Condition {
        if (feature1.len != C * cfg.cond_channels or feature2.len != C * cfg.cond2_channels) return error.ConditionRows;
        var c: Condition = .{ .rows = try t.device.buffer(C * H * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked) };
        errdefer c.rows.deinit();
        @memcpy(floats(t.f1, feature1.len), feature1);
        @memcpy(floats(t.f2, feature2.len), feature2);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        const a = t.ln;
        const b = t.ln.at(C * H * 4);
        t.k.linear32(e, t.f1, t.w.cond, a, C, H, cfg.cond_channels, H);
        t.k.linear32(e, t.f2, t.w.cond2, b, C, H, cfg.cond2_channels, H);
        t.k.cond(e, a, t.w.cond_b, b, t.w.cond2_b, t.x, C);
        for (t.w.context, t.w.context_repo) |blk, r| t.block(e, blk, r, C, null);
        _ = try finish(cb, e, started, 0);
        @memcpy(halves(.{ .buf = c.rows }, C * H), halves(t.x, C * H));
        return c;
    }

    /// The velocity of latent [8192, 16] and camera [5] at model time `time` (1000 times t) under condition `c`.
    pub fn forward(t: *Model, c: *const Condition, latent: []const f32, camera: *const [cfg.camera_channels]f32, time: f32, velocity: []f32, camera_velocity: *[cfg.camera_channels]f32) !Timing {
        if (latent.len != R * cfg.latent_channels or velocity.len != latent.len) return error.LatentRows;
        const started = mtl.clock.seconds();
        if (t.time != time) {
            t.time = null;
            host.modulation(t.w.host, time, @ptrCast(@alignCast(t.mods.addr())), @ptrCast(@alignCast(t.head.addr())));
            t.time = time;
        }
        @memcpy(floats(t.lat, latent.len), latent);
        @memcpy(halves(t.x.at(R * H * 2), C * H), halves(.{ .buf = c.rows }, C * H));
        host.cameraToken(t.w.host, camera, halves(t.x.at((L - 1) * H * 2), H));
        const spent = mtl.clock.seconds() - started;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.k.embed(e, t.lat, t.w.input, t.w.input_b, t.pe, t.x);
        for (t.w.noise, t.w.noise_repo, 0..) |blk, r, i| t.block(e, blk, r, R, i);
        for (t.w.blocks, t.w.repo, 0..) |blk, r, i| t.block(e, blk, r, L, cfg.refiners + i);
        const head: Ref = t.head;
        t.k.norm32(e, t.x, head.at(H * 4), head, t.ln, R, cfg.default_eps, .modulated);
        t.k.linear32(e, t.ln, t.w.out, t.vel, R, cfg.latent_channels, H, cfg.latent_channels);
        t.k.bias32(e, t.vel, t.w.out_b, R, cfg.latent_channels);
        const timing = try finish(cb, e, started, spent);
        @memcpy(velocity, floats(t.vel, velocity.len));
        host.cameraVelocity(t.w.host, halves(t.x.at((L - 1) * H * 2), H), @ptrCast(@alignCast(t.head.addr())), camera_velocity);
        return timing;
    }

    /// Modulated block `index` alone at model time `time` on `rows` (the noise refiner's 8192, else 12294) given rows.
    pub fn blockAt(t: *Model, index: usize, rows: []const f16, time: f32) !void {
        const n: usize = if (index < cfg.refiners) R else L;
        if (index >= cfg.modulated or rows.len != n * H) return error.BlockRows;
        if (t.time != time) {
            t.time = null;
            host.modulation(t.w.host, time, @ptrCast(@alignCast(t.mods.addr())), @ptrCast(@alignCast(t.head.addr())));
            t.time = time;
        }
        @memcpy(halves(t.x, rows.len), rows);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (index < cfg.refiners) t.block(e, t.w.noise[index], t.w.noise_repo[index], n, index) else t.block(e, t.w.blocks[index - cfg.refiners], t.w.repo[index - cfg.refiners], n, index);
        _ = try finish(cb, e, mtl.clock.seconds(), 0);
    }

    /// The stream's first `rows` rows as the last pass left them, fp16.
    pub fn stream(t: *const Model, rows: usize) []const f16 {
        return halves(t.x, rows * H);
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, started: f64, spent: f64) !Timing {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("idol: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds(), .host = spent };
    }

    /// One block over the stream's first `rows` rows: modulated block `mod`, or the condition refiner's (null).
    fn block(t: *const Model, e: mtl.ComputeEncoder, b: wts.Block, r: wts.Repo, rows: usize, mod: ?usize) void {
        const k = &t.k;
        // RePo: angles from the block's input rows
        k.normSplit(e, t.x, r.norm, r.norm_b, t.qkv, rows, cfg.default_eps);
        k.split(e, t.qkv, r.maps, t.gc, rows);
        k.repo(e, t.gc, r.inv, r.final, r.freqs, t.cos, t.sin, rows);
        if (mod) |m| {
            k.norm(e, t.x, t.ln, t.ln, t.vec(m, 1), t.vec(m, 0), t.xn, rows, cfg.block_eps, .modulated);
        } else {
            k.norm(e, t.x, b.norm1, b.norm1_b, t.ones, t.ones, t.xn, rows, cfg.block_eps, .affine);
        }
        k.linear(e, t.xn, b.qkv, b.qkv_b, t.qkv, rows, 3 * H, H, false);
        k.heads(e, t.qkv, b.q_norm, b.k_norm, t.cos, t.sin, t.q, t.k_, t.v, rows);
        k.attention(e, t.q, t.k_, t.v, t.att, rows, rows);
        k.linear(e, t.att, b.out, b.out_b, t.y, rows, H, H, false);
        k.gateAdd(e, t.x, t.y, if (mod) |m| t.vec(m, 2) else t.ones, rows);
        if (mod) |m| {
            k.norm(e, t.x, t.ln, t.ln, t.vec(m, 4), t.vec(m, 3), t.xn, rows, cfg.block_eps, .modulated);
        } else {
            k.norm(e, t.x, b.norm2, b.norm2_b, t.ones, t.ones, t.xn, rows, cfg.block_eps, .affine);
        }
        k.linear(e, t.xn, b.up, b.up_b, t.u, rows, F, H, true);
        k.linear(e, t.u, b.down, b.down_b, t.y, rows, H, F, false);
        k.gateAdd(e, t.x, t.y, if (mod) |m| t.vec(m, 5) else t.ones, rows);
    }
};

test "scratch for the joint rows fits a GPU buffer, each part 256-aligned" {
    try std.testing.expect(scratchBytes() < 1 << 31);
    try std.testing.expectEqual(@as(usize, 12294), L);
}
