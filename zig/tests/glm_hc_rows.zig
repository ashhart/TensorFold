//! tf-glm-hc-rows: GLM-5.3-Flash's hyper-connection boundary, the expand on six groups a slice against one; bits, time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const hc = tf.glm.hc;
const glm = tf.glm;
const R = glm.weights.Ref;

const D = 4096;
const F = 4 * D;
const boundaries = 90; // a round's
const pool_fn = 1024; // fn copies the timed boundaries walk (768 MB: past the GPU's caches)
const fn_bytes = 24 * F * 2;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

const Io = struct { xo: mtl.Buffer, br: mtl.Buffer, post: mtl.Buffer, comb: mtl.Buffer, fns: mtl.Buffer, scale: mtl.Buffer, base: mtl.Buffer, norm: mtl.Buffer, out: [2]struct { xn: mtl.Buffer, part: mtl.Buffer, normed: mtl.Buffer, post: mtl.Buffer, comb: mtl.Buffer } };

fn one(e: mtl.ComputeEncoder, i: usize, pipes: [6]mtl.Pipeline, io: *const Io, side: usize, rows: u32, expand: bool, parts: bool) void {
    const o = io.out[side];
    hc.boundaryWith(e, pipes, glm.kernels.hc_shape, expand, rows, 1e-6, .{
        .x_old = R{ .buf = io.xo },
        .branch = R{ .buf = io.br },
        .post = R{ .buf = io.post },
        .comb = R{ .buf = io.comb },
        .fn_packed = (R{ .buf = io.fns }).at((i % pool_fn) * fn_bytes),
        .x_new = R{ .buf = o.xn },
        .part = R{ .buf = o.part },
        .scale = R{ .buf = io.scale },
        .base = R{ .buf = io.base },
        .norm = R{ .buf = io.norm },
        .normed = R{ .buf = o.normed },
    }, parts);
    _ = o.post;
}

fn run(q: mtl.Queue, comptime f: anytype, args: anytype, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |i| @call(.auto, f, .{ e, i } ++ args);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, comptime f: anytype, args: anytype) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try run(q, f, args, boundaries);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[4];
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try glm.kernels.load(gpa, device);
    const opts = mtl.ResourceOptions.shared;
    const M = 16;
    var io: Io = undefined;
    io.xo = try device.buffer(M * F * 2, opts);
    io.br = try device.buffer(M * D * 2, opts);
    io.post = try device.buffer(M * 4 * 4, opts);
    io.comb = try device.buffer(M * 16 * 4, opts);
    io.fns = try device.buffer(pool_fn * fn_bytes, opts);
    io.scale = try device.buffer(16, opts);
    io.base = try device.buffer(24 * 4, opts);
    io.norm = try device.buffer(D * 2, opts);
    for (&io.out) |*o| o.* = .{ .xn = try device.buffer(M * F * 2, opts), .part = try device.buffer(M * hc.partBytes(glm.kernels.hc_shape), opts), .normed = try device.buffer(M * D * 2, opts), .post = try device.buffer(M * 4 * 4, opts), .comb = try device.buffer(M * 16 * 4, opts) };
    var prng = std.Random.DefaultPrng.init(0x4c);
    const r = prng.random();
    for (io.xo.slice(u16, M * F)) |*v| v.* = toBf((r.float(f32) - 0.5) * 4);
    for (io.br.slice(u16, M * D)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    for (io.post.slice(f32, M * 4)) |*v| v.* = r.float(f32) * 2;
    for (io.comb.slice(f32, M * 16)) |*v| v.* = r.float(f32) * 0.5;
    for (io.fns.slice(u16, pool_fn * fn_bytes / 2)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.05);
    io.scale.slice(f32, 3)[0..3].* = .{ 0.9, 1.1, 0.7 };
    for (io.base.slice(f32, 24)) |*v| v.* = (r.float(f32) - 0.5);
    for (io.norm.slice(u16, D)) |*v| v.* = toBf(0.5 + r.float(f32));
    var post0: [M * 4]f32 = undefined;
    var comb0: [M * 16]f32 = undefined;
    @memcpy(&post0, io.post.slice(f32, M * 4)); // the split writes the next boundary's post and comb over them
    @memcpy(&comb0, io.comb.slice(f32, M * 16));
    var outs: [2][2][M * 16]f32 = undefined;
    std.debug.print("GPU {s}: a round's {d} boundaries, the expand's parts on six threadgroups a slice against one\n", .{ device.name(), boundaries });
    var checked: usize = 0;
    for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| for ([_]bool{ true, false }) |expand| {
        for (0..2) |side| {
            const o = io.out[side];
            @memset(o.xn.slice(u16, M * F), 0x7fc1);
            @memset(o.normed.slice(u16, M * D), 0x7fc1);
            @memset(o.part.slice(u32, M * hc.partBytes(glm.kernels.hc_shape) / 4), 0x7fc00001);
            @memcpy(io.post.slice(f32, M * 4), &post0);
            @memcpy(io.comb.slice(f32, M * 16), &comb0);
            _ = try run(queue, one, .{ k.hc_core, &io, side, rows, expand, side == 1 }, 1);
            @memcpy(outs[side][0][0 .. M * 4], io.post.slice(f32, M * 4));
            @memcpy(&outs[side][1], io.comb.slice(f32, M * 16));
        }
        if (!std.mem.eql(f32, &outs[0][1], &outs[1][1]) or !std.mem.eql(f32, outs[0][0][0 .. M * 4], outs[1][0][0 .. M * 4])) {
            std.debug.print("{d} rows (expand {}): post or comb BITS DIFFER\n", .{ rows, expand });
            return error.BitsDiffer;
        }
        const a = io.out[0];
        const b = io.out[1];
        const pb = rows * hc.partBytes(glm.kernels.hc_shape) / 4;
        if (!std.mem.eql(u16, a.xn.slice(u16, rows * F), b.xn.slice(u16, rows * F)) or !std.mem.eql(u32, a.part.slice(u32, pb), b.part.slice(u32, pb)) or !std.mem.eql(u16, a.normed.slice(u16, rows * D), b.normed.slice(u16, rows * D))) {
            std.debug.print("{d} rows (expand {}): BITS DIFFER\n", .{ rows, expand });
            return error.BitsDiffer;
        }
        checked += rows * (F + D) + pb;
        for (0..16) |_| _ = try run(queue, one, .{ k.hc_core, &io, 0, rows, expand, false }, boundaries);
        const t0 = try time(queue, one, .{ k.hc_core, &io, 0, rows, expand, false });
        const t1 = try time(queue, one, .{ k.hc_core, &io, 1, rows, expand, true });
        std.debug.print("  {d:2} rows{s}: one group a slice {d:6.2} us, six {d:6.2} us a boundary ({d:.2}x)\n", .{ rows, if (expand) "" else " (first)", t0, t1, t0 / t1 });
    };
    std.debug.print("{d} outputs equal\n", .{checked});
}
