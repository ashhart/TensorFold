//! tf-glm-hc-rows: GLM's hyper-connection boundary (expand on six groups a slice, split on two) against one each; time.
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

/// The reference split (tests/glm_hc_split_one.metal): one group a row, sg 0's Sinkhorn before the collapse.
var split_one: mtl.Pipeline = undefined;

fn bindSplit(e: mtl.ComputeEncoder, io: *const Io, side: usize, expand: bool) void {
    const o = io.out[side];
    inline for (.{ if (expand) o.xn else io.xo, o.part, io.scale, io.base, io.norm }, 0..) |b, j| e.setBuffer(b, 0, j);
    e.setValue(@as(f32, 1e-6), 5);
    inline for (.{ o.normed, io.post, io.comb }, 6..) |b, j| e.setBuffer(b, 0, j);
}

/// The split alone: the reference's one group a row (side 0) or core/hc.zig's (side 1).
fn split(e: mtl.ComputeEncoder, _: usize, pipes: [6]mtl.Pipeline, io: *const Io, side: usize, rows: u32, expand: bool) void {
    e.setPipeline(if (side == 0) split_one else pipes[2]);
    bindSplit(e, io, side, expand);
    e.dispatchGroups(mtl.Size.of(rows, if (side == 0) 1 else hc.split_groups, 1), mtl.Size.of(1024, 1, 1));
}

fn one(e: mtl.ComputeEncoder, i: usize, pipes: [6]mtl.Pipeline, io: *const Io, side: usize, rows: u32, expand: bool, parts: bool) void {
    const o = io.out[side];
    if (side == 0) { // the reference: the expand on one group a slice, then the one-group split
        e.setPipeline(pipes[if (expand) 0 else 1]);
        inline for (.{ io.xo, io.br, io.post, io.comb }, 0..) |b, j| e.setBuffer(b, 0, j);
        const fnp = (R{ .buf = io.fns }).at((i % pool_fn) * fn_bytes);
        e.setBuffer(fnp.buf, fnp.off, 4);
        e.setBuffer(o.xn, 0, 5);
        e.setBuffer(o.part, 0, 6);
        e.dispatchGroups(mtl.Size.of(4 * D / 1024, rows, 1), mtl.Size.of(256, 1, 1));
        return split(e, i, pipes, io, side, rows, expand);
    }
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
    const ref_src = try std.mem.concat(gpa, u8, &.{ try hc.source(gpa, glm.kernels.hc_shape), @embedFile("glm_hc_split_one.metal") });
    const ref_lib = try mtl.Library.fromSource(device, ref_src, mtl.CompileOptions.mlx());
    split_one = try mtl.Pipeline.init(device, ref_lib, "tf_hc_split_one", false);
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
    io.scale.slice(f32, 3)[0..3].* = .{ 0.9, 1.1, 0.7 };
    var post0: [M * 4]f32 = undefined;
    var comb0: [M * 16]f32 = undefined;
    var outs: [2][2][M * 16]f32 = undefined;
    std.debug.print("GPU {s}: a round's {d} boundaries; the expand on six groups a slice (to 4 rows) and the split on two a row, against one each\n", .{ device.name(), boundaries });
    var checked: usize = 0;
    for (0..2) |draw| for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| for ([_]bool{ true, false }) |expand| {
        if (rows == 1 and expand) { // draw 1: wider values, zeros, -0 and large streams among them
            const w: f32 = if (draw == 0) 1 else 16;
            for (io.xo.slice(u16, M * F), 0..) |*v, i| v.* = if (draw == 1 and i % 61 == 0) 0x8000 else if (draw == 1 and i % 67 == 0) toBf(3e4) else toBf((r.float(f32) - 0.5) * 4 * w);
            for (io.br.slice(u16, M * D)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2 * w);
            for (io.post.slice(f32, M * 4)) |*v| v.* = r.float(f32) * 2;
            for (io.comb.slice(f32, M * 16)) |*v| v.* = r.float(f32) * 0.5;
            for (io.fns.slice(u16, pool_fn * fn_bytes / 2)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.05 * w);
            for (io.base.slice(f32, 24)) |*v| v.* = (r.float(f32) - 0.5) * w;
            for (io.norm.slice(u16, D)) |*v| v.* = toBf(0.5 + r.float(f32));
            @memcpy(&post0, io.post.slice(f32, M * 4)); // the split writes the next boundary's post and comb over them
            @memcpy(&comb0, io.comb.slice(f32, M * 16));
        }
        for (0..2) |side| {
            const o = io.out[side];
            @memset(o.xn.slice(u16, M * F), 0x7fc1);
            @memset(o.normed.slice(u16, M * D), 0x7fc1);
            @memset(o.part.slice(u32, M * hc.partBytes(glm.kernels.hc_shape) / 4), 0x7fc00001);
            @memcpy(io.post.slice(f32, M * 4), &post0);
            @memcpy(io.comb.slice(f32, M * 16), &comb0);
            _ = try run(queue, one, .{ k.hc_core, &io, side, rows, expand, rows <= hc.parts_rows }, 1);
            @memcpy(outs[side][0][0 .. M * 4], io.post.slice(f32, M * 4));
            @memcpy(&outs[side][1], io.comb.slice(f32, M * 16));
        }
        if (!std.mem.eql(f32, &outs[0][1], &outs[1][1]) or !std.mem.eql(f32, outs[0][0][0 .. M * 4], outs[1][0][0 .. M * 4])) {
            std.debug.print("draw {d}, {d} rows (expand {}): post or comb BITS DIFFER\n", .{ draw, rows, expand });
            return error.BitsDiffer;
        }
        const a = io.out[0];
        const b = io.out[1];
        const pb = rows * hc.partBytes(glm.kernels.hc_shape) / 4;
        if (!std.mem.eql(u16, a.xn.slice(u16, rows * F), b.xn.slice(u16, rows * F)) or !std.mem.eql(u32, a.part.slice(u32, pb), b.part.slice(u32, pb)) or !std.mem.eql(u16, a.normed.slice(u16, rows * D), b.normed.slice(u16, rows * D))) {
            std.debug.print("draw {d}, {d} rows (expand {}): BITS DIFFER\n", .{ draw, rows, expand });
            return error.BitsDiffer;
        }
        checked += rows * (F + D + 20) + pb;
        if (draw > 0) continue;
        for (0..16) |_| _ = try run(queue, one, .{ k.hc_core, &io, 0, rows, expand, false }, boundaries);
        const t0 = try time(queue, one, .{ k.hc_core, &io, 0, rows, expand, false });
        const t1 = try time(queue, one, .{ k.hc_core, &io, 1, rows, expand, rows <= hc.parts_rows });
        const s0 = try time(queue, split, .{ k.hc_core, &io, 0, rows, expand });
        const s1 = try time(queue, split, .{ k.hc_core, &io, 1, rows, expand });
        std.debug.print("  {d:2} rows{s}: reference {d:6.2} us, ours {d:6.2} us a boundary ({d:.2}x); the split alone {d:5.2} -> {d:5.2} us\n", .{ rows, if (expand) "" else " (first)", t0, t1, t0 / t1, s0, s1 });
    };
    std.debug.print("{d} outputs equal over two draws\n", .{checked});
}
