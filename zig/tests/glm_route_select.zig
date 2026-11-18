//! tf-glm-route-select: GLM's route select (top-k, weights, expert groups) against the kept reference, bytes; GPU time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const route = tf.moe_route;

const E = 288;
const TOPK = 8;
const M = 16;
const reps = 42; // a round's MoE layers

const Out = struct { pick: mtl.Buffer, wts: mtl.Buffer, ids: mtl.Buffer, members: mtl.Buffer, count: mtl.Buffer, mine: mtl.Buffer, theirs: mtl.Buffer, counts: mtl.Buffer, word: mtl.Buffer };
const sizes = [_]usize{ M * TOPK * 4, M * TOPK * 4, M * TOPK * 4, M * TOPK * M * 4, 16, M * TOPK * 4, M * TOPK * 4, 16, 16 };

fn buffers(o: *const Out) [9]mtl.Buffer {
    return .{ o.pick, o.wts, o.ids, o.members, o.count, o.mine, o.theirs, o.counts, o.word };
}

const Rig = struct { logits: mtl.Buffer, bias: mtl.Buffer, out: [2]Out, pipes: [2]mtl.Pipeline };

fn launch(e: mtl.ComputeEncoder, r: *const Rig, side: usize, rows: u32, own: [2]u32) void {
    const o = &r.out[side];
    const B = struct {
        fn of(b: mtl.Buffer) glm.weights.Ref {
            return .{ .buf = b };
        }
    };
    route.select(e, r.pipes[side], B.of(r.logits), B.of(r.bias), 2.5, rows, own, route.Outputs(glm.weights.Ref){ .pick = B.of(o.pick), .wts = B.of(o.wts), .ids = B.of(o.ids), .members = B.of(o.members), .count = B.of(o.count), .mine = B.of(o.mine), .theirs = B.of(o.theirs), .counts = B.of(o.counts), .word = B.of(o.word) });
}

fn run(q: mtl.Queue, r: *const Rig, side: usize, rows: u32, own: [2]u32, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |_| launch(e, r, side, rows, own);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, r: *const Rig, side: usize, rows: u32) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try run(q, r, side, rows, .{ 0, E }, reps);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[4];
}

/// Draw `d`'s logits and bias: 0 random, 1 a few levels (ties within and across rows), 2 edge values, 3 all equal.
fn fill(r: *const Rig, rnd: std.Random, d: usize) void {
    const lg = r.logits.slice(f32, M * E);
    const bs = r.bias.slice(f32, E);
    for (lg, 0..) |*v, i| v.* = switch (d) {
        0 => (rnd.float(f32) - 0.5) * 8,
        1 => @round((rnd.float(f32) - 0.5) * 6) / 2,
        2 => if (i % 97 == 0) -0.0 else if (i % 89 == 0) 90 else if (i % 83 == 0) -90 else if (i % 79 == 0) 1e-30 else if (i % 211 == 5) std.math.nan(f32) else (rnd.float(f32) - 0.5) * 4,
        else => 0.25,
    };
    for (bs, 0..) |*v, i| v.* = switch (d) {
        0 => (rnd.float(f32) - 0.5) * 0.2,
        1, 3 => @as(f32, @floatFromInt(i % 4)) * 0.125,
        else => if (i == 7 or i == 200) -std.math.inf(f32) else if (i % 13 == 0) 0.0 else (rnd.float(f32) - 0.5) * 0.1,
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try glm.kernels.load(gpa, device);
    const src = try std.mem.concat(gpa, u8, &.{ try route.source(gpa, glm.kernels.route_shape), @embedFile("glm_route_select_ref.metal") });
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    const opts = mtl.ResourceOptions.shared;
    var r: Rig = .{ .logits = try device.buffer(M * E * 4, opts), .bias = try device.buffer(E * 4, opts), .out = undefined, .pipes = .{ try mtl.Pipeline.init(device, lib, "tf_route_select_ref", false), k.route_select } };
    for (&r.out) |*o| {
        var bs: [9]mtl.Buffer = undefined;
        for (&bs, sizes) |*b, n| b.* = try device.buffer(n, opts);
        o.* = .{ .pick = bs[0], .wts = bs[1], .ids = bs[2], .members = bs[3], .count = bs[4], .mine = bs[5], .theirs = bs[6], .counts = bs[7], .word = bs[8] };
    }
    var prng = std.Random.DefaultPrng.init(0x5e);
    std.debug.print("GPU {s}: the route select of 1-16 rows against the reference kernel ({d} launches a timing)\n", .{ device.name(), reps });
    var checked: usize = 0;
    const owns = [_][2]u32{ .{ 0, E }, .{ 0, E / 2 }, .{ E / 2, E } };
    for (0..4) |d| {
        for (0..3) |_| { // three fills a draw (draw 3 the same each time)
            fill(&r, prng.random(), d);
            for (1..M + 1) |rows| for (owns) |own| {
                for (&r.out) |*o| for (buffers(o), sizes) |b, n| @memset(b.slice(u8, n), 0x5a);
                for (0..2) |side| _ = try run(queue, &r, side, @intCast(rows), own, 1);
                for (buffers(&r.out[0]), buffers(&r.out[1]), sizes, 0..) |a, b, n, i| if (!std.mem.eql(u8, a.slice(u8, n), b.slice(u8, n))) {
                    std.debug.print("draw {d}, {d} rows, experts {d}-{d}: output {d} BYTES DIFFER\n", .{ d, rows, own[0], own[1], i });
                    return error.BitsDiffer;
                };
                checked += rows * TOPK;
            };
        }
    }
    fill(&r, prng.random(), 0);
    for (0..16) |_| _ = try run(queue, &r, 0, 16, .{ 0, E }, reps); // clocks up
    for ([_]u32{ 1, 2, 3, 4, 8, 16 }) |rows| {
        const t0 = try time(queue, &r, 0, rows);
        const t1 = try time(queue, &r, 1, rows);
        std.debug.print("  {d:2} rows: reference {d:5.2} us, ours {d:5.2} us a launch\n", .{ rows, t0, t1 });
    }
    std.debug.print("{d} picks equal (with weights, groups and pick lists) over four draws\n", .{checked});
}
