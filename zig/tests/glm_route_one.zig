//! tf-glm-route-one: GLM's router and indexer-gate logits for a one-row window against tf_route_logits; GPU time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const route = tf.moe_route;

const K = 4096;
const copies = 64; // router copies the timed launches walk (past the GPU's caches)
const reps = 53; // a round's: 42 MoE routers and 11 indexer gates

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

const Rig = struct { x: mtl.Buffer, w: mtl.Buffer, out: [2]mtl.Buffer, experts: u32, s: route.Shape };

fn launch(e: mtl.ComputeEncoder, i: usize, pipe: mtl.Pipeline, r: *const Rig, side: usize) void {
    const wb = @as(usize, r.experts) * K * 2;
    route.logits(e, pipe, r.s, .{ .buf = r.x, .off = 0 }, .{ .buf = r.w, .off = (i % copies) * wb }, .{ .buf = r.out[side], .off = 0 }, 1);
}

fn run(q: mtl.Queue, pipe: mtl.Pipeline, r: *const Rig, side: usize, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |i| launch(e, i, pipe, r, side);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, pipe: mtl.Pipeline, r: *const Rig) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try run(q, pipe, r, 0, reps);
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
    var prng = std.Random.DefaultPrng.init(0x52);
    const rnd = prng.random();
    std.debug.print("GPU {s}: one-row router logits against tf_route_logits ({d} launches a timing)\n", .{ device.name(), reps });
    var checked: usize = 0;
    const kinds = [_]struct { name: []const u8, s: route.Shape, full: mtl.Pipeline, one: mtl.Pipeline }{
        .{ .name = "router (288, fp32)", .s = glm.kernels.route_shape, .full = k.route_logits, .one = k.route_logits_one },
        .{ .name = "indexer gate (128, bf16)", .s = glm.kernels.igate_shape, .full = k.igate_logits, .one = k.igate_logits_one },
    };
    for (kinds) |kd| {
        const E = kd.s.experts;
        const ob: usize = if (kd.s.out_bf16) 2 else 4;
        var r = Rig{ .x = try device.buffer(16 * K * 2, opts), .w = try device.buffer(copies * @as(usize, E) * K * 2, opts), .out = undefined, .experts = E, .s = kd.s };
        for (&r.out) |*o| o.* = try device.buffer(16 * @as(usize, E) * 4, opts);
        for (r.w.slice(u16, copies * @as(usize, E) * K)) |*v| v.* = toBf((rnd.float(f32) - 0.5) * 0.1);
        for (0..3) |draw| { // draw 2: zeros, -0, large and tiny values among x
            for (r.x.slice(u16, 16 * K), 0..) |*v, i| v.* = if (draw == 2 and i % 53 == 0) 0x8000 else if (draw == 2 and i % 59 == 0) toBf(6e4) else if (draw == 2 and i % 61 == 0) toBf(1e-30) else if (draw == 2 and i % 67 == 0) 0 else toBf((rnd.float(f32) - 0.5) * 4);
            for (0..copies) |c| {
                for (&r.out) |o| @memset(o.slice(u8, 16 * @as(usize, E) * 4), 0x5a);
                try runAt(queue, kd.full, &r, 0, c);
                try runAt(queue, kd.one, &r, 1, c);
                if (!std.mem.eql(u8, r.out[0].slice(u8, E * ob), r.out[1].slice(u8, E * ob))) {
                    std.debug.print("{s}, draw {d}, copy {d}: BITS DIFFER\n", .{ kd.name, draw, c });
                    return error.BitsDiffer;
                }
                checked += E;
            }
        }
        for (0..8) |_| _ = try run(queue, kd.full, &r, 0, reps); // clocks up
        const t0 = try time(queue, kd.full, &r);
        const t1 = try time(queue, kd.one, &r);
        std.debug.print("  {s}: tf_route_logits {d:5.2} us, one row {d:5.2} us a launch\n", .{ kd.name, t0, t1 });
    }
    std.debug.print("{d} logits equal over three draws\n", .{checked});
}

fn runAt(q: mtl.Queue, pipe: mtl.Pipeline, r: *const Rig, side: usize, copy: usize) !void {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    launch(e, copy, pipe, r, side);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}
