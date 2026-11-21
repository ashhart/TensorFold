//! tf-glm-moe-pick: one-row experts with the select in the gate/up groups against the select launch; bytes, time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;
const affine = tf.glm.affine;
const route = tf.moe_route;
const Ref = affine.Ref;

const opts = mtl.ResourceOptions.shared;
const a = std.heap.page_allocator;
const E = 288;
const D = 4096;
const N = 2048;
const TOPK = 8;
const MAXR = 16;
const layers = 42; // a round's MoE layers: each launch its own logits row
const block = 1 << 24; // random bytes the weights repeat

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fillRepeat(dst: []u8, src: []const u8) void {
    var i: usize = 0;
    while (i < dst.len) : (i += src.len) {
        const n = @min(src.len, dst.len - i);
        @memcpy(dst[i..][0..n], src[0..n]);
    }
}

fn mat(device: mtl.Device, noise: []const u8, n: usize, k: usize, r: std.Random) !affine.Affine {
    const w: affine.Width = .{ .bits = 4, .group = 64 };
    const wb = affine.wBytes(n, k, w);
    const sb = affine.sBytes(n, k, w);
    const bs = [3]mtl.Buffer{ try device.buffer(wb, opts), try device.buffer(sb, opts), try device.buffer(sb, opts) };
    fillRepeat(bs[0].slice(u8, wb), noise);
    for (bs[1].slice(u16, sb / 2)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
    for (bs[2].slice(u16, sb / 2)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
    return .{ .w = .{ .buf = bs[0] }, .s = .{ .buf = bs[1] }, .b = .{ .buf = bs[2] }, .n = @intCast(n), .k = @intCast(k), .bits = 4, .group = 64 };
}

const W = struct { gate: affine.Affine, up: affine.Affine, down: affine.Affine, sh_gate_up: affine.Affine, sh_down: affine.Affine };
const Out = struct { act: mtl.Buffer, ye: mtl.Buffer, pick: mtl.Buffer, wts: mtl.Buffer, ids: mtl.Buffer, members: mtl.Buffer, count: mtl.Buffer, lists: mtl.Buffer };
const Rig = struct { k: *const kernels.Kernels, w: W, x: mtl.Buffer, logits: mtl.Buffer, bias: mtl.Buffer, none: mtl.Buffer, out: [2]Out };

fn of(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

fn bind(e: mtl.ComputeEncoder, first: usize, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, first + i);
}

/// moe.zig's one-row `both`: side 0 select then experts, side 1 the pick kernels; launch i reads logits row i.
fn launch(e: mtl.ComputeEncoder, i: usize, r: *const Rig, side: usize) void {
    const k = r.k;
    const o = r.out[side];
    const w = r.w;
    const lg = of(r.logits).at((i % layers) * E * 4);
    if (side == 0) {
        const T: usize = TOPK * MAXR * 4;
        const l = of(o.lists);
        route.select(e, k.route_select, lg, of(r.bias), 2.5, 1, .{ 0, E }, route.Outputs(Ref){ .pick = of(o.pick), .wts = of(o.wts), .ids = of(o.ids), .members = of(o.members), .count = of(o.count), .mine = l, .theirs = l.at(T), .counts = l.at(2 * T), .word = l.at(2 * T + 64) });
    }
    e.setPipeline(if (side == 0) k.moe_gateup_0_one else k.moe_gateup_pick);
    bind(e, 0, .{of(r.x)});
    e.setBytes(std.mem.asBytes(&[2]i32{ 1, D }), 1);
    bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b });
    bind(e, 11, if (side == 0) .{ of(o.ids), of(o.members), of(o.count) } else .{ lg, of(r.bias), of(o.pick) });
    e.setValue(@as(f32, 10.0), 14);
    bind(e, 15, .{of(o.act)});
    if (side == 1) bind(e, 16, .{of(o.wts)});
    if (side == 1) e.setValue(@as(f32, 2.5), 17);
    e.dispatchThreads(mtl.Size.of(32, N / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
    e.setPipeline(if (side == 0) k.moe_down_0_one else k.moe_down_pick);
    bind(e, 0, .{of(o.act)});
    e.setBytes(std.mem.asBytes(&[3]i32{ 1, TOPK + 1, N }), 1);
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b });
    bind(e, 8, if (side == 0) .{ of(o.ids), of(o.members), of(o.count) } else .{ of(o.pick), of(r.none), of(r.none) });
    bind(e, 11, .{of(o.ye)});
    e.dispatchThreads(mtl.Size.of(32, D / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
}

fn run(q: mtl.Queue, r: *const Rig, side: usize, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |i| launch(e, i, r, side);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, r: *const Rig, side: usize) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try run(q, r, side, layers);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[4];
}

/// Draw `d`'s logits rows and bias: 0 random, 1 a few levels (ties), 2 edge values (NaN, -inf bias, -0), 3 all equal.
fn fill(r: *const Rig, rnd: std.Random, d: usize) void {
    for (r.logits.slice(f32, layers * E), 0..) |*v, i| v.* = switch (d) {
        0 => (rnd.float(f32) - 0.5) * 8,
        1 => @round((rnd.float(f32) - 0.5) * 6) / 2,
        2 => if (i % 97 == 0) -0.0 else if (i % 89 == 0) 90 else if (i % 211 == 5) std.math.nan(f32) else (rnd.float(f32) - 0.5) * 4,
        else => 0.25,
    };
    for (r.bias.slice(f32, E), 0..) |*v, i| v.* = switch (d) {
        0 => (rnd.float(f32) - 0.5) * 0.2,
        1, 3 => @as(f32, @floatFromInt(i % 4)) * 0.125,
        else => if (i == 7 or i == 200) -std.math.inf(f32) else (rnd.float(f32) - 0.5) * 0.1,
    };
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try kernels.load(a, device);
    var prng = std.Random.DefaultPrng.init(0x7c);
    const rnd = prng.random();
    const noise = try a.alloc(u8, block);
    rnd.bytes(noise);
    var r: Rig = .{ .k = k, .w = undefined, .x = try device.buffer(D * 2, opts), .logits = try device.buffer(layers * E * 4, opts), .bias = try device.buffer(E * 4, opts), .none = try device.buffer(64, opts), .out = undefined };
    r.w = .{ .gate = try mat(device, noise, E * N, D, rnd), .up = try mat(device, noise[17..], E * N, D, rnd), .down = try mat(device, noise[33..], E * D, N, rnd), .sh_gate_up = try mat(device, noise[49..], 2 * N, D, rnd), .sh_down = try mat(device, noise[65..], D, N, rnd) };
    for (&r.out) |*o| o.* = .{ .act = try device.buffer((TOPK + 1) * N * 2, opts), .ye = try device.buffer((TOPK + 1) * D * 2, opts), .pick = try device.buffer(TOPK * MAXR * 4, opts), .wts = try device.buffer(TOPK * MAXR * 4, opts), .ids = try device.buffer(TOPK * MAXR * 4, opts), .members = try device.buffer(TOPK * MAXR * MAXR * 4, opts), .count = try device.buffer(64, opts), .lists = try device.buffer(3 * TOPK * MAXR * 4 + 512, opts) };
    std.debug.print("GPU {s}: a one-row MoE layer's experts with the select in the gate/up groups against the select launch\n", .{device.name()});
    var checked: usize = 0;
    for (0..4) |d| for (0..2) |_| {
        fill(&r, rnd, d);
        for (r.x.slice(u16, D)) |*v| v.* = toBf((rnd.float(f32) - 0.5) * 2);
        for (0..layers) |i| {
            for (r.out) |o| {
                @memset(o.act.slice(u16, (TOPK + 1) * N), 0x7fc1);
                @memset(o.ye.slice(u16, (TOPK + 1) * D), 0x7fc1);
                @memset(o.pick.slice(u8, TOPK * 4), 0x5a);
                @memset(o.wts.slice(u8, TOPK * 4), 0x5a);
            }
            for (0..2) |side| {
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                launch(e, i, &r, side);
                e.end();
                cb.commit();
                cb.wait();
            }
            const x0 = r.out[0];
            const x1 = r.out[1];
            if (!std.mem.eql(u16, x0.act.slice(u16, (TOPK + 1) * N), x1.act.slice(u16, (TOPK + 1) * N)) or !std.mem.eql(u16, x0.ye.slice(u16, (TOPK + 1) * D), x1.ye.slice(u16, (TOPK + 1) * D)) or
                !std.mem.eql(u8, x0.pick.slice(u8, TOPK * 4), x1.pick.slice(u8, TOPK * 4)) or !std.mem.eql(u8, x0.wts.slice(u8, TOPK * 4), x1.wts.slice(u8, TOPK * 4)))
            {
                std.debug.print("draw {d}, logits row {d}: BYTES DIFFER\n", .{ d, i });
                return error.BitsDiffer;
            }
            checked += (TOPK + 1) * (N + D) + 2 * TOPK;
        }
    };
    fill(&r, rnd, 0);
    for (0..8) |_| _ = try run(queue, &r, 0, layers); // clocks up
    const t0 = try time(queue, &r, 0);
    const t1 = try time(queue, &r, 1);
    const t2 = try time(queue, &r, 0);
    std.debug.print("  one row: select and experts {d:6.1} / {d:6.1} us, the select in the gate/up groups {d:6.1} us a layer\n", .{ t0, t2, t1 });
    std.debug.print("{d} outputs equal (activations, expert rows, picks, weights) over four draws\n", .{checked});
}
