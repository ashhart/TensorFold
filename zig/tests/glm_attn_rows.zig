//! tf-glm-attn-rows: a window's dense attention in three launches against each row's glm_attn_ref.metal launches; time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const R = glm.weights.Ref;

const H = 64;
const RANK = 512;
const TOP = 2048;
const MAXR = 16;
const reps = 11; // MLA layers a round

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

const Bufs = struct { keys: mtl.Buffer, q: mtl.Buffer, s: [2]mtl.Buffer, p: [2]mtl.Buffer, o: [2]mtl.Buffer };

fn perRow(e: mtl.ComputeEncoder, x: *const glm.forward.Ctx, b: *const Bufs, side: usize, n0: u32, rows: u32) void {
    const lat = H * RANK * 2;
    const plane = H * TOP * 2;
    for (0..rows) |r| glm.forward.attendDense(x, e, R{ .buf = b.keys }, (R{ .buf = b.q }).at(r * lat), (R{ .buf = b.s[side] }).at(r * plane), (R{ .buf = b.p[side] }).at(r * plane), (R{ .buf = b.o[side] }).at(r * lat), n0 + @as(u32, @intCast(r)));
}

fn batched(e: mtl.ComputeEncoder, x: *const glm.forward.Ctx, b: *const Bufs, side: usize, n0: u32, rows: u32) void {
    glm.forward.attendRows(x, e, R{ .buf = b.keys }, R{ .buf = b.q }, R{ .buf = b.s[side] }, R{ .buf = b.p[side] }, R{ .buf = b.o[side] }, n0, rows);
}

fn run(q: mtl.Queue, comptime f: anytype, args: anytype, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |_| @call(.auto, f, .{e} ++ args);
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
    for (&t) |*v| v.* = try run(q, f, args, reps);
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
    const c: glm.config.Config = .{};
    const x: glm.forward.Ctx = .{ .k = k, .c = &c, .w = undefined, .s = undefined, .sc = undefined };
    const ref_src = try tf.frags.source(device, gpa, @embedFile("glm_attn_ref.metal"));
    const ref_lib = try mtl.Library.fromSource(device, ref_src, mtl.CompileOptions.mlx());
    var kr = k.*; // the reference kernels in the production launches
    kr.latent_scores = try mtl.Pipeline.init(device, ref_lib, "glm_latent_scores", false);
    kr.latent_values = try mtl.Pipeline.init(device, ref_lib, "glm_latent_values", false);
    kr.latent_scores_rows = try mtl.Pipeline.init(device, ref_lib, "glm_latent_scores_rows", false);
    kr.latent_values_rows = try mtl.Pipeline.init(device, ref_lib, "glm_latent_values_rows", false);
    const xr: glm.forward.Ctx = .{ .k = &kr, .c = &c, .w = undefined, .s = undefined, .sc = undefined };
    const opts = mtl.ResourceOptions.shared;
    const planes = MAXR * H * TOP;
    var b: Bufs = .{ .keys = try device.buffer((TOP + MAXR) * RANK * 2, opts), .q = try device.buffer(MAXR * H * RANK * 2, opts), .s = undefined, .p = undefined, .o = undefined };
    for (0..2) |i| {
        b.s[i] = try device.buffer(planes * 2, opts);
        b.p[i] = try device.buffer(planes * 2, opts);
        b.o[i] = try device.buffer(MAXR * H * RANK * 2, opts);
    }
    var prng = std.Random.DefaultPrng.init(0x3a);
    const r = prng.random();
    for (b.keys.slice(u16, (TOP + MAXR) * RANK)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    for (b.q.slice(u16, MAXR * H * RANK)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.25);
    const cases = [_][2]u32{ .{ 1, 1 }, .{ 1, 16 }, .{ 5, 3 }, .{ 31, 2 }, .{ 33, 4 }, .{ 240, 16 }, .{ 255, 2 }, .{ 256, 1 }, .{ 400, 1 }, .{ 400, 2 }, .{ 400, 4 }, .{ 400, 8 }, .{ 700, 3 }, .{ 1010, 16 }, .{ 1024, 2 }, .{ 1500, 8 }, .{ 2033, 16 }, .{ 2048, 1 } };
    var outputs: usize = 0;
    std.debug.print("GPU {s}: a window's dense rows in three launches against each row's reference launches ({d} layers a timing)\n", .{ device.name(), reps });
    for (cases) |cs| {
        const n0 = cs[0];
        const rows = cs[1];
        for (0..2) |i| {
            @memset(b.s[i].slice(u16, planes), 0x7fc1);
            @memset(b.p[i].slice(u16, planes), 0x7fc1);
            @memset(b.o[i].slice(u16, MAXR * H * RANK), 0x7fc1);
        }
        _ = try run(queue, perRow, .{ &xr, &b, 0, n0, rows }, 1);
        _ = try run(queue, batched, .{ &x, &b, 1, n0, rows }, 1);
        const same = std.mem.eql(u16, b.s[0].slice(u16, planes), b.s[1].slice(u16, planes)) and std.mem.eql(u16, b.p[0].slice(u16, planes), b.p[1].slice(u16, planes)) and std.mem.eql(u16, b.o[0].slice(u16, MAXR * H * RANK), b.o[1].slice(u16, MAXR * H * RANK));
        if (!same) {
            std.debug.print("keys {d}, {d} rows: BITS DIFFER\n", .{ n0, rows });
            return error.BitsDiffer;
        }
        var n: usize = 0;
        for (0..rows) |i| n += 2 * H * (n0 + i) + H * RANK;
        outputs += n;
        for (0..16) |_| _ = try run(queue, batched, .{ &xr, &b, 0, n0, rows }, reps); // clocks up
        const t0 = try time(queue, batched, .{ &xr, &b, 0, n0, rows });
        const t1 = try time(queue, batched, .{ &x, &b, 1, n0, rows });
        std.debug.print("  keys {d:4}, {d:2} rows: reference window {d:7.1} us, ours {d:7.1} us a layer ({d:.2}x)\n", .{ n0, rows, t0, t1, t0 / t1 });
    }
    std.debug.print("{d} outputs equal each row's own reference launches\n", .{outputs});
}
