//! tf-glm-kda-speed: GLM's KDA decode step on synthetic weights, the window kernel against kda_rows bit for bit.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;
const reps = 16;
const runs = 7;

const H = 64;
const D = 128;
const W = H * D;
const PS = 3 * W + 2 * D + H; // the stacked projection
const TAPS = 4;
const MAX_ROWS = 16;

fn buf(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), opts);
}

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fillBf(b: mtl.Buffer, n: usize, r: std.Random, lo: f32, hi: f32) void {
    for (b.slice(u16, n)) |*v| v.* = toBf(lo + r.float(f32) * (hi - lo));
}

fn fillF(b: mtl.Buffer, n: usize, r: std.Random, lo: f32, hi: f32) void {
    for (b.slice(f32, n)) |*v| v.* = lo + r.float(f32) * (hi - lo);
}

const Rig = struct {
    queue: mtl.Queue,
    p: mtl.Buffer,
    cs: mtl.Buffer,
    cw: mtl.Buffer,
    fb: [3]mtl.Buffer,
    gb: [3]mtl.Buffer,
    av: mtl.Buffer,
    dtb: mtl.Buffer,
    st: mtl.Buffer,
    onw: mtl.Buffer,
    y: mtl.Buffer,
    st_out: mtl.Buffer,
    cs_out: mtl.Buffer,
};

/// forward.zig's kdaStep for `rows` rows of the generated gate.
fn encode(e: mtl.ComputeEncoder, r: *const Rig, pipe: mtl.Pipeline, rows: u32, heads: u32) void {
    e.setPipeline(pipe);
    e.setBuffer(r.p, 0, 0);
    var shape = [2]i32{ @intCast(rows), @intCast(3 * heads * D + 2 * D + heads) };
    e.setBytes(std.mem.asBytes(&shape), 1);
    const ins = [_]mtl.Buffer{ r.cs, r.cw, r.fb[0], r.fb[1], r.fb[2], r.gb[0], r.gb[1], r.gb[2], r.av, r.dtb, r.st, r.onw };
    for (ins, 2..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(@as(f32, -5.0), 14);
    e.setValue(@as(f32, 1e-5), 15);
    e.setBuffer(r.y, 0, 16);
    e.setBuffer(r.st_out, 0, 17);
    e.setBuffer(r.cs_out, 0, 18);
    e.dispatchThreads(mtl.Size.of(32, 32, heads), mtl.Size.of(32, 32, 1));
}

fn submit(r: *const Rig, pipe: mtl.Pipeline, rows: u32, count: usize) !f64 {
    return submitHeads(r, pipe, rows, count, H);
}

fn submitHeads(r: *const Rig, pipe: mtl.Pipeline, rows: u32, count: usize, heads: u32) !f64 {
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..count) |_| encode(e, r, pipe, rows, heads);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(count));
}

fn time(r: *const Rig, pipe: mtl.Pipeline, rows: u32) !f64 {
    var t: [runs + 1]f64 = undefined;
    for (&t) |*v| v.* = try submit(r, pipe, rows, reps);
    const sl = t[1..];
    std.mem.sort(f64, sl, {}, std.sort.asc(f64));
    return sl[runs / 2];
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const fbw = W * D * 4 / 8; // [W, D] at 4 bits
    const fbs = W * (D / 64) * 2;
    const r = Rig{
        .queue = queue,
        .p = try buf(device, MAX_ROWS * PS * 2),
        .cs = try buf(device, (TAPS - 1) * 3 * W * 2),
        .cw = try buf(device, TAPS * 3 * W * 4),
        .fb = .{ try buf(device, fbw), try buf(device, fbs), try buf(device, fbs) },
        .gb = .{ try buf(device, fbw), try buf(device, fbs), try buf(device, fbs) },
        .av = try buf(device, H * 4),
        .dtb = try buf(device, W * 4),
        .st = try buf(device, H * D * D * 4),
        .onw = try buf(device, D * 4),
        .y = try buf(device, MAX_ROWS * W * 2),
        .st_out = try buf(device, H * D * D * 4),
        .cs_out = try buf(device, (TAPS - 1) * 3 * W * 2),
    };
    fillBf(r.p, MAX_ROWS * PS, rnd, -2, 2);
    fillBf(r.cs, (TAPS - 1) * 3 * W, rnd, -2, 2);
    fillF(r.cw, TAPS * 3 * W, rnd, -0.5, 0.5);
    for ([_]*const [3]mtl.Buffer{ &r.fb, &r.gb }) |q| {
        for (q[0].slice(u32, fbw / 4)) |*v| v.* = rnd.int(u32);
        fillBf(q[1], fbs / 2, rnd, 0.001, 0.02);
        fillBf(q[2], fbs / 2, rnd, -0.05, 0.05);
    }
    fillF(r.av, H, rnd, 0.5, 2);
    fillF(r.dtb, W, rnd, -1, 1);
    fillF(r.st, H * D * D, rnd, -0.1, 0.1);
    fillF(r.onw, D, rnd, 0.5, 1.5);
    // every row count, three draws of inputs: the window kernel's outputs, state and conv window against kda_rows'
    var cases: usize = 0;
    var failures: usize = 0;
    for (0..3) |seed| {
        fillBf(r.p, MAX_ROWS * PS, rnd, -2, 2);
        fillBf(r.cs, (TAPS - 1) * 3 * W, rnd, -2, 2);
        fillF(r.st, H * D * D, rnd, -0.1, 0.1);
        if (seed == 2) for (r.p.slice(u16, MAX_ROWS * PS), 0..) |*v, i| if (i % 97 == 0) {
            v.* = 0x8000; // -0 and large values among the inputs
        } else if (i % 89 == 0) {
            v.* = toBf(40);
        };
        for (1..MAX_ROWS + 1) |rows_| for ([_]u32{ H, H / 2 }) |heads| { // all heads, and TP2's 32
            const rows: u32 = @intCast(rows_);
            const old_k = if (heads == H) k.kda_rows else k.kda_rows_tp;
            const new_k = if (heads == H) k.kda_window else k.kda_window_tp;
            const outs = [_]mtl.Buffer{ r.y, r.st_out, r.cs_out };
            const lens = [_]usize{ rows * heads * D * 2, heads * D * D * 4, (TAPS - 1) * 3 * heads * D * 2 };
            var want: [3][]u8 = undefined;
            _ = try submitHeads(&r, old_k, rows, 1, heads);
            for (outs, lens, 0..) |b, n, i| want[i] = try a.dupe(u8, b.contents()[0..n]);
            for (outs, lens) |b, n| @memset(b.contents()[0..n], 0x5a);
            _ = try submitHeads(&r, new_k, rows, 1, heads);
            var same = true;
            for (outs, lens, 0..) |b, n, i| {
                same = same and std.mem.eql(u8, want[i], b.contents()[0..n]);
                a.free(want[i]);
            }
            cases += 1;
            if (!same) {
                failures += 1;
                std.debug.print("DIFFER: draw {d}, {d} rows, {d} heads\n", .{ seed, rows, heads });
            }
        };
    }
    std.debug.print("kda window: {d} of {d} cases give kda_rows' outputs, state and conv window\n", .{ cases - failures, cases });
    if (platform.c.getenv("GLM_KDA_TIMES") != null) for ([_]u32{ 1, 2, 3, 4, 5, 6, 8, 12, 16 }) |rows| {
        const t_old = try time(&r, k.kda_rows, rows);
        const t_new = try time(&r, k.kda_window, rows);
        std.debug.print("{d:2} rows: kda_rows {d:6.1} us, window {d:6.1} us\n", .{ rows, t_old, t_new });
    };
    if (failures > 0) std.process.exit(1);
}
