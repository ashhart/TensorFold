//! glm_rows_check's KDA step (f_b, g_b made apart == the generated 4-bit dots, exact) and bf16 product (fp64) parts.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const H = 64;
const D = 128;
const W = H * D;
const PS = 3 * W + 2 * D + H;
const R = 3;
const TAPS = 4;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn submit(queue: mtl.Queue, comptime f: anytype, args: anytype) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    @call(.auto, f, .{e} ++ args);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

const Bufs = struct { p: mtl.Buffer, cs: mtl.Buffer, cw: mtl.Buffer, fb: [3]mtl.Buffer, gb: [3]mtl.Buffer, a: mtl.Buffer, dtb: mtl.Buffer, st: mtl.Buffer, onw: mtl.Buffer };
const Outs = struct { y: mtl.Buffer, st: mtl.Buffer, cs: mtl.Buffer };

fn step(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, b: Bufs, f: [3]mtl.Buffer, g: [3]mtl.Buffer, o: Outs) void {
    e.setPipeline(pipe);
    e.setBuffer(b.p, 0, 0);
    e.setBytes(std.mem.asBytes(&[2]i32{ R, PS }), 1);
    const ins = [_]mtl.Buffer{ b.cs, b.cw, f[0], f[1], f[2], g[0], g[1], g[2], b.a, b.dtb, b.st, b.onw };
    for (ins, 2..) |x, i| e.setBuffer(x, 0, i);
    e.setValue(@as(f32, -5.0), 14);
    e.setValue(@as(f32, 1e-6), 15);
    e.setBuffer(o.y, 0, 16);
    e.setBuffer(o.st, 0, 17);
    e.setBuffer(o.cs, 0, 18);
    e.dispatchThreads(mtl.Size.of(32, 32, H), mtl.Size.of(32, 32, 1));
}

fn products(e: mtl.ComputeEncoder, p: *const wide.Pipes, fq: affine.Affine, gq: affine.Affine, proj: mtl.Buffer, f: mtl.Buffer, g: mtl.Buffer) void {
    const in: affine.Ref = .{ .buf = proj };
    wide.dense(p, e, fq, .{ .in = in.at(3 * W * 2), .ldx = PS, .out = .{ .buf = f }, .ldo = W, .rows = R });
    wide.dense(p, e, gq, .{ .in = in.at((3 * W + D) * 2), .ldx = PS, .out = .{ .buf = g }, .ldo = W, .rows = R });
}

/// kda_rows on f_b, g_b products made apart == the generated 4-bit dots: every code, dyadic scales and biases, exact.
pub fn check(p: *const wide.Pipes, device: mtl.Device, queue: mtl.Queue) !usize {
    const kernels = tf.glm.kernels;
    const gen_lib = try mtl.Library.fromSource(device, comptime kernels.generatedSource("kda_rows"), mtl.CompileOptions.mlx());
    defer gen_lib.deinit();
    const generated = try mtl.Pipeline.init(device, gen_lib, comptime kernels.generatedFunction("kda_rows"), false);
    defer generated.deinit();
    var prng = std.Random.DefaultPrng.init(4242);
    const r = prng.random();
    const mk = struct {
        fn of(dev: mtl.Device, bytes: usize) !mtl.Buffer {
            return dev.buffer(bytes, opts);
        }
    }.of;
    var b: Bufs = undefined;
    b.p = try mk(device, R * PS * 2);
    b.cs = try mk(device, (TAPS - 1) * 3 * W * 2);
    b.cw = try mk(device, TAPS * 3 * W * 4);
    for (&b.fb, &b.gb, [_]usize{ W * D / 2, W * 2 * 2, W * 2 * 2 }) |*f, *g, n| {
        f.* = try mk(device, n);
        g.* = try mk(device, n);
    }
    b.a = try mk(device, H * 4);
    b.dtb = try mk(device, W * 4);
    b.st = try mk(device, H * D * D * 4);
    b.onw = try mk(device, D * 4);
    defer for ([_]mtl.Buffer{ b.p, b.cs, b.cw, b.fb[0], b.fb[1], b.fb[2], b.gb[0], b.gb[1], b.gb[2], b.a, b.dtb, b.st, b.onw }) |x| x.deinit();
    const pv = b.p.slice(u16, R * PS);
    for (pv, 0..) |*v, i| {
        const c = i % PS;
        const small = (c >= 3 * W and c < 3 * W + 2 * D); // f_a and g_a: small integers, the dots exact in any order
        v.* = toBf(if (small) @floatFromInt(r.intRangeAtMost(i32, -2, 2)) else (r.float(f32) - 0.5) * 2);
    }
    for (b.cs.slice(u16, (TAPS - 1) * 3 * W)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    for (b.cw.slice(f32, TAPS * 3 * W)) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
    for ([_]mtl.Buffer{ b.fb[0], b.gb[0] }) |x| for (x.slice(u8, W * D / 2)) |*v| {
        v.* = r.int(u8); // every 4-bit code
    };
    for ([_]mtl.Buffer{ b.fb[1], b.gb[1] }) |x| for (x.slice(u16, W * 2)) |*v| {
        v.* = toBf(std.math.ldexp(@as(f32, 1), -r.intRangeAtMost(i32, 5, 8))); // a scale a group
    };
    for ([_]mtl.Buffer{ b.fb[2], b.gb[2] }) |x| for (x.slice(u16, W * 2)) |*v| {
        const m = r.intRangeAtMost(i32, 1, 8);
        v.* = toBf(@as(f32, @floatFromInt(if (r.boolean()) m else -m)) / 64); // a nonzero bias a group
    };
    for (b.a.slice(f32, H)) |*v| v.* = -r.float(f32);
    for (b.dtb.slice(f32, W)) |*v| v.* = (r.float(f32) - 0.5) * 0.1;
    for (b.st.slice(f32, H * D * D)) |*v| v.* = (r.float(f32) - 0.5) * 0.1;
    for (b.onw.slice(f32, D)) |*v| v.* = 1 + (r.float(f32) - 0.5) * 0.1;
    var outs: [2]Outs = undefined;
    for (&outs) |*o| o.* = .{ .y = try mk(device, R * W * 2), .st = try mk(device, H * D * D * 4), .cs = try mk(device, (TAPS - 1) * 3 * W * 2) };
    defer for (outs) |o| for ([_]mtl.Buffer{ o.y, o.st, o.cs }) |x| x.deinit();
    try submit(queue, step, .{ generated, b, b.fb, b.gb, outs[0] }); // the generated 4-bit dots
    const fq: affine.Affine = .{ .w = .{ .buf = b.fb[0] }, .s = .{ .buf = b.fb[1] }, .b = .{ .buf = b.fb[2] }, .n = W, .k = D, .bits = 4, .group = 64 };
    var gq = fq;
    gq.w = .{ .buf = b.gb[0] };
    gq.s = .{ .buf = b.gb[1] };
    gq.b = .{ .buf = b.gb[2] };
    const f = try mk(device, 16 * W * 2);
    defer f.deinit();
    const g = try mk(device, 16 * W * 2);
    defer g.deinit();
    try submit(queue, products, .{ p, fq, gq, b.p, f, g });
    try submit(queue, step, .{ p.kda_pre, b, [3]mtl.Buffer{ f, f, f }, [3]mtl.Buffer{ g, g, g }, outs[1] });
    var same: usize = 0;
    for ([_][2][]const u8{ .{ std.mem.sliceAsBytes(outs[0].y.slice(u16, R * W)), std.mem.sliceAsBytes(outs[1].y.slice(u16, R * W)) }, .{ outs[0].st.slice(u8, H * D * D * 4), outs[1].st.slice(u8, H * D * D * 4) }, .{ outs[0].cs.slice(u8, (TAPS - 1) * 3 * W * 2), outs[1].cs.slice(u8, (TAPS - 1) * 3 * W * 2) } }) |pair| {
        if (!std.mem.eql(u8, pair[0], pair[1])) return error.KdaPreDiffers;
        same += pair[0].len;
    }
    var nonzero: usize = 0;
    for (outs[0].y.slice(u16, R * W)) |v| nonzero += @intFromBool(v & 0x7fff != 0);
    if (nonzero < R * W / 2) return error.KdaOutputsMostlyZero;
    std.debug.print("kda_rows on f_b, g_b made apart == the generated 4-bit dots: {d} rows, outputs, state and conv window byte-equal ({d} bytes)\n", .{ R, same });
    return R * W;
}

/// kda_rows with f_b, g_b made inside (`.fused`) == made apart (`.pre`) at every width, g64: Y, state, conv equal.
pub fn fusedCheck(p: *const wide.Pipes, device: mtl.Device, queue: mtl.Queue) !usize {
    var prng = std.Random.DefaultPrng.init(4343);
    const r = prng.random();
    const mk = struct {
        fn of(dev: mtl.Device, bytes: usize) !mtl.Buffer {
            return dev.buffer(bytes, opts);
        }
    }.of;
    var b: Bufs = undefined;
    b.p = try mk(device, R * PS * 2);
    b.cs = try mk(device, (TAPS - 1) * 3 * W * 2);
    b.cw = try mk(device, TAPS * 3 * W * 4);
    b.a = try mk(device, H * 4);
    b.dtb = try mk(device, W * 4);
    b.st = try mk(device, H * D * D * 4);
    b.onw = try mk(device, D * 4);
    const f = try mk(device, 16 * W * 2);
    const g = try mk(device, 16 * W * 2);
    defer for ([_]mtl.Buffer{ b.p, b.cs, b.cw, b.a, b.dtb, b.st, b.onw, f, g }) |x| x.deinit();
    for (b.p.slice(u16, R * PS)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    for (b.cs.slice(u16, (TAPS - 1) * 3 * W)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    for (b.cw.slice(f32, TAPS * 3 * W)) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
    for (b.a.slice(f32, H)) |*v| v.* = -r.float(f32);
    for (b.dtb.slice(f32, W)) |*v| v.* = (r.float(f32) - 0.5) * 0.1;
    for (b.st.slice(f32, H * D * D)) |*v| v.* = (r.float(f32) - 0.5) * 0.1;
    for (b.onw.slice(f32, D)) |*v| v.* = 1 + (r.float(f32) - 0.5) * 0.1;
    var outs: [2]Outs = undefined;
    for (&outs) |*o| o.* = .{
        .y = try mk(device, R * W * 2),
        .st = try mk(device, H * D * D * 4),
        .cs = try mk(device, (TAPS - 1) * 3 * W * 2),
    };
    defer for (outs) |o| for ([_]mtl.Buffer{ o.y, o.st, o.cs }) |x| x.deinit();
    var same: usize = 0;
    for (affine.row_bits, 0..) |bits, i| {
        var wq: [2]affine.Affine = undefined;
        var wb: [2][3]mtl.Buffer = undefined;
        for (&wq, &wb) |*q, *x| {
            x.* = .{
                try mk(device, W * D * @as(usize, bits) / 8),
                try mk(device, W * 2 * 2),
                try mk(device, W * 2 * 2),
            };
            r.bytes(x[0].slice(u8, W * D * @as(usize, bits) / 8)); // every code
            for (x[1].slice(u16, W * 2)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
            for (x[2].slice(u16, W * 2)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
            q.* = .{
                .w = .{ .buf = x[0] },
                .s = .{ .buf = x[1] },
                .b = .{ .buf = x[2] },
                .n = W,
                .k = D,
                .bits = bits,
                .group = 64,
            };
        }
        defer for (wb) |x| for (x) |y| y.deinit();
        if (bits != 4 and wide.kdaGate(
            p,
            wq[0],
            wq[1],
        ) != .fused) return error.NotFused; // 4/64 takes the generated dots
        try submit(queue, products, .{ p, wq[0], wq[1], b.p, f, g });
        try submit(queue, step, .{ p.kda_pre, b, [3]mtl.Buffer{ f, f, f }, [3]mtl.Buffer{ g, g, g }, outs[0] });
        try submit(queue, step, .{ p.kda_fg[i], b, wb[0], wb[1], outs[1] });
        for ([_][2][]const u8{
            .{ std.mem.sliceAsBytes(outs[0].y.slice(u16, R * W)), std.mem.sliceAsBytes(outs[1].y.slice(u16, R * W)) },
            .{ outs[0].st.slice(u8, H * D * D * 4), outs[1].st.slice(u8, H * D * D * 4) },
            .{ outs[0].cs.slice(u8, (TAPS - 1) * 3 * W * 2), outs[1].cs.slice(u8, (TAPS - 1) * 3 * W * 2) },
        }) |pair| {
            if (!std.mem.eql(u8, pair[0], pair[1])) {
                std.debug.print("kda_rows fused q{d} differs from .pre\n", .{bits});
                return error.KdaFusedDiffers;
            }
            same += pair[0].len;
        }
        var nonzero: usize = 0;
        for (outs[1].y.slice(u16, R * W)) |v| nonzero += @intFromBool(v & 0x7fff != 0);
        if (nonzero < R * W / 2) return error.KdaOutputsMostlyZero;
    }
    std.debug.print(
        "kda_rows with f_b, g_b made inside == made apart (.pre) at q2..q8 g64: {d} rows, " ++
            "outputs, state and conv window byte-equal ({d} bytes)\n",
        .{ R, same },
    );
    return same / 4;
}

/// The bf16 product (an unquantized eh_proj) against fp64 per column, and a row alone == the row in a window.
pub fn bf16Check(p: *const wide.Pipes, device: mtl.Device, queue: mtl.Queue, worst: *f64) !usize {
    const n = 40;
    const k = 8192;
    const rows = 5;
    var prng = std.Random.DefaultPrng.init(99);
    const r = prng.random();
    const w = try device.buffer(n * k * 2, opts);
    defer w.deinit();
    const x = try device.buffer(rows * k * 2, opts);
    defer x.deinit();
    for (w.slice(u16, n * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.05);
    for (x.slice(u16, rows * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    const all = try device.buffer(rows * n * 2, opts);
    defer all.deinit();
    const one = try device.buffer(n * 2, opts);
    defer one.deinit();
    const run = struct {
        fn f(e: mtl.ComputeEncoder, pp: *const wide.Pipes, wb: mtl.Buffer, in: affine.Ref, out: mtl.Buffer, m: u32) void {
            wide.bf16(pp, e, .{ .buf = wb }, n, k, .{ .in = in, .ldx = k, .out = .{ .buf = out }, .ldo = n, .rows = m });
        }
    }.f;
    try submit(queue, run, .{ p, w, affine.Ref{ .buf = x }, all, rows });
    var values: usize = 0;
    for (0..rows) |i| {
        try submit(queue, run, .{ p, w, (affine.Ref{ .buf = x }).at(i * k * 2), one, 1 });
        if (!std.mem.eql(u16, one.slice(u16, n), all.slice(u16, rows * n)[i * n ..][0..n])) return error.Bf16WindowDiffers;
        for (0..n) |c| {
            var y: f64 = 0;
            var terms: f64 = 0;
            for (0..k) |j| {
                const t = fromBf(x.slice(u16, rows * k)[i * k + j]) * fromBf(w.slice(u16, n * k)[c * k + j]);
                y += t;
                terms += @abs(t);
            }
            const bound = std.math.ldexp(@abs(y), -8) + @as(f64, k + 32) * 0x1p-24 * terms;
            const err = @abs(fromBf(one.slice(u16, n)[c]) - y);
            worst.* = @max(worst.*, err / bound);
            if (err > bound) return error.Bf16PastBound;
            values += 1;
        }
    }
    std.debug.print("bf16 [{d}, {d}]: {d} values vs fp64 within the fp32-sum bound; each row alone == the row in a {d}-row window\n", .{ n, k, values, rows });
    return values;
}

/// The 6- and 8-bit embedding's gathers vs fp64 within two bf16 roundings; a neighbouring group's bias must be seen.
pub fn embedCheck(p: *const wide.Pipes, device: mtl.Device, queue: mtl.Queue, worst: *f64) !usize {
    const V = 40;
    const width = 4096;
    const ids = [_]u32{ 7, 0, 39, 7, 21 };
    var prng = std.Random.DefaultPrng.init(808);
    const r = prng.random();
    var values: usize = 0;
    for ([_]u8{ 6, 8 }, 0..) |bits, pi| {
        const rb: usize = width * @as(usize, bits) / 8;
        const G = width / 64;
        const t = try device.buffer(V * rb, opts);
        defer t.deinit();
        const s = try device.buffer(V * G * 2, opts);
        defer s.deinit();
        const b = try device.buffer(V * G * 2, opts);
        defer b.deinit();
        const idb = try device.buffer(ids.len * 4, opts);
        defer idb.deinit();
        const out = try device.buffer(ids.len * width * 2, opts);
        defer out.deinit();
        r.bytes(t.slice(u8, V * rb));
        for (s.slice(u16, V * G)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (b.slice(u16, V * G)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.2);
        @memcpy(idb.slice(u32, ids.len), &ids);
        const run = struct {
            fn f(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, bs: [5]mtl.Buffer, per: u32) void {
                e.setPipeline(pipe);
                for (bs[0..4], 0..) |x, i| e.setBuffer(x, 0, i);
                e.setBytes(std.mem.asBytes(&[1]i32{width}), 4);
                e.setBuffer(bs[4], 0, 5);
                e.dispatchThreads(mtl.Size.of(width / per, ids.len, 1), mtl.Size.of(256, 1, 1));
            }
        }.f;
        try submit(queue, run, .{ p.embed[pi], [5]mtl.Buffer{ idb, t, s, b, out }, @as(u32, if (bits == 6) 4 else 1) });
        var caught: usize = 0;
        for (ids, 0..) |id, i| for (0..width) |j| {
            const g = j / 64;
            const q: f64 = @floatFromInt(@import("glm_rows_classes.zig").code(t.slice(u8, V * rb)[id * rb ..][0..rb], j, bits));
            const sq = fromBf(s.slice(u16, V * G)[id * G + g]) * q;
            const bias = fromBf(b.slice(u16, V * G)[id * G + g]);
            const got = fromBf(out.slice(u16, ids.len * width)[i * width + j]);
            const bound = std.math.ldexp(2 * @abs(sq) + @abs(bias), -8) + std.math.ldexp(@abs(sq), -16);
            const err = @abs(got - (sq + bias));
            worst.* = @max(worst.*, err / bound);
            if (err > bound) return error.EmbedPastBound;
            const other = fromBf(b.slice(u16, V * G)[id * G + (g ^ 1)]);
            caught += @intFromBool(@abs(got - (sq + other)) > bound);
            values += 1;
        };
        if (caught == 0) return error.ControlNotCaught;
        std.debug.print("embedding q{d} g64: {d} gathered values vs fp64 within two bf16 roundings; a neighbour group's bias caught in {d}\n", .{ bits, ids.len * width, caught });
    }
    return values;
}
