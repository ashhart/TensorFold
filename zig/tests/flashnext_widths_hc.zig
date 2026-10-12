//! tf-flashnext-widths, hyper-connections: the recorded any-width kernels at each oQ width, decode rows 1 to 16.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lib = @import("flashnext_widths_lib.zig");
const fz = tf.flashnext_replay;
const variants = fz.variants;
const Buf = fz.Buf;
const Run = fz.Run;
const a = lib.a;

const W = 4 * 2560; // the streams' width: the down projection's K
const LOW = 320;
const counts = [_]usize{ 1, 2, 3, 4, 8, 9, 16 }; // the row kernels at 1-2 rows, the MMA kernels past them
const R = 16;

/// The streams and their sums of squares set so every stream's norm is exactly one: x = bf16(hn) as stored.
const Streams = struct { hn: mtl.Buffer, hd: []f64, ssp: mtl.Buffer, nw: mtl.Buffer, eps: mtl.Buffer };

fn streams(device: mtl.Device) !Streams {
    const in = try lib.Inputs.init(device, R, W, 0x4c);
    const ssp = try device.buffer(R * 10 * 4 * 4, lib.opts);
    @memset(ssp.slice(f32, R * 40), 256); // ten partial sums of 256 a stream: mean square 1, rsqrt 1
    const nw = try device.buffer(W * 4, lib.opts);
    @memset(nw.slice(f32, W), 1);
    const eps = try device.buffer(16, lib.opts);
    eps.slice(f32, 1)[0] = 0;
    return .{ .hn = in.buf, .hd = in.x, .ssp = ssp, .nw = nw, .eps = eps };
}

const Weights = struct { dw: Buf, ds: Buf, db: Buf, uw: Buf, us: Buf, ub: Buf };

fn upload(device: mtl.Device, down: lib.Mat, up: lib.Mat) !Weights {
    return .{
        .dw = .{ .b = try lib.upload(device, down.words) },
        .ds = .{ .b = try lib.upload(device, std.mem.sliceAsBytes(down.scales)) },
        .db = .{ .b = try lib.upload(device, std.mem.sliceAsBytes(down.biases)) },
        .uw = .{ .b = try lib.upload(device, up.words) },
        .us = .{ .b = try lib.upload(device, std.mem.sliceAsBytes(up.scales)) },
        .ub = .{ .b = try lib.upload(device, std.mem.sliceAsBytes(up.biases)) },
    };
}

/// One run's outputs: the down partials and the up's mixed rows and inject gates.
const Out = struct { part: mtl.Buffer, mixed: mtl.Buffer, inj: mtl.Buffer };

fn outputs(device: mtl.Device, nd: usize) !Out {
    return .{ .part = try lib.poisoned(device, 10 * R * nd * 4), .mixed = try lib.poisoned(device, R * 2560 * 2), .inj = try lib.poisoned(device, R * 4 * 2) };
}

/// Rows [start, start + rows) through the down and up roles at `consts` (none: the recorded 6/32 kernels).
fn project(r: *Run, s: Streams, w: Weights, which: []const u8, start: usize, rows: usize, nd: usize, o: Out, kd: []const variants.Const, ku: []const variants.Const) !void {
    var role: [64]u8 = undefined;
    const rb = try r.buffer(16);
    rb.slice(i32, 1)[0] = @intCast(rows);
    const rows_buf: Buf = .{ .b = rb };
    const hn: Buf = .{ .b = s.hn, .off = start * W * 2 };
    const ssp: Buf = .{ .b = s.ssp, .off = start * 40 * 4 };
    const part: Buf = .{ .b = o.part, .off = start * 10 * nd * 4 };
    const down = try std.fmt.bufPrint(&role, "qa_hc_down@{s}", .{which});
    try variants.callAs(r, down, rows, &.{ hn, ssp, .{ .b = s.nw }, w.dw, w.ds, w.db, .{ .b = s.eps }, rows_buf }, &.{part}, kd);
    const up = try std.fmt.bufPrint(&role, "qa_hc_up@{s}", .{which});
    const ins = [_]Buf{ hn, ssp, .{ .b = s.nw }, part, w.uw, w.us, w.ub, .{ .b = s.eps }, rows_buf };
    try variants.callAs(r, up, rows, &ins, &.{ .{ .b = o.mixed, .off = start * 2560 * 2 }, .{ .b = o.inj, .off = start * 4 * 2 } }, ku);
}

fn runAll(r: *Run, s: Streams, w: Weights, which: []const u8, rows: usize, nd: usize, o: Out, kd: []const variants.Const, ku: []const variants.Const) !void {
    const cb = r.queue.commandBuffer();
    r.enc = cb.compute(.serial);
    var start: usize = 0;
    while (start + rows <= R) : (start += rows) try project(r, s, w, which, start, rows, nd, o, kd, ku);
    try lib.commit(cb, r.enc);
}

/// One HC at a down and up width: recorded 6/32 bits, float64, row bits; `full`: codes past 63, no 6/32 twin.
fn check(t: *lib.Tally, r: *Run, s: Streams, which: []const u8, nd: usize, dw: [2]usize, uw: [2]usize, seed: u64, full: bool) !void {
    var label_buf: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "hc {s} down q{d}/g{d} up q{d}/g{d}{s}", .{ which, dw[0], dw[1], uw[0], uw[1], if (full) " full-range codes" else "" });
    const top: u32 = if (full) 256 else 64; // below 64: codes the recorded 6/32 kernels read too
    const down = try lib.Mat.initBelow(nd, W, @intCast(dw[0]), dw[1], seed, @min(top, @as(u32, 1) << @intCast(dw[0])));
    const up = try lib.Mat.initBelow(W, LOW, @intCast(uw[0]), uw[1], seed + 1, @min(top, @as(u32, 1) << @intCast(uw[0])));
    const w = try upload(r.device, down, up);
    const w6 = if (full) w else try upload(r.device, try down.recode(6, 32), try up.recode(6, 32));
    var cd: [2]variants.Const = undefined;
    var cu: [2]variants.Const = undefined;
    const kd = variants.width(&cd, "BITS", "GS", dw);
    const ku = variants.width(&cu, "BITS", "GS", uw);
    const one = try outputs(r.device, nd);
    try runAll(r, s, w, which, 1, nd, one, kd, ku);
    var same = true;
    for (counts) |n| {
        const got = try outputs(r.device, nd);
        try runAll(r, s, w, which, n, nd, got, kd, ku);
        const rows = R / n * n;
        if (!full) {
            const rec = try outputs(r.device, nd);
            try runAll(r, s, w6, which, n, nd, rec, &.{}, &.{});
            same = lib.sameRows(t, label, n, rec.mixed.slice(u8, R * 5120), got.mixed.slice(u8, R * 5120), rows, 5120) and same;
            same = lib.sameRows(t, label, n, rec.inj.slice(u8, R * 8), got.inj.slice(u8, R * 8), rows, 8) and same;
        }
        same = lib.sameRows(t, label, n, one.mixed.slice(u8, R * 5120), got.mixed.slice(u8, R * 5120), rows, 5120) and same;
        same = lib.sameRows(t, label, n, one.inj.slice(u8, R * 8), got.inj.slice(u8, R * 8), rows, 8) and same;
    }
    if (same) std.debug.print("RESULT {s}: {s}every row's bits at 1-16 rows\n", .{ label, if (full) "" else "the recorded 6/32 kernels' bits on the same values, and " });
    // the down partials of the one-row launches against float64: the ten K parts summed
    const terms = try down.dense();
    defer for (terms) |v| a.free(v);
    const wrong = try down.skewed(1.01).dense();
    defer for (wrong) |v| a.free(v);
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    const part = one.part.slice(f32, 10 * R * nd);
    for (0..R) |row| for (0..nd) |o| {
        var sum: f64 = 0;
        for (0..10) |k| sum += part[(row * 10 + k) * nd + o];
        const ref = lib.dot(s.hd[row * W ..][0..W], terms, o);
        worst.add(o, sum, ref[0], lib.ratio(sum, ref[0], ref[1], 32 + W / dw[1] + 24, false));
        const cref = lib.dot(s.hd[row * W ..][0..W], wrong, o);
        ctrl.add(o, sum, cref[0], lib.ratio(sum, cref[0], cref[1], 32 + W / dw[1] + 24, false));
    };
    var down_label: [160]u8 = undefined;
    const dl = try std.fmt.bufPrint(&down_label, "{s} down", .{label});
    worst.report(t, dl);
    ctrl.control(t, dl);
    try upCheck(t, r, s, which, nd, down, terms, up, kd, ku, label);
}

/// One bf16 step where fp32's sum and exp could land either side of a bf16 tie (2^-18 of p: a few fp32 steps).
fn tieStep(p: f64, got: f64) f64 {
    return if (@abs(@abs(p - got) - lib.bfStep(p) / 2) <= @abs(p) * 0x1p-18) lib.bfStep(p) else 0;
}

/// The up product against float64 through sigmoid(up) * stream 0 (other streams zero), centred so 1% scales show.
fn upCheck(t: *lib.Tally, r: *Run, s: Streams, which: []const u8, nd: usize, down: lib.Mat, dterms: [2][]f64, up0: lib.Mat, kd: []const variants.Const, ku: []const variants.Const, label: []const u8) !void {
    var up = up0;
    const gl = LOW / up.group;
    up.biases = try a.alloc(u16, up0.biases.len);
    for (0..W) |o| for (0..gl) |g| {
        var codes: f64 = 0;
        for (0..up.group) |i| codes += @floatFromInt(up.code(o, g * up.group + i));
        up.biases[o * gl + g] = lib.toBf(@floatCast(-lib.fromBf(up.scales[o * gl + g]) * codes / @as(f64, @floatFromInt(up.group))));
    };
    // stream 0's scale: a power of two, so bf16(scale x) is exact; picked on row 0 from float64 estimates
    const hd = try a.alloc(f64, R * W);
    @memset(hd, 0);
    var dsum: [LOW]f64 = undefined;
    for (0..LOW) |c| {
        var v: f64 = 0;
        for (0..2560) |i| v += dterms[0][c * W + i] * s.hd[i];
        dsum[c] = v;
    }
    var scale: f64 = 1;
    var best: f64 = std.math.inf(f64);
    var e: i32 = -14;
    while (e <= 0) : (e += 1) {
        const f = std.math.ldexp(@as(f64, 1), e);
        var accs: [256]f64 = undefined;
        for (&accs, 0..) |*acc, o| {
            acc.* = 0;
            for (0..LOW) |i| {
                const v = f * dsum[i] / 4;
                acc.* += up.deq(o * 10, i) * v / (1 + @exp(-v));
            }
            acc.* = @abs(acc.*);
        }
        std.mem.sort(f64, &accs, {}, std.sort.asc(f64));
        const miss = @abs(@log2(accs[128] + 1e-30));
        if (miss < best) {
            best = miss;
            scale = f;
        }
    }
    const hn = try r.device.buffer(R * W * 2, lib.opts);
    const hv = hn.slice(u16, R * W);
    @memset(hv, 0);
    for (0..R) |row| for (0..2560) |i| {
        hd[row * W + i] = scale * s.hd[row * W + i];
        hv[row * W + i] = lib.toBf(@floatCast(hd[row * W + i]));
    };
    const s0: Streams = .{ .hn = hn, .hd = hd, .ssp = s.ssp, .nw = s.nw, .eps = s.eps };
    const o = try outputs(r.device, nd);
    try runAll(r, s0, try upload(r.device, down, up), which, 1, nd, o, kd, ku);
    const part = o.part.slice(f32, 10 * R * nd);
    const mixed = o.mixed.slice(u16, R * 2560);
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    var act: [LOW]f64 = undefined;
    var slack: [LOW]f64 = undefined;
    const steps: f64 = @floatFromInt(32 + LOW / up.group + 16);
    for (0..R) |row| {
        for (0..LOW) |c| { // the kernel's prologue: the parts summed in fp32, bf16, a quarter, bf16(silu)
            var v: f32 = 0;
            for (0..10) |k| v += part[(row * 10 + k) * nd + c];
            const v4 = lib.bfRound(v) / 4;
            const p = v4 / (1 + @exp(-v4));
            act[c] = lib.bfRound(p);
            slack[c] = tieStep(p, act[c]);
        }
        for (0..2560) |d| for ([_]f64{ 1, 1.01 }) |skew| {
            var acc: [3]f64 = .{ 0, 0, 0 }; // the product, its magnitude, the activations' tie slack
            for (0..LOW) |i| {
                const g = d * gl + i / up.group;
                const sq = skew * lib.fromBf(up.scales[g]) * @as(f64, @floatFromInt(up.code(d, i)));
                const b = lib.fromBf(up.biases[g]);
                acc[0] += (sq + b) * act[i];
                acc[1] += (sq + @abs(b)) * @abs(act[i]);
                acc[2] += @abs(sq + b) * slack[i];
            }
            const ea = steps * lib.unit * acc[1] + acc[2];
            const sig = 1 / (1 + @exp(-acc[0]));
            const normed = hd[row * W + d];
            const want = sig * normed;
            // bf16(up), the sigmoid's slope at most 1/4, bf16(sigmoid), bf16(sigmoid times the stream)
            const tol = @abs(normed) * (0.25 * (ea + 0x1p-8 * (@abs(acc[0]) + ea)) + 0x1p-8 * sig + 1e-6) + 0x1p-8 * 1.01 * @abs(want) + 1e-30;
            const got = 4 * lib.fromBf(mixed[row * 2560 + d]);
            (if (skew == 1) &worst else &ctrl).add(d, got, want, @abs(got - want) / tol);
        };
    }
    var up_label: [160]u8 = undefined;
    const ul = try std.fmt.bufPrint(&up_label, "{s} up (stream scale 2^{d})", .{ label, std.math.log2(scale) });
    worst.report(t, ul);
    ctrl.control(t, ul);
}

pub fn run(t: *lib.Tally, r: *Run) !void {
    const s = try streams(r.device);
    var seed: u64 = 0x4c00;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        seed += 4;
        const up_group = @min(group, 64); // 320 inputs: the pack reads an up projection's group 128 as 64
        try check(t, r, s, "ahc", 324, .{ bits, group }, .{ bits, up_group }, seed, false);
        if (bits == 8) try check(t, r, s, "ahc", 324, .{ bits, group }, .{ bits, up_group }, seed + 2, true);
    };
    // the mixer (no inject rows) at the oQ2 and oQ4e widths, and a down and up of different widths
    try check(t, r, s, "mix", 320, .{ 2, 32 }, .{ 2, 32 }, 0x4d01, false);
    try check(t, r, s, "mix", 320, .{ 4, 32 }, .{ 4, 32 }, 0x4d02, false);
    try check(t, r, s, "mhc", 324, .{ 5, 64 }, .{ 4, 64 }, 0x4d03, false);
}
