//! x3gm v3 against v2 at the served prefill shapes on synthetic data: byte-exact outputs, then timed (x3gm3-bench).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const exl3 = dsv41.exl3;
const Gpu = check.Gpu;

const E = 384;
const slots = 6;
const D = 5120;
const I = 1152;
const widths = [_]u32{ 4, 6, 8, 10 };

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return r.s;
    }
    fn unit(r: *Rng) f32 {
        return @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
    }
};

fn halves(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, lo: f32, hi: f32) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f16, n);
    defer gpa.free(h);
    for (h) |*v| v.* = @floatCast(lo + (hi - lo) * rng.unit());
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

/// One matrix kind's words for every expert (expert e at its own width), and the int64 pointer table.
const Mats = struct {
    words: cuda.DeviceBuffer,
    table: cuda.DeviceBuffer,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, k2e: []const u32, K: usize, N: usize) !Mats {
        var total: usize = 0;
        for (k2e) |k2| total += (K / 16) * (N / 16) * 4 * k2;
        const host = try gpa.alloc(u32, total);
        defer gpa.free(host);
        for (host) |*w| w.* = @truncate(rng.next());
        const words = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        const tab = try gpa.alloc(i64, k2e.len);
        defer gpa.free(tab);
        var at: usize = 0;
        for (k2e, tab) |k2, *t| {
            t.* = @intCast(words.ptr + 4 * at);
            at += (K / 16) * (N / 16) * 4 * k2;
        }
        return .{ .words = words, .table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(tab)) };
    }

    fn deinit(m: *Mats) void {
        m.words.free();
        m.table.free();
    }
};

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

const Launch = struct {
    ctx: *const anyopaque,
    run: *const fn (ctx: *const anyopaque) anyerror!void,
};

/// The median ms of `reps` launches (each after its ticket's reset, inside the timing: both kernels pay it alike).
fn timeIt(stream: cuda.Stream, d: *const cuda.Driver, ticket: cuda.DeviceBuffer, l: Launch, reps: usize) !f32 {
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    for (0..2) |_| {
        try ticket.fill8(0, stream.handle);
        try l.run(l.ctx);
    }
    var ms: [64]f32 = undefined;
    const n = @min(reps, ms.len);
    for (0..n) |i| {
        try ticket.fill8(0, stream.handle);
        try t0.record(stream);
        try l.run(l.ctx);
        try t1.record(stream);
        try t1.synchronize();
        ms[i] = try cuda.Event.elapsedMs(t0, t1);
    }
    return median(ms[0..n]);
}

fn sameDevice(gpa: std.mem.Allocator, what: []const u8, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer, n: usize) !void {
    const x = try gpa.alloc(u8, n);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, n);
    defer gpa.free(y);
    try a.download(0, x);
    try b.download(0, y);
    try check.sameBytes(what, x, y);
}

const State = struct {
    o: exl3.Ops,
    xg: u64,
    xu: u64,
    gate: *const Mats,
    up: *const Mats,
    dn_mats: *const Mats,
    k2g: u64,
    k2d: u64,
    pl: [5]u64,
    ticket: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    xd: u64,
    y: u64,
    variant: usize = 0, // v3 variant, or none for v2
    v2: bool = true,

    fn gateup(ctx: *const anyopaque) anyerror!void {
        const s: *const State = @ptrCast(@alignCast(ctx));
        if (s.v2) return s.o.gm2Gateup(s.xg, s.xu, s.gate.table.ptr, s.up.table.ptr, s.k2g, s.pl[0], s.pl[1], s.pl[2], s.pl[3], s.pl[4], s.ticket, s.svh_g, s.svh_u, s.suh_d, s.xd, D, I, false, exl3.gmTuned2(null, null, false)[0], true, 10.0, true);
        return s.o.gm3Gateup(s.variant, s.xg, s.xu, s.gate.table.ptr, s.up.table.ptr, s.k2g, s.pl[0], s.pl[1], s.pl[2], s.pl[3], s.pl[4], s.ticket, s.svh_g, s.svh_u, s.suh_d, s.xd, D, I, false, true, 10.0, true);
    }

    fn down(ctx: *const anyopaque) anyerror!void {
        const s: *const State = @ptrCast(@alignCast(ctx));
        if (s.v2) return s.o.gm2Down(s.xd, s.dn_mats.table.ptr, s.k2d, s.pl[0], s.pl[1], s.pl[2], s.pl[3], s.pl[4], s.ticket, s.svh_d, s.y, I, D, exl3.gmTuned2(null, null, false)[1], true, true);
        return s.o.gm3Down(s.variant, s.xd, s.dn_mats.table.ptr, s.k2d, s.pl[0], s.pl[1], s.pl[2], s.pl[3], s.pl[4], s.ticket, s.svh_d, s.y, I, D, true, true);
    }
};

fn variantName(buf: []u8, v: usize) ![]const u8 {
    const x = exl3.v3_variants[v];
    return std.fmt.bufPrint(buf, "v3[{d}] tile {d},{d},{d},{d},{d} lean {d}", .{ v, x.t[0], x.t[1], x.t[2], x.t[3], x.t[4], x.lean });
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, rows_arg: ?[]const u8, reps_arg: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var rows_list: [8]usize = undefined;
    var nrows: usize = 0;
    var it = std.mem.tokenizeScalar(u8, rows_arg orelse "512,1024,2048", ',');
    while (it.next()) |t| : (nrows += 1) rows_list[nrows] = try std.fmt.parseInt(usize, t, 10);
    const reps: usize = if (reps_arg) |r| try std.fmt.parseInt(usize, r, 10) else 9;
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    const o = k.experts(stream);
    var rng: Rng = .{ .s = 0x9e3779b97f4a7c15 };

    // weights: expert e at widths[e % 4] for gate / up, widths[(e / 4) % 4] for down (every pairing present)
    var k2g: [E]u32 = undefined;
    var k2d: [E]u32 = undefined;
    for (0..E) |e| {
        k2g[e] = widths[e % 4];
        k2d[e] = widths[(e / 4) % 4];
    }
    var gate = try Mats.init(gpa, d, &rng, &k2g, D, I);
    defer gate.deinit();
    var up = try Mats.init(gpa, d, &rng, &k2g, D, I);
    defer up.deinit();
    var down = try Mats.init(gpa, d, &rng, &k2d, I, D);
    defer down.deinit();
    var k2g_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&k2g));
    defer k2g_d.free();
    var k2d_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&k2d));
    defer k2d_d.free();
    var svh_g = try halves(gpa, d, &rng, E * I, 0.01, 0.03);
    defer svh_g.free();
    var svh_u = try halves(gpa, d, &rng, E * I, 0.01, 0.03);
    defer svh_u.free();
    var suh_d = try halves(gpa, d, &rng, E * I, 0.5, 1.5);
    defer suh_d.free();
    var svh_d = try halves(gpa, d, &rng, E * D, 0.01, 0.03);
    defer svh_d.free();
    std.debug.print("x3gm3-bench: {d} experts, top-{d}, D {d}, I {d}, widths 4/6/8/10, {d} reps (median), {d} SMs\n", .{ E, slots, D, I, reps, k.sms });

    var total_v2: f64 = 0;
    var total_v3: f64 = 0;
    for ([_]bool{ false, true }) |skew| for (rows_list[0..nrows]) |R| {
        const P = R * slots;
        // routing: distinct experts a row, uniform or Zipf 0.8 (rejection over the CDF)
        const pick = try gpa.alloc(i32, P);
        defer gpa.free(pick);
        var cdf: [E]f64 = undefined;
        var acc: f64 = 0;
        for (0..E) |e| {
            acc += if (skew) 1.0 / std.math.pow(f64, @floatFromInt(e + 1), 0.8) else 1.0;
            cdf[e] = acc;
        }
        for (0..R) |r| {
            var n: usize = 0;
            while (n < slots) {
                const u = @as(f64, @floatFromInt(rng.next() >> 11)) / @as(f64, 1 << 53) * acc;
                var e: usize = 0;
                while (cdf[e] < u) e += 1;
                const ei: i32 = @intCast(e);
                if (std.mem.indexOfScalar(i32, pick[r * slots ..][0..n], ei) != null) continue;
                pick[r * slots + n] = ei;
                n += 1;
            }
        }
        var pick_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pick));
        defer pick_d.free();
        const T = exl3.planPasses(P, E, 64);
        var bufs: [5]cuda.DeviceBuffer = undefined;
        for (&bufs, [_]usize{ P, T, T, T, 1 }) |*bb, n| bb.* = try cuda.DeviceBuffer.alloc(d, 4 * n);
        defer for (&bufs) |*bb| bb.free();
        try o.gmPlan(pick_d.ptr, P, E, 64, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr);
        var xs = try halves(gpa, d, &rng, 2 * P * D, -1.0, 1.0);
        defer xs.free();
        var ticket = try cuda.DeviceBuffer.alloc(d, 16);
        defer ticket.free();
        var xd2 = try cuda.DeviceBuffer.alloc(d, 2 * P * I);
        defer xd2.free();
        var xd3 = try cuda.DeviceBuffer.alloc(d, 2 * P * I);
        defer xd3.free();
        var y2 = try cuda.DeviceBuffer.alloc(d, 4 * P * D);
        defer y2.free();
        var y3 = try cuda.DeviceBuffer.alloc(d, 4 * P * D);
        defer y3.free();
        var st: State = .{ .o = o, .xg = xs.ptr, .xu = xs.ptr + 2 * P * D, .gate = &gate, .up = &up, .dn_mats = &down, .k2g = k2g_d.ptr, .k2d = k2d_d.ptr, .pl = .{ bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr }, .ticket = ticket.ptr, .svh_g = svh_g.ptr, .svh_u = svh_u.ptr, .suh_d = suh_d.ptr, .svh_d = svh_d.ptr, .xd = xd2.ptr, .y = y2.ptr };
        try stream.synchronize();
        var np: [1]i32 = undefined;
        try bufs[4].download(0, std.mem.sliceAsBytes(&np));
        std.debug.print("rows {d} ({s} routing): {d} pairs, {d} passes\n", .{ R, if (skew) "Zipf 0.8" else "uniform", P, np[0] });

        // v2 (the served tiles): the reference outputs and times
        st.v2 = true;
        const gu2 = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.gateup }, reps);
        const dn2 = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.down }, reps);
        std.debug.print("  v2 gate/up {d:.3} ms, down {d:.3} ms, sum {d:.3}\n", .{ gu2, dn2, gu2 + dn2 });
        var best_gu: f32 = gu2;
        var best_dn: f32 = dn2;
        var nb: [64]u8 = undefined;
        var vb: [exl3.v3_variants.len]usize = undefined;
        st.v2 = false;
        st.xd = xd3.ptr;
        for (exl3.v3Of(2, 2, &vb)) |v| {
            st.variant = v;
            const t = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.gateup }, reps);
            try stream.synchronize();
            try sameDevice(gpa, "x3gm3-bench gate/up Xd vs v2", xd3, xd2, 2 * P * I);
            std.debug.print("  {s} gate/up {d:.3} ms ({d:.1}% vs v2) BITEXACT\n", .{ try variantName(&nb, v), t, 100.0 * (t - gu2) / gu2 });
            best_gu = @min(best_gu, t);
        }
        st.xd = xd2.ptr; // down reads v2's Xd
        st.y = y3.ptr;
        for (exl3.v3Of(1, 1, &vb)) |v| {
            st.variant = v;
            const t = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.down }, reps);
            try stream.synchronize();
            try sameDevice(gpa, "x3gm3-bench down Y vs v2", y3, y2, 4 * P * D);
            std.debug.print("  {s} down {d:.3} ms ({d:.1}% vs v2) BITEXACT\n", .{ try variantName(&nb, v), t, 100.0 * (t - dn2) / dn2 });
            best_dn = @min(best_dn, t);
        }
        // the v3 default (v3_default) against v2
        st.variant = exl3.v3_default.gu[1];
        st.xd = xd3.ptr;
        const gud = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.gateup }, reps);
        st.variant = exl3.v3_default.dn;
        st.xd = xd2.ptr;
        const dnd = try timeIt(stream, d, ticket, .{ .ctx = &st, .run = State.down }, reps);
        std.debug.print("  RESULT rows {d} {s}: v2 {d:.3} ms, v3 default {d:.3} ms ({d:.1}%), best variants {d:.3} ms ({d:.1}%)\n", .{ R, if (skew) "zipf" else "uniform", gu2 + dn2, gud + dnd, 100.0 * ((gud + dnd) - (gu2 + dn2)) / (gu2 + dn2), best_gu + best_dn, 100.0 * ((best_gu + best_dn) - (gu2 + dn2)) / (gu2 + dn2) });
        if (!skew) {
            total_v2 += gu2 + dn2;
            total_v3 += gud + dnd;
        }
    };
    check.pass("BITEXACT x3gm3-bench: every v3 variant equals v2 at {d} row counts x 2 routings; uniform v2 {d:.3} ms vs v3 default {d:.3} ms", .{ nrows, total_v2, total_v3 });
}
