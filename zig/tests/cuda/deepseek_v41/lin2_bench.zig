//! TF_DSV41_LIN2 / _LIN2_CLUSTER kernels against the originals: byte-exact outputs, then timed (lin2 [reps]).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const dense = dsv41.dense;
const Gpu = check.Gpu;

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

fn words(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize) !cuda.DeviceBuffer {
    const h = try gpa.alloc(u32, n);
    defer gpa.free(h);
    for (h) |*w| w.* = @truncate(rng.next());
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

fn zeroed(d: *const cuda.Driver, n: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, @max(n, 4));
    try b.fill8(0, null);
    return b;
}

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

const Run = struct {
    ctx: *const anyopaque,
    run: *const fn (ctx: *const anyopaque, o: dense.Ops) anyerror!void,
};

fn timeIt(stream: cuda.Stream, d: *const cuda.Driver, o: dense.Ops, r: Run, reps: usize) !f32 {
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    for (0..2) |_| try r.run(r.ctx, o);
    var ms: [64]f32 = undefined;
    const n = @min(reps, ms.len);
    for (0..n) |i| {
        try t0.record(stream);
        try r.run(r.ctx, o);
        try t1.record(stream);
        try t1.synchronize();
        ms[i] = try cuda.Event.elapsedMs(t0, t1);
    }
    return median(ms[0..n]);
}

fn bytesOf(gpa: std.mem.Allocator, b: cuda.DeviceBuffer) ![]u8 {
    const x = try gpa.alloc(u8, b.len);
    try b.download(0, x);
    return x;
}

/// One linear.cu launch's buffers (y bf16 [M, N], Z [SK, M, N], counters [8 N / 128]).
const Lin = struct {
    K: usize,
    N: usize,
    k2: u32,
    sk: usize,
    wk: usize,
    M: usize = 0,
    xh: cuda.DeviceBuffer,
    T: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    z: cuda.DeviceBuffer,
    counters: cuda.DeviceBuffer,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, K: usize, N: usize, k2: u32, sk: usize, wk: usize) !Lin {
        return .{
            .K = K,
            .N = N,
            .k2 = k2,
            .sk = sk,
            .wk = wk,
            .xh = try halves(gpa, d, rng, 32 * K, -1.0, 1.0),
            .T = try words(gpa, d, rng, (K / 16) * (N / 16) * 4 * k2),
            .svh = try halves(gpa, d, rng, N, 0.01, 0.03),
            .y = try zeroed(d, 32 * N * 2),
            .z = try zeroed(d, sk * 32 * N * 4),
            .counters = try zeroed(d, 8 * (N / 128) * 4),
        };
    }

    fn deinit(l: *Lin) void {
        inline for (.{ "xh", "T", "svh", "y", "z", "counters" }) |f| @field(l, f).free();
    }

    fn run(ctx: *const anyopaque, o: dense.Ops) anyerror!void {
        const l: *const Lin = @ptrCast(@alignCast(ctx));
        const st = dense.strides(l.K, l.k2);
        try o.linear(l.xh.ptr, l.T.ptr, st[0], st[1], l.svh.ptr, 0, l.y.ptr, .bf16, if (l.sk > 1) l.z.ptr else 0, l.counters.ptr, l.M, l.K, l.N, l.k2, l.sk, l.wk, 0);
    }
};

/// One dense3 launch of a single segment (the lanes layout's words: random, either layout reads them alike).
const Lanes = struct {
    seg: dense.Seg,
    minb: u32,
    M: usize = 0,
    xh: cuda.DeviceBuffer,
    T: cuda.DeviceBuffer,
    svh: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    z: cuda.DeviceBuffer,
    counters: cuda.DeviceBuffer,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, seg: dense.Seg, minb: u32) !Lanes {
        const K = seg.k;
        const N = seg.n;
        return .{
            .seg = seg,
            .minb = minb,
            .xh = try halves(gpa, d, rng, 32 * K, -1.0, 1.0),
            .T = try words(gpa, d, rng, (K / 16) * (N / 16) * 4 * seg.k2),
            .svh = try halves(gpa, d, rng, N, 0.01, 0.03),
            .y = try zeroed(d, 32 * N * 2),
            .z = try zeroed(d, seg.sk * 32 * N * 4),
            .counters = try zeroed(d, 8 * (N / 128) * 4),
        };
    }

    fn deinit(l: *Lanes) void {
        inline for (.{ "xh", "T", "svh", "y", "z", "counters" }) |f| @field(l, f).free();
    }

    fn run(ctx: *const anyopaque, o: dense.Ops) anyerror!void {
        const l: *const Lanes = @ptrCast(@alignCast(ctx));
        const segs = [_]dense.Seg{l.seg};
        const lay = try dense.layout(&segs, l.M);
        const bufs = [_]dense.SegBufs{.{ .T = l.T.ptr, .svh = l.svh.ptr, .y = l.y.ptr, .ld = l.seg.n }};
        const lt = lay.launches[0];
        const tab = dense.linTable(lay, lt, &segs, &bufs, .bf16, l.xh.ptr, l.z.ptr, l.counters.ptr);
        try o.lanesLinear(tab, .bf16, l.M, lt, l.minb, false);
    }
};

/// The original's and the variant's outputs of one launch, byte for byte, and the counters back at zero.
fn compare(gpu: Gpu, stream: cuda.Stream, orig: dense.Ops, two: dense.Ops, r: Run, y: cuda.DeviceBuffer, counters: cuda.DeviceBuffer, what: []const u8) !void {
    const gpa = gpu.gpa;
    try y.fill8(0, null);
    try r.run(r.ctx, orig);
    try stream.synchronize();
    const want = try bytesOf(gpa, y);
    defer gpa.free(want);
    try y.fill8(0, null); // rows past M stay zero on both sides
    try r.run(r.ctx, two);
    try stream.synchronize();
    const got = try bytesOf(gpa, y);
    defer gpa.free(got);
    try check.sameBytes(what, got, want);
    const c = try bytesOf(gpa, counters);
    defer gpa.free(c);
    for (c) |b| try check.expect(b == 0, "{s}: k-split counters not back at zero", .{what});
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, reps_arg: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const reps: usize = if (reps_arg) |r| try std.fmt.parseInt(usize, r, 10) else 15;
    if (dsv41.lin2_image.len == 0) return error.Lin2KernelNotBuilt;
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    var module = try cuda.Module.load(d, dsv41.lin2_image);
    defer module.unload();
    // the originals, and the variants with / without linear2's prefetch
    const orig: dense.Ops = .{ .f = &k.dense, .s = stream };
    var f_pf = k.dense;
    f_pf.lin2 = try dense.Lin2.resolve(module, .{ .lanes = true, .linear = true, .pf = true });
    var f_np = k.dense;
    f_np.lin2 = try dense.Lin2.resolve(module, .{ .lanes = true, .linear = true, .pf = false });
    const two_pf: dense.Ops = .{ .f = &f_pf, .s = stream };
    const two_np: dense.Ops = .{ .f = &f_np, .s = stream };
    // TF_DSV41_LIN2_CLUSTER: lanes2 plus the cluster reduction (1-16 rows the original's single pass, 17-32 lanes2's)
    var f_cl = k.dense;
    f_cl.lin2 = try dense.Lin2.resolve(module, .{ .lanes = true, .linear = true, .pf = true, .cluster = true });
    const cl: dense.Ops = .{ .f = &f_cl, .s = stream };
    var rng: Rng = .{ .s = 0x2545f4914f6cdd1d };
    const rows = [_]usize{ 17, 20, 24, 29, 32 };
    var name: [160]u8 = undefined;
    var cases: usize = 0;

    // -- bits: linear.cu, every width, CTA width and a split / no split shape --------------------------------------
    for (dense.linear_k2s) |k2| for ([_]usize{ 2, 4, 8 }) |wk| for ([_][3]usize{ .{ 1024, 1024, 1 }, .{ 2048, 512, 4 } }) |sh| {
        if ((sh[0] / 16) % (sh[2] * wk) != 0) continue;
        var l = try Lin.init(gpa, d, &rng, sh[0], sh[1], k2, sh[2], wk);
        defer l.deinit();
        for (rows) |m| {
            l.M = m;
            for ([_]dense.Ops{ two_pf, two_np }, [_][]const u8{ "pf", "no pf" }) |two, pf| {
                const what = try std.fmt.bufPrint(&name, "linear2 K2 {d} WK {d} K {d} N {d} SK {d} M {d} {s}", .{ k2, wk, sh[0], sh[1], sh[2], m, pf });
                try compare(gpu, stream, orig, two, .{ .ctx = &l, .run = Lin.run }, l.y, l.counters, what);
                cases += 1;
            }
        }
    };
    check.pass("lin2 linear2: {d} launches equal to linear_kernel byte for byte", .{cases});

    // -- bits: dense3, every width, (WK, MINB) instance, one and four k splits -------------------------------------
    cases = 0;
    for (dense.dense3_k2s) |k2| for (dense.lanes_cfgs) |c| for ([_][3]usize{ .{ 2048, 1024, 1 }, .{ 4096, 512, 4 } }) |sh| {
        const wk: usize = c[0];
        const per_warp = sh[0] / 16 / sh[2] / wk;
        if ((sh[0] / 16) % (sh[2] * wk) != 0 or per_warp % dense.groupSteps(k2) != 0) continue;
        var l = try Lanes.init(gpa, d, &rng, .{ .k = sh[0], .n = sh[1], .k2 = k2, .sk = sh[2], .wk = wk }, c[1]);
        defer l.deinit();
        for (rows) |m| {
            l.M = m;
            const what = try std.fmt.bufPrint(&name, "lanes2 K2 {d} WK {d} MINB {d} K {d} N {d} SK {d} M {d}", .{ k2, wk, c[1], sh[0], sh[1], sh[2], m });
            try compare(gpu, stream, orig, two_pf, .{ .ctx = &l, .run = Lanes.run }, l.y, l.counters, what);
            cases += 1;
        }
    };
    check.pass("lin2 lanes2: {d} launches equal to lanes_linear_kernel byte for byte", .{cases});

    // -- bits: the cluster reduction, every width and (WK, MINB), split counts 2 / 4 / 8, 1-32 rows -----------------
    cases = 0;
    const cl_rows = [_]usize{ 1, 6, 12, 16, 17, 24, 29, 32 };
    for (dense.dense3_k2s) |k2| for (dense.lanes_cfgs) |c| for ([_][3]usize{ .{ 2048, 1024, 2 }, .{ 4096, 512, 4 }, .{ 4096, 1024, 8 } }) |sh| {
        const wk: usize = c[0];
        const per_warp = sh[0] / 16 / sh[2] / wk;
        if ((sh[0] / 16) % (sh[2] * wk) != 0 or per_warp == 0 or per_warp % dense.groupSteps(k2) != 0) continue;
        var l = try Lanes.init(gpa, d, &rng, .{ .k = sh[0], .n = sh[1], .k2 = k2, .sk = sh[2], .wk = wk }, c[1]);
        defer l.deinit();
        for (cl_rows) |m| {
            l.M = m;
            const what = try std.fmt.bufPrint(&name, "lanesc K2 {d} WK {d} MINB {d} K {d} N {d} SK {d} M {d}", .{ k2, wk, c[1], sh[0], sh[1], sh[2], m });
            try compare(gpu, stream, orig, cl, .{ .ctx = &l, .run = Lanes.run }, l.y, l.counters, what);
            cases += 1;
        }
    };
    check.pass("lin2 lanesc: {d} cluster launches equal to lanes_linear_kernel byte for byte", .{cases});

    // -- time: the head (linear.cu at 6 bits, the served plan) and attention-sized dense3 launches ---------------
    std.debug.print("lin2 timing ({d} reps, median us; 'one pass' = the original at 16 rows)\n", .{reps});
    {
        const K = 4096;
        const N = 64640; // the head's columns a rank
        const p = dense.plan(K, N);
        var l = try Lin.init(gpa, d, &rng, K, N, 12, p[0], p[1]);
        defer l.deinit();
        l.M = 16;
        const one = try timeIt(stream, d, orig, .{ .ctx = &l, .run = Lin.run }, reps);
        for ([_]usize{ 20, 24, 32 }) |m| {
            l.M = m;
            const a = try timeIt(stream, d, orig, .{ .ctx = &l, .run = Lin.run }, reps);
            const b = try timeIt(stream, d, two_pf, .{ .ctx = &l, .run = Lin.run }, reps);
            const cc = try timeIt(stream, d, two_np, .{ .ctx = &l, .run = Lin.run }, reps);
            std.debug.print("  head K {d} N {d} K2 12 SK {d} WK {d} M {d}: original {d:.1} us, linear2 pf {d:.1} us, no pf {d:.1} us (one pass {d:.1})\n", .{ K, N, p[0], p[1], m, a * 1e3, b * 1e3, cc * 1e3, one * 1e3 });
        }
    }
    for ([_][2]usize{ .{ 4096, 5120 }, .{ 5120, 1024 }, .{ 2048, 4096 } }) |sh| for ([_]u32{ 6, 8 }) |k2| for (dense.lanes_cfgs) |c| {
        const p = dense.plan(sh[0], sh[1]);
        const wk: usize = c[0];
        var sk = p[0];
        while (sk > 1 and ((sh[0] / 16) % (sk * wk) != 0 or (sh[0] / 16 / sk / wk) % dense.groupSteps(k2) != 0)) sk /= 2;
        if ((sh[0] / 16) % (sk * wk) != 0 or (sh[0] / 16 / sk / wk) % dense.groupSteps(k2) != 0) continue;
        var l = try Lanes.init(gpa, d, &rng, .{ .k = sh[0], .n = sh[1], .k2 = k2, .sk = sk, .wk = wk }, c[1]);
        defer l.deinit();
        l.M = 16;
        const one = try timeIt(stream, d, orig, .{ .ctx = &l, .run = Lanes.run }, reps);
        const one_cl = if (sk > 1) try timeIt(stream, d, cl, .{ .ctx = &l, .run = Lanes.run }, reps) else one;
        std.debug.print("  lanes K {d} N {d} K2 {d} SK {d} WK {d} MINB {d} M 16: original {d:.1} us, cluster {d:.1} us\n", .{ sh[0], sh[1], k2, sk, wk, c[1], one * 1e3, one_cl * 1e3 });
        for ([_]usize{ 24, 32 }) |m| {
            l.M = m;
            const a = try timeIt(stream, d, orig, .{ .ctx = &l, .run = Lanes.run }, reps);
            const b = try timeIt(stream, d, two_pf, .{ .ctx = &l, .run = Lanes.run }, reps);
            const cc = if (sk > 1) try timeIt(stream, d, cl, .{ .ctx = &l, .run = Lanes.run }, reps) else b;
            std.debug.print("  lanes K {d} N {d} K2 {d} SK {d} WK {d} MINB {d} M {d}: original {d:.1} us, lanes2 {d:.1} us, lanes2 + cluster {d:.1} us (one pass {d:.1})\n", .{ sh[0], sh[1], k2, sk, wk, c[1], m, a * 1e3, b * 1e3, cc * 1e3, one * 1e3 });
        }
    };
}
