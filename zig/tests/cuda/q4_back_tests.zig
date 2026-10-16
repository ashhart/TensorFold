//! q4-back: the loss through a 4-bit head (cuda.train_q4's softmax and split product) against float64, 64 and 256 rows.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const q4 = cuda.train_q4;
const rig = @import("train_back_rig.zig");
const Rig = rig.Rig;
const HQ4 = rig.HQ4;
const Timer = rig.Timer;
const D = rig.D;
const eps = rig.eps;
const row_counts = rig.row_counts;
const tol_col = rig.tol_col;
const tol_l2 = rig.tol_l2;
const bf16 = rig.bf16;
const bf = rig.bf;
const rb = rig.rb;
const rf = rig.rf;
const sig = rig.sig;
const silu = rig.silu;
const siluGrad = rig.siluGrad;
const dot = rig.dot;
const axpy = rig.axpy;
const dotF = rig.dotF;
const rms = rig.rms;
const toF32 = rig.toF32;
const toBf16 = rig.toBf16;
const parallel = rig.parallel;
const mib = rig.mib;
const threads = rig.threads;

const V = 248320;

const HeadCtx = struct {
    w: *const HQ4,
    logits: []const u16,
    targets: []const u32,
    weights: []const f32,
    rows: usize,
    top: []const f64,
    total: []const f64,
    acc: [][]f64, // a thread's partial [rows, D]
};

fn headChunk(c: *const HeadCtx, i: usize, _: usize) void {
    const chunk = (V + threads - 1) / threads;
    const acc = c.acc[i];
    @memset(acc, 0);
    var wrow: [D]f64 = undefined;
    var v = i * chunk;
    while (v < @min(V, (i + 1) * chunk)) : (v += 1) {
        c.w.row(v, &wrow);
        for (0..c.rows) |r| {
            const p = @exp(bf(c.logits[r * V + v]) - c.top[r]) / c.total[r];
            const dl = (p - @as(f64, if (c.targets[r] == v) 1 else 0)) * c.weights[r];
            axpy(acc[r * D ..][0..D], dl, &wrow);
        }
    }
}

fn headTest(r: *Rig, o: q4.Ops) !void {
    const R = row_counts[row_counts.len - 1];
    const head = try r.q4(V, D, @sqrt(3.0 / @as(f64, D)));
    const logits = try r.a.alloc(u16, R * V);
    for (logits) |*l| l.* = bf16(r.uni(-3, 3));
    const targets = try r.a.alloc(u32, R);
    const weights = try r.a.alloc(f32, R);
    for (targets, weights, 0..) |*t, *w, i| {
        t.* = r.rng.uintLessThan(u32, V);
        w.* = if (i % 7 == 3) 0 else @floatCast(1.0 / @as(f64, @floatFromInt(R)) * r.uni(0.5, 1.5));
        logits[i * V + t.*] = bf16(r.uni(2, 6)); // the target a little likelier
    }
    // the softmax's float64 statistics, then dmixed = dlogits head on threads over the vocabulary
    const top = try r.a.alloc(f64, R);
    const total = try r.a.alloc(f64, R);
    for (0..R) |i| {
        top[i] = -std.math.inf(f64);
        for (logits[i * V ..][0..V]) |l| top[i] = @max(top[i], bf(l));
        total[i] = 0;
        for (logits[i * V ..][0..V]) |l| total[i] += @exp(bf(l) - top[i]);
    }
    // the loss's gradient against central differences of the float64 cross-entropy (row 0's target and others)
    {
        var num: [4]f64 = undefined;
        var want: [4]f64 = undefined;
        const ce = struct {
            fn f(row: []const u16, t: u32, at: usize, add: f64) f64 {
                var m: f64 = -std.math.inf(f64);
                for (row, 0..) |l, i| m = @max(m, bf(l) + if (i == at) add else 0);
                var s: f64 = 0;
                for (row, 0..) |l, i| s += @exp(bf(l) + (if (i == at) add else 0) - m);
                return @log(s) + m - (bf(row[t]) + if (t == at) add else 0);
            }
        };
        const row = logits[0..V];
        for ([_]usize{ targets[0], 5, 100000, V - 1 }, 0..) |at, j| {
            num[j] = (ce.f(row, targets[0], at, 1e-5) - ce.f(row, targets[0], at, -1e-5)) / 2e-5;
            want[j] = @exp(bf(row[at]) - top[0]) / total[0] - @as(f64, if (at == targets[0]) 1 else 0);
        }
        r.fd("head cross-entropy dlogits", &num, &want, 1.0 / @sqrt(@as(f64, V)));
    }
    const accs = try r.a.alloc([]f64, threads);
    for (accs) |*a| a.* = try r.a.alloc(f64, R * D);
    parallel(threads, &HeadCtx{ .w = &head, .logits = logits, .targets = targets, .weights = weights, .rows = R, .top = top, .total = total, .acc = accs }, headChunk);
    const dmixed = try r.a.alloc(f64, R * D);
    @memset(dmixed, 0);
    for (accs) |a| axpy(dmixed, 1, a);
    const before_w = r.bytes;
    const dh = try head.dev(r);
    std.debug.print("RESULT head weights: {d:.1} MiB (vocab {d})\n", .{ mib(r.bytes - before_w), V });
    for (row_counts) |rows| {
        const ld = try r.dev(u16, logits[0 .. rows * V]);
        const td = try r.dev(u32, targets[0..rows]);
        const wd = try r.dev(f32, weights[0..rows]);
        const before = r.bytes;
        const stats = try r.zeros(rows * 8);
        const dmix = try r.zeros(rows * D * 4);
        const parts = try r.zeros(q4.splitsFor(rows, D, V) * rows * D * 4);
        const scratch = r.bytes - before;
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        try o.softmax(ld, td, wd, stats, rows, V);
        try o.headBack(ld, dh, dmix, rows, parts);
        const ms_head = try tm.stop(r);
        // the loss statistics, the bf16 gradient rows and dmixed
        const st = try r.back(f32, stats, rows * 2);
        const want_st = try r.a.alloc(f64, rows * 2);
        for (0..rows) |i| {
            const lt = bf(logits[i * V + targets[i]]);
            want_st[2 * i] = @log(total[i]) + top[i] - lt;
            want_st[2 * i + 1] = @exp(lt - top[i]) / total[i];
        }
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows (loss, target probability)", .{rows}), st, want_st, 2, 1e-5, 1e-5);
        const dl = try r.back(u16, ld, rows * V);
        var most: f64 = 0;
        var big: f64 = 0;
        for (0..rows) |i| for (0..V) |v| {
            const p = @exp(bf(logits[i * V + v]) - top[i]) / total[i];
            const want = (p - @as(f64, if (targets[i] == v) 1 else 0)) * weights[i];
            most = @max(most, @abs(bf(dl[i * V + v]) - want));
            big = @max(big, @abs(want));
        };
        const dl_ok = most <= 4e-3 * big;
        r.checks += 1;
        if (!dl_ok) r.bad += 1;
        std.debug.print("RESULT head {d} rows dlogits (bf16): largest difference {e:.2} of the largest |ref| {e:.2}{s}\n", .{ rows, most / big, big, if (dl_ok) "" else "  <- FAIL" });
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows dmixed", .{rows}), try r.back(f32, dmix, rows * D), dmixed[0 .. rows * D], D, tol_col, tol_l2);
        std.debug.print("RESULT head {d} rows: {d:.3} ms softmax and head, {d:.1} MiB of gradients and scratch (+ {d:.1} MiB bf16 logits)\n", .{ rows, ms_head, mib(scratch), mib(rows * V * 2) });
    }
}

pub fn run(gpu: Gpu) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP q4-back: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_4ead);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var k = try q4.Kernels.load(gpu.d);
    defer k.deinit();
    const t0 = check.now(gpu.io);
    try headTest(&r, .{ .k = &k, .s = s });
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "q4-back: {d} of {d} checks outside tolerance", .{ r.bad, r.checks });
    check.pass("q4-back: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ r.checks, secs });
}
