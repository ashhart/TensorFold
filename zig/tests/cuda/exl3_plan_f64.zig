//! `tf-cuda-test exl3-plan-f64`: the 27B's EXL3 shapes against float64, under plan()'s launch and the fixed one.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const bench = @import("exl3_plan_bench.zig");
const exl3fmt = @import("core").exl3_format;
const exl3 = cuda.exl3;
const Gpu = check.Gpu;
const Rig = quant_ref_tests.RigT;

/// One launch's error against float64: the largest share of the bound, the worst column and the root mean square.
const Stats = struct { q: f64 = 0, worst: f64 = 0, col: usize = 0, rms: f64 = 0 };

pub fn run(gpu: Gpu, only: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(13);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    var bad: usize = 0;
    for (bench.shapes, bench.fixed) |sh, f| {
        if (only) |o| if (!std.mem.eql(u8, o, sh.name)) continue;
        bad += try shape(&r, &lin, sh, f.sk, f.wk);
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
    try check.expect(bad == 0, "exl3.plan.f64: {d} launches outside float64's bound", .{bad});
    check.pass("exl3.plan.f64", .{});
}

/// Decode rows (16) and prompt rows (128) on identical inputs, each plan's fp32 out against one float64 reference.
fn shape(r: *Rig, lin: *const exl3.Linear, sh: bench.Shape, sk: u32, wk: u32) !usize {
    const width = exl3fmt.Width{ .halves = @intCast(sh.halves) };
    const rows: usize = if (sh.n > 65536) 16 else 128; // the head scores decode rows and verify windows only
    const trellis = try r.a.alloc(i16, (sh.k / 16) * (sh.n / 16) * width.tileWords());
    for (trellis) |*v| v.* = r.rng.int(i16);
    const suh = try r.a.alloc(u16, sh.k);
    const svh = try r.a.alloc(u16, sh.n);
    for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    const x = try r.bfs(rows * sh.k, 1.0);
    const ref = try exl3_tests.reference(r, width, .mul1, trellis, sh.k, sh.n, x, suh, svh, rows);
    const st = exl3.strides(sh.k, sh.n, sh.halves, false);
    const base: exl3.Weight = .{ .trellis = try r.dev(std.mem.sliceAsBytes(trellis)), .suh = try r.dev(std.mem.sliceAsBytes(suh)), .svh = try r.dev(std.mem.sliceAsBytes(svh)), .k = @intCast(sh.k), .n = @intCast(sh.n), .halves = sh.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    var tuned = base;
    tuned.sk = sk;
    tuned.wk = wk;
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const xh = try r.zeros(rows * sh.k * 2);
    const z = try r.zeros(@max(exl3.Linear.zBytes(base, rows), exl3.Linear.zBytes(tuned, rows)) + 16);
    const counters = try exl3_tests.zeroed(r, exl3.Linear.counterBytes(base));
    const y = try r.zeros(rows * sh.n * 4);
    var bad: usize = 0;
    const ms: []const usize = if (rows == 16) &.{16} else &.{ 16, 128 };
    for (ms) |m| {
        var got: [2]Stats = undefined;
        var outs: [2][]u8 = undefined;
        for ([_]exl3.Weight{ base, tuned }, &got, &outs, [_][]const u8{ "plan()", "fixed" }) |w, *g, *o, name| {
            try lin.matmul(r.s, w, dx, .bf16, m, y, .f32, xh, z, counters);
            g.* = measure(try r.back(y, m * sh.n * 4), ref.want, ref.tol, m, sh.n);
            if (g.q > 1) bad += 1;
            try lin.matmul(r.s, w, dx, .bf16, m, y, .bf16, xh, z, counters);
            o.* = try r.back(y, m * sh.n * 2);
            const at = exl3.planOf(w);
            std.debug.print("RESULT exl3.plan.f64 {s} {s} m={d} {s} sk={d} wk={d}: fp32 out worst {e:.3} (column {d}), rms {e:.3}, {d:.3} of the bound\n", .{ sh.name, if (m == 16) "decode" else "prompt", m, name, at.sk, at.wk, g.worst, g.col, g.rms, g.q });
        }
        var differ: usize = 0;
        for (0..m * sh.n) |i| differ += @intFromBool(!std.mem.eql(u8, outs[0][2 * i ..][0..2], outs[1][2 * i ..][0..2]));
        std.debug.print("RESULT exl3.plan.f64 {s} m={d}: fixed / plan() fp32 rms {d:.4}, worst {d:.4}; bf16 outputs that differ {d} of {d}\n", .{ sh.name, m, got[1].rms / got[0].rms, got[1].worst / got[0].worst, differ, m * sh.n });
    }
    return bad;
}

fn measure(raw: []const u8, want: []const f64, tol: []const f64, m: usize, n: usize) Stats {
    var s: Stats = .{};
    var sq: f64 = 0;
    for (0..m * n) |i| {
        const e = @abs(@as(f64, std.mem.bytesToValue(f32, raw[4 * i ..][0..4])) - want[i]);
        s.q = @max(s.q, e / tol[i]);
        sq += e * e;
        if (e > s.worst) {
            s.worst = e;
            s.col = i % n;
        }
    }
    s.rms = @sqrt(sq / @as(f64, @floatFromInt(m * n)));
    return s;
}
