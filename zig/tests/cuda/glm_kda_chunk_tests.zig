//! glm_kda_chunk.cu on synthetic GLM rows: float64 bounds, the wide chain's saves, its replay, resumed vs fresh.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const f64r = @import("glm_kda_f64.zig");
const rig = @import("glm_kda_rig.zig");
const ref = @import("glm_ref_tests.zig");
const Rig = rig.Rig;
const kda = rig.kda;
const dk = rig.dk;

const heads = 8;
const n = heads * dk;
const p_stride = 3 * n + 64;
const layout: f64r.Layout = .{ .heads = heads, .p_stride = p_stride, .b_off = 3 * n, .lower = -5, .eps = 1e-6 };

fn uniform(r: *Rig, count: usize, lo: f32, hi: f32) ![]f32 {
    const out = try r.a.alloc(f32, count);
    for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
    return out;
}

/// glm_ref_kda's synthetic inputs (its scales) at `heads` heads, and the chain over them on the device.
const Case = struct {
    in: f64r.Inputs,
    c: kda.Chain,

    fn make(r: *Rig, rows: usize) !Case {
        const in: f64r.Inputs = .{ .p = try r.bfs(rows * p_stride, 1.0), .a = try r.bfs(rows * n, 2.0), .g = try r.bfs(rows * n, 2.0), .cs = try r.bfs(3 * 3 * n, 1.0), .cw = try uniform(r, 3 * n * 4, -0.5, 0.5), .state = try uniform(r, n * dk, -0.05, 0.05), .a_log = try uniform(r, heads, -1, 1), .dt_bias = try uniform(r, n, -0.5, 0.5), .norm = try r.bfs(dk, 1.0) };
        const B = std.mem.sliceAsBytes;
        const c: kda.Chain = .{ .p = try r.dev(B(in.p)), .p_stride = p_stride, .b_off = 3 * n, .a = try r.dev(B(in.a)), .a_stride = n, .g = try r.dev(B(in.g)), .g_stride = n, .conv_state = try r.dev(B(in.cs)), .conv_w = try r.dev(B(in.cw)), .state_in = try r.dev(B(in.state)), .a_log = try r.dev(B(in.a_log)), .dt_bias = try r.dev(B(in.dt_bias)), .norm_w = try r.dev(B(in.norm)), .eps = 1e-6, .lower = -5, .heads = heads };
        return .{ .in = in, .c = c };
    }

    /// The chain from row `lo` on: its rows, and the conv window of rows lo - 3 .. lo - 1 (lo >= 3) on the device.
    fn from(k: Case, r: *Rig, lo: usize, state_in: u64) !kda.Chain {
        const cs = try r.a.alloc(u16, 3 * 3 * n);
        for (0..3) |j| @memcpy(cs[j * 3 * n ..][0 .. 3 * n], k.in.p[(lo - 3 + j) * p_stride ..][0 .. 3 * n]);
        var c = k.c;
        c.p += lo * p_stride * 2;
        c.a += lo * n * 2;
        c.g += lo * n * 2;
        c.conv_state = try r.dev(std.mem.sliceAsBytes(cs));
        c.state_in = state_in;
        return c;
    }
};

/// Worst fractions of the end-to-end bound (out, state) and of the recurrence's from the path's saves (y, state).
fn bounds(r: *Rig, k: Case, o: rig.Out, rows: usize, e2e: f64r.Want) ![4]f64 {
    const rec = try f64r.recurrence(r.a, heads, k.in.state, try rig.saved(r, o, heads), rows);
    const out = try rig.wide64(r, o.out, rows * n, true);
    const st = try rig.wide64(r, o.state, n * dk, false);
    const y = try rig.wide64(r, o.y, rows * n, true);
    return .{ f64r.fraction(out, e2e.out, e2e.out_tol), f64r.fraction(st, e2e.state, e2e.state_tol), f64r.fraction(y, rec.out, rec.out_tol), f64r.fraction(st, rec.state, rec.state_tol) };
}

fn rowsCheck(r: *Rig, m: *const rig.Mods, rows: usize) !f64 {
    const k = try Case.make(r, rows);
    const wide = try rig.Out.alloc(r, heads, rows);
    const chunk = try rig.Out.alloc(r, heads, rows);
    try rig.run(m, r.s, k.c, rows, wide, .wide);
    try rig.run(m, r.s, k.c, rows, chunk, .chunked);
    var name: [128]u8 = undefined;
    const P = std.fmt.bufPrint;
    for ([_]u64{ wide.q, wide.k, wide.v, wide.g, wide.b }, [_]u64{ chunk.q, chunk.k, chunk.v, chunk.g, chunk.b }, [_]usize{ 4 * n, 4 * n, 2 * n, 4 * n, 4 * heads }, [_][]const u8{ "q", "k", "v", "decay", "beta" }) |a, b, size, what|
        try ref.sameRows(r, try P(&name, "KDA chunked {d} rows: saved {s} equal the wide chain's", .{ rows, what }), a, b, size * rows);
    const replayed = try r.zeros(n * dk * 4);
    try kda.chunkReplay(m.ck, r.s, k.c.state_in, chunk.k, chunk.v, chunk.g, chunk.b, rows, heads, replayed);
    try ref.sameRows(r, try P(&name, "KDA chunked {d} rows: the chunked replay rebuilds its state", .{rows}), chunk.state, replayed, n * dk * 4);
    const e2e = try f64r.chain(r.a, layout, k.in, rows);
    const bc = try bounds(r, k, chunk, rows, e2e);
    const bw = try bounds(r, k, wide, rows, e2e);
    const dout = rig.diff(try rig.wide64(r, chunk.out, rows * n, true), try rig.wide64(r, wide.out, rows * n, true));
    const dst = rig.diff(try rig.wide64(r, chunk.state, n * dk, false), try rig.wide64(r, wide.state, n * dk, false));
    std.debug.print("RESULT KDA {d} rows x {d} heads, worst fraction of float64's bound, chunked (wide): end to end out {d:.3} ({d:.3}), state {d:.4} ({d:.4}); " ++
        "recurrence from the saves y {d:.3} ({d:.3}), state {d:.3} ({d:.3})\n", .{ rows, heads, bc[0], bw[0], bc[1], bw[1], bc[2], bw[2], bc[3], bw[3] });
    std.debug.print("RESULT KDA {d} rows: chunked vs wide, out {d} of {d} bf16 differ (max {e:.2} of max |out|), state {d} of {d} fp32 differ (max {e:.2} of max |S|)\n", .{ rows, dout.count, rows * n, dout.rel, dst.count, n * dk, dst.rel });
    var worst: f64 = 0;
    for (bc ++ bw) |x| worst = @max(worst, x);
    try check.expect(worst <= 1, "KDA {d} rows: chunked and wide within float64's bounds (worst {d:.3})", .{ rows, worst });
    return worst;
}

/// 300 rows resumed at `at` (second call from the first's state and conv window) against one call: equal bytes?
fn cut(r: *Rig, m: *const rig.Mods, at: usize) !bool {
    const rows = 300;
    const k = try Case.make(r, rows);
    const whole = try rig.Out.alloc(r, heads, rows);
    try rig.run(m, r.s, k.c, rows, whole, .chunked);
    const parts = try rig.Out.alloc(r, heads, rows);
    try rig.run(m, r.s, k.c, at, parts, .chunked);
    var second = parts.from(heads, at);
    second.state = try r.zeros(n * dk * 4);
    try rig.run(m, r.s, try k.from(r, at, parts.state), rows - at, second, .chunked);
    const out_same = std.mem.eql(u8, try r.back(whole.out, rows * n * 2), try r.back(parts.out, rows * n * 2));
    const st_same = std.mem.eql(u8, try r.back(whole.state, n * dk * 4), try r.back(second.state, n * dk * 4));
    const d = rig.diff(try rig.wide64(r, second.state, n * dk, false), try rig.wide64(r, whole.state, n * dk, false));
    const dy = rig.diff(try rig.wide64(r, parts.out, rows * n, true), try rig.wide64(r, whole.out, rows * n, true));
    std.debug.print("RESULT KDA chunked 300 rows resumed at {d} vs fresh: outputs equal {} ({d} bf16 differ, max {e:.2} of max |out|), state equal {} ({d} fp32 differ, max {e:.2} of max |S|)\n", .{ at, out_same, dy.count, dy.rel, st_same, d.count, d.rel });
    return out_same and st_same;
}

/// `glm-kda-chunk`.
pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(804);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var m = try rig.Mods.load(gpu.d);
    defer m.unload();
    var worst: f64 = 0;
    for ([_]usize{ 1, 15, 16, 17, 70, 300 }) |rows| worst = @max(worst, try rowsCheck(&r, &m, rows));
    for ([_]usize{ 16, 48, 160 }) |at| try check.expect(try cut(&r, &m, at), "KDA chunked: a cut at {d} rows keeps one call's bits", .{at});
    var off = false;
    for ([_]usize{ 37, 100, 255 }) |at| off = off or try cut(&r, &m, at);
    check.pass("KDA chunked {d} heads: within float64's bounds (worst {d:.3}), saves equal the wide chain's, chunked replay rebuilds the state, cuts at 16/48/160 == uncut (resumed at 37/100/255: {s})", .{ heads, worst, if (off) "equal somewhere" else "differ, as expected" });
}
