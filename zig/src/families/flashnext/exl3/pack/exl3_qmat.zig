//! Flash Next EXL3 CPU oracle: the unpacked quantized matrix, its Hadamard transform and the row-parallel helper.

const std = @import("std");
const format = @import("exl3_format.zig");

// -- parallel for ------------------------------------------------------------------------------------------

/// Runs `f(ctx, i)` for i in 0..n over up to `max_threads` threads (the caller included).
pub fn parallelFor(n: usize, max_threads: usize, ctx: anytype, comptime f: anytype) void {
    const Ctx = @TypeOf(ctx);
    const W = struct {
        fn run(next: *std.atomic.Value(usize), count: usize, c: Ctx) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= count) return;
                f(c, i);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    const cpus = std.Thread.getCpuCount() catch 4;
    const want = @min(@min(cpus, max_threads), n);
    var handles: [128]std.Thread = undefined;
    var spawned: usize = 0;
    while (spawned + 1 < want and spawned < handles.len) : (spawned += 1) {
        handles[spawned] = std.Thread.spawn(.{}, W.run, .{ &next, n, ctx }) catch break;
    }
    W.run(&next, n, ctx);
    for (handles[0..spawned]) |h| h.join();
}

// -- QMat --------------------------------------------------------------------------------------------------

/// In-place orthonormal Walsh-Hadamard transform of each 128 block of `x` (Sylvester order, H[i][j] =).
pub fn fwht128(x: []f64) void {
    std.debug.assert(x.len % 128 == 0);
    const inv = 1.0 / @sqrt(128.0);
    var base: usize = 0;
    while (base < x.len) : (base += 128) {
        const b = x[base..][0..128];
        var h: usize = 1;
        while (h < 128) : (h *= 2) {
            var i: usize = 0;
            while (i < 128) : (i += 2 * h) {
                for (i..i + h) |j| {
                    const lo = b[j];
                    const hi = b[j + h];
                    b[j] = lo + hi;
                    b[j + h] = lo - hi;
                }
            }
        }
        for (b) |*v| v.* *= inv;
    }
}

pub const QMat = struct {
    k: usize,
    n: usize,
    wq: []f32,
    suh: []f32,
    svh: []f32,

    /// `threads` parallelises the trellis unpack over tile rows.
    pub fn init(gpa: std.mem.Allocator, halves: u16, trellis: []const i16, tiles_k: usize, tiles_n: usize, suh: []const f16, svh: []const f16, threads: usize) !QMat {
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        if (k % 128 != 0 or n % 128 != 0 or suh.len != k or svh.len != n) return error.Length;
        if (trellis.len != tiles_k * tiles_n * @as(usize, halves) * 8) return error.Length;
        const wq = try gpa.alloc(f32, k * n);
        errdefer gpa.free(wq);
        const su = try gpa.alloc(f32, k);
        errdefer gpa.free(su);
        const sv = try gpa.alloc(f32, n);
        errdefer gpa.free(sv);
        for (su, suh) |*d, s| d.* = @floatCast(s);
        for (sv, svh) |*d, s| d.* = @floatCast(s);
        var job: UnpackJob = .{ .halves = halves, .trellis = trellis, .tiles_n = tiles_n, .wq = wq, .n = n };
        parallelFor(tiles_k, threads, &job, UnpackJob.row);
        return .{ .k = k, .n = n, .wq = wq, .suh = su, .svh = sv };
    }

    pub fn deinit(m: *QMat, gpa: std.mem.Allocator) void {
        gpa.free(m.wq);
        gpa.free(m.suh);
        gpa.free(m.svh);
        m.* = undefined;
    }

    /// y[0..n] = x[0..k] W. `tmp` is k long.
    pub fn apply(m: QMat, x: []const f64, y: []f64, tmp: []f64) void {
        std.debug.assert(x.len == m.k and y.len == m.n and tmp.len == m.k);
        for (tmp, x, m.suh) |*d, v, s| d.* = v * s;
        fwht128(tmp);
        @memset(y, 0);
        for (0..m.k) |i| {
            const u = tmp[i];
            if (u == 0) continue;
            const row = m.wq[i * m.n ..][0..m.n];
            for (y, row) |*o, w| o.* += u * @as(f64, w);
        }
        fwht128(y);
        for (y, m.svh) |*o, s| o.* *= s;
    }

    /// Same as `apply` with the output columns [lo, hi) only (the Hadamard blocks must be whole: lo.
    pub fn applyCols(m: QMat, tmp_h: []const f64, y: []f64, lo: usize, hi: usize) void {
        std.debug.assert(lo % 128 == 0 and hi % 128 == 0 and y.len == hi - lo);
        @memset(y, 0);
        for (0..m.k) |i| {
            const u = tmp_h[i];
            if (u == 0) continue;
            const row = m.wq[i * m.n + lo ..][0 .. hi - lo];
            for (y, row) |*o, w| o.* += u * @as(f64, w);
        }
        fwht128(y);
        for (y, m.svh[lo..hi]) |*o, s| o.* *= s;
    }

    /// The transformed input (x * suh) H_K that `applyCols` consumes.
    pub fn prepare(m: QMat, x: []const f64, out: []f64) void {
        for (out, x, m.suh) |*d, v, s| d.* = v * s;
        fwht128(out);
    }
};

const UnpackJob = struct {
    halves: u16,
    trellis: []const i16,
    tiles_n: usize,
    wq: []f32,
    n: usize,
    fn row(j: *UnpackJob, tk: usize) void {
        var placement: [256]format.Placement = undefined;
        format.tilePlacement(&placement);
        const w: format.Width = .{ .halves = j.halves };
        const words = w.tileWords();
        for (0..j.tiles_n) |tn| {
            const tile = j.trellis[(tk * j.tiles_n + tn) * words ..][0..words];
            var states: [256]u32 = undefined;
            format.tileStates(w, tile, &states);
            for (0..256) |p| {
                const v: f16 = format.codebookValue(.mul1, @intCast(states[p]));
                j.wq[(tk * 16 + placement[p].row) * j.n + tn * 16 + placement[p].col] = @floatCast(v);
            }
        }
    }
};

test "fwht128 is orthonormal and equals the dense Sylvester matrix" {
    var rng = std.Random.DefaultPrng.init(3);
    var x: [256]f64 = undefined;
    for (&x) |*v| v.* = rng.random().float(f64) * 2 - 1;
    var y = x;
    fwht128(&y);
    for (0..2) |blk| {
        for (0..128) |i| {
            var s: f64 = 0;
            for (0..128) |j| {
                const sign: f64 = if (@popCount(i & j) & 1 == 1) -1 else 1;
                s += sign * x[blk * 128 + j];
            }
            try std.testing.expectApproxEqAbs(s / @sqrt(128.0), y[blk * 128 + i], 1e-12);
        }
    }
    fwht128(&y);
    for (x, y) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
}

test "QMat.apply equals the dense decoder's dequantize then matvec" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(17);
    const r = rng.random();
    for ([_]u16{ 4, 6, 10 }) |halves| {
        const tiles_k: usize = 8;
        const tiles_n: usize = 16;
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        const words = @as(usize, halves) * 8;
        const trellis = try gpa.alloc(i16, tiles_k * tiles_n * words);
        defer gpa.free(trellis);
        for (trellis) |*v| v.* = @bitCast(r.int(u16));
        const suh = try gpa.alloc(f16, k);
        defer gpa.free(suh);
        const svh = try gpa.alloc(f16, n);
        defer gpa.free(svh);
        for (suh) |*v| v.* = if (r.boolean()) 1.0 else -1.0;
        for (svh) |*v| v.* = @floatCast(0.5 + r.float(f32));
        const dense = try gpa.alloc(f64, k * n);
        defer gpa.free(dense);
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        format.dequantize(.{ .halves = halves }, .mul1, trellis, tiles_k, tiles_n, suh, svh, dense, scratch);
        var q = try QMat.init(gpa, halves, trellis, tiles_k, tiles_n, suh, svh, 4);
        defer q.deinit(gpa);
        var x: [k]f64 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f64);
        var want: [n]f64 = undefined;
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |i| s += x[i] * dense[i * n + j];
            want[j] = s;
        }
        var got: [n]f64 = undefined;
        var tmp: [k]f64 = undefined;
        q.apply(&x, &got, &tmp);
        for (want, got) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9 * (1 + @abs(a)));
        // column slices equal the whole
        var prep: [k]f64 = undefined;
        q.prepare(&x, &prep);
        var part: [128]f64 = undefined;
        q.applyCols(&prep, &part, 128, 256);
        for (part, got[128..256]) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    }
}

test "parallelFor visits every index once" {
    const Ctx = struct {
        hits: []std.atomic.Value(u8),
        fn f(c: *@This(), i: usize) void {
            _ = c.hits[i].fetchAdd(1, .monotonic);
        }
    };
    var hits: [1000]std.atomic.Value(u8) = undefined;
    for (&hits) |*h| h.* = std.atomic.Value(u8).init(0);
    var c: Ctx = .{ .hits = &hits };
    parallelFor(hits.len, 8, &c, Ctx.f);
    for (&hits) |*h| try std.testing.expectEqual(@as(u8, 1), h.load(.monotonic));
}
