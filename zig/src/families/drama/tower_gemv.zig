//! drama_tower_gemv's qualification: a projection leaves cuBLASLt only once its one-row plan matched the kernel.
const std = @import("std");
const cuda = @import("cuda");
const Buffer = cuda.DeviceBuffer;
const tower = @import("tower.zig");
const intermediate = tower.intermediate;
const projections = tower.projections;
const Impl = tower.Impl;
const loadModule = tower.loadModule;

/// Switches a projection to drama_tower_gemv only after its own one-row plan matched the kernel bit for bit.
pub fn qualifyGemv(p: *Impl, io: std.Io, ctx: *cuda.Context, kernel_dir: []const u8) !void {
    const capability = try ctx.capability();
    const version = p.library.api.cublasLtGetVersion();
    if (capability != 121 or version != 130101) {
        std.log.info("drama tower: one-row projections on cuBLASLt: drama_tower_gemv is qualified on sm 121 with cuBLASLt 130101, not sm {d} with {d}", .{ capability, version });
        return;
    }
    p.gemv_module = try loadModule(p.gpa, io, p.driver, kernel_dir, "drama_tower_gemv.fatbin", cuda.kernels.drama_tower_gemv);
    errdefer {
        p.gemv_module.?.unload();
        p.gemv_module = null;
    }
    p.gemv = try p.gemv_module.?.function("drama_tower_gemv");
    const most = comptime blk: {
        var m: struct { weights: usize = 0, n: usize = 0, k: usize = 0 } = .{};
        for (projections) |shape| {
            m.weights = @max(m.weights, shape[0] * shape[1]);
            m.n = @max(m.n, shape[0]);
            m.k = @max(m.k, shape[1]);
        }
        break :blk m;
    };
    const host = try p.gpa.alloc(u16, @max(probe_rows * most.k, 2 * most.n));
    defer p.gpa.free(host);
    var w = try Buffer.alloc(p.driver, most.weights * 2);
    defer w.free();
    var x = try Buffer.alloc(p.driver, most.k * 2);
    defer x.free();
    var planned = try Buffer.alloc(p.driver, most.n * 2);
    defer planned.free();
    var kernel = try Buffer.alloc(p.driver, most.n * 2);
    defer kernel.free();
    // blocking: its work waits for the uploads, which may return before a pageable copy has landed
    var stream = try cuda.Stream.init(p.driver, false);
    defer stream.deinit();
    // drained before the scratch is freed or the image unloaded, on every path
    defer stream.synchronize() catch {};
    @memset(host[0..most.k], 0x3f80); // x = 1
    try x.upload(0, std.mem.sliceAsBytes(host[0..most.k]));
    try qualifyAll(&p.gemv_exact, GemvProbe{ .p = p, .stream = stream, .host = host, .w = w, .x = x, .planned = planned, .kernel = kernel });
    const e = p.gemv_exact;
    std.log.info("drama tower: one-row projections on drama_tower_gemv (1) or cuBLASLt (0): eh {d} wqkv {d} wo {d} gate_up {d} down {d} head {d}", .{
        @intFromBool(e[0]), @intFromBool(e[1]), @intFromBool(e[2]), @intFromBool(e[3]), @intFromBool(e[4]), @intFromBool(e[5]),
    });
}

/// Distinct probe rows: qualifyGemv repeats them down each weight matrix (every output row is summed alone).
const probe_rows = 256;

/// Each projection's probe result, committed to exact only once all ran, so any error leaves every plan in place.
fn qualifyAll(exact: *[projections.len]bool, probe: anytype) !void {
    var staged: [projections.len]bool = undefined;
    for (&staged, 0..) |*s, kind| s.* = try probe.bitIdentical(kind);
    exact.* = staged;
}

/// qualifyGemv's scratch, on its blocking stream.
const GemvProbe = struct {
    p: *Impl,
    stream: cuda.Stream,
    host: []u16,
    w: Buffer,
    x: Buffer,
    planned: Buffer,
    kernel: Buffer,

    /// The one-row plan against drama_tower_gemv on probe rows; different output fills, so an unwritten output fails.
    fn bitIdentical(g: GemvProbe, kind: usize) !bool {
        const n = projections[kind][0];
        const k = projections[kind][1];
        probeWeights(g.host[0 .. probe_rows * k], k);
        try g.w.upload(0, std.mem.sliceAsBytes(g.host[0 .. probe_rows * k]));
        var filled: usize = probe_rows;
        while (filled < n) {
            const more = @min(filled, n - filled);
            try g.w.copyFrom(filled * k * 2, g.w.ptr, more * k * 2, g.stream.handle);
            filled += more;
        }
        try g.planned.fill8(0xff, g.stream.handle);
        try g.kernel.fill8(0x00, g.stream.handle);
        try g.p.project(g.stream, 1, kind, g.x, g.w, g.planned);
        try g.p.gemvRow(g.stream, kind, g.x, g.w, g.kernel);
        try g.stream.synchronize();
        try g.planned.download(0, std.mem.sliceAsBytes(g.host[0..n]));
        try g.kernel.download(0, std.mem.sliceAsBytes(g.host[n .. 2 * n]));
        return std.mem.eql(u16, g.host[0..n], g.host[n .. 2 * n]);
    }
};

/// Probe rows whose x = 1 sums depend on the association order: cancelling large pairs among small terms.
fn probeWeights(w: []u16, k: usize) void {
    for (0..w.len / k) |r| {
        const h = splitmix(r);
        const m: usize = @intCast(1 + h % 2047);
        const row = w[r * k ..][0..k];
        for (0..k) |c| {
            const partner = c ^ m;
            if (partner < c) continue;
            const q = splitmix(h ^ @as(u64, c));
            const small = (q & 7) == 0;
            row[c] = probeValue(q, small);
            row[partner] = if (small) probeValue(splitmix(q), true) else row[c] ^ 0x8000;
        }
    }
}

fn probeValue(q: u64, small: bool) u16 {
    const exponent: u64 = if (small) 120 + (q >> 8) % 8 else 127 + (q >> 8) % 21;
    return @intCast((q >> 24 & 1) << 15 | exponent << 7 | (q >> 16 & 0x7f));
}

fn splitmix(z0: u64) u64 {
    var z = z0 +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/// One output of a bf16 row times x = 1 as chains fp32 fma chains joined by a tree, then cuBLASLt's epilogue.
fn probeSum(row: []const u16, chains: usize, strided: bool, halving: bool) u16 {
    var p: [32]f32 = @splat(0);
    const len = row.len / chains;
    for (0..chains) |t| for (0..len) |j| {
        const v: f32 = @bitCast(@as(u32, row[if (strided) t + j * chains else t * len + j]) << 16);
        p[t] = @mulAdd(f32, v, 1, p[t]);
    };
    if (halving) {
        var h = chains / 2;
        while (h >= 1) : (h /= 2) for (0..h) |t| {
            p[t] = p[t] + p[t + h];
        };
    } else for (1..chains) |t| {
        p[0] = p[0] + p[t];
    }
    const u: u32 = @bitCast(@mulAdd(f32, 1, p[0], 0) + 0);
    return @truncate((u + 0x7fff + (u >> 16 & 1)) >> 16);
}

test "the gemv probe separates the one-row plans' summation order from its neighbours" {
    const rows = probe_rows;
    const w = try std.testing.allocator.alloc(u16, rows * intermediate);
    defer std.testing.allocator.free(w);
    // (chains, strided, halving tree); the first is cuBLASLt's order on GB10, the others what another kernel might do
    const orders = [_]struct { usize, bool, bool }{
        .{ 16, true, true }, .{ 1, true, true }, .{ 16, true, false }, .{ 8, true, true }, .{ 32, true, true }, .{ 16, false, true },
    };
    for (projections) |shape| {
        const k = shape[1];
        probeWeights(w[0 .. rows * k], k);
        var differ: [orders.len]usize = @splat(0);
        for (0..rows) |r| {
            const row = w[r * k ..][0..k];
            const model = probeSum(row, orders[0][0], orders[0][1], orders[0][2]);
            for (orders[1..], differ[1..]) |o, *d| d.* += @intFromBool(probeSum(row, o[0], o[1], o[2]) != model);
        }
        // every other order changes at least half of these rows
        for (differ[1..]) |d| try std.testing.expect(d >= rows / 2);
    }
}

test "gemv qualification commits its decisions only when every projection was probed" {
    // Stands in for GemvProbe: fail_at injects the error any probe step would return, at that projection.
    const Fake = struct {
        fail_at: usize,
        mismatch: usize,
        probed: *usize,
        fn bitIdentical(f: @This(), kind: usize) !bool {
            if (kind == f.fail_at) return error.InjectedProbeFailure;
            f.probed.* += 1;
            return kind != f.mismatch;
        }
    };
    const none = projections.len;
    var probed: usize = 0;
    var exact: [projections.len]bool = @splat(false);
    try qualifyAll(&exact, Fake{ .fail_at = none, .mismatch = 2, .probed = &probed });
    try std.testing.expectEqual([_]bool{ true, true, false, true, true, true }, exact);
    for (0..projections.len) |kind| {
        exact = @splat(false);
        probed = 0;
        try std.testing.expectError(error.InjectedProbeFailure, qualifyAll(&exact, Fake{ .fail_at = kind, .mismatch = none, .probed = &probed }));
        try std.testing.expectEqual(kind, probed); // every projection before it passed...
        try std.testing.expectEqual(@as([projections.len]bool, @splat(false)), exact); // ...and none switched
    }
}
