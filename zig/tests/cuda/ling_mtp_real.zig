//! `ling-mtp-real DIR FIXTURE`: the real MTP layer (model.layers.72) vs HF's BailingMoeV3MTPLayer; exactness on it.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_forward_rig.zig");

const F = rig.F;

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. n * @sizeOf(T)]);
    at.* += n * @sizeOf(T);
    return out;
}

const Kit = struct {
    r: *rig.Rig,
    h: *const ling.mtp.Head,
    emb: []const u16,
    hid: []const u16,
    hd_dev: u64, // the hidden rows on the device
    D: usize,

    /// Rows lo .. hi through the MTP layer on `hd` (prompt or decode buffers), committed; its final-norm rows.
    fn pass(k: Kit, b: *const F.Buffers, hd: *F.State, lo: usize, hi: usize, picks: ?u64) ![]u16 {
        const d = k.r.gpu.d;
        const n = hi - lo;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(b.x, k.emb[lo * k.D ..].ptr, n * k.D * 2, k.r.s.handle), "cuMemcpyHtoDAsync");
        const f = k.h.forward(k.r.m, k.r.s, b);
        try k.h.step(f, hd, n, k.hd_dev + lo * k.D * 2, picks);
        try f.final(n);
        const out = try k.r.a.alloc(u16, n * k.D);
        @memcpy(std.mem.sliceAsBytes(out), try k.r.back(b.normed, n * k.D * 2));
        return out;
    }
};

pub fn run(gpu: check.Gpu, dir: []const u8, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 30));
    if (bytes.len < 20 or !std.mem.eql(u8, bytes[0..8], "TFLINGM1")) return error.BadFixture;
    var at: usize = 8;
    const dims = try take(u32, a, bytes, &at, 3);
    const T: usize = dims[0];
    const D: usize = dims[1];
    const K: usize = dims[2];
    const emb = try take(u16, a, bytes, &at, T * D);
    const hid = try take(u16, a, bytes, &at, T * D);
    const want = try take(f32, a, bytes, &at, T * D);
    const topk = try take(i32, a, bytes, &at, T * K);
    if (at != bytes.len) return error.BadFixture;
    var why: ling.config.Why = .{};
    var o = rig.W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    var m = try F.Modules.load(gpu.d, gpu.ctx);
    defer m.deinit();
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    const L = o.c.layers;
    const t0 = check.now(gpu.io);
    var w = try rig.W.load(gpu.gpa, gpu.d, s, &pack, &o.ck, o.c, &.{L}, .{ .globals = false });
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    const cap = T + 32;
    var h: ling.mtp.Head = undefined;
    try h.init(gpu.gpa, gpu.d, &m, s, &w, T, cap);
    defer h.deinit();
    const c = &w.config;
    try check.expect(D == c.hidden and K == c.topk, "fixture: {d} rows {d} wide, top {d}", .{ T, D, K });
    var r: rig.Rig = .{ .gpu = gpu, .a = a, .s = s, .m = &m, .g = &h.g };
    defer r.deinit();
    var pb = try F.Buffers.init(gpu.d, &m, &w, T, true, cap);
    defer pb.deinit();
    var db = try F.Buffers.init(gpu.d, &m, &w, F.window_rows, false, cap);
    defer db.deinit();
    const hd_dev = try r.zeros(T * D * 2);
    try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(hd_dev, hid.ptr, T * D * 2, s.handle), "cuMemcpyHtoDAsync");
    const k: Kit = .{ .r = &r, .h = &h, .emb = emb, .hid = hid, .hd_dev = hd_dev, .D = D };
    // the whole prompt against HF, picks against HF's
    const pk = try r.zeros(T * (K + 1) * 4);
    const st = try r.state(cap);
    defer st.deinit();
    const got = try k.pass(&pb, st, 0, T, pk);
    const ours = try rig.sortedPicks(a, try r.back(pk, T * (K + 1) * 4), 1, T, K);
    const theirs = try a.dupe(i32, topk);
    for (0..T) |t| std.mem.sort(i32, theirs[t * K ..][0..K], {}, std.sort.asc(i32));
    const skip = try a.alloc(bool, T);
    @memset(skip, false);
    const nf = rig.flips(ours, theirs, 1, T, 0, T, K, skip);
    const dv = dist(got, want, T, D, skip);
    std.debug.print("RESULT ling-mtp-real layer {d} ({t}): {d} rows, loaded in {d:.1} s\n", .{ L, c.quant, T, load_s });
    std.debug.print("RESULT   final-norm rows vs HF: rel L2 {e:.3}, {e:.3} without the {d} rows whose top-{d} differs, worst kept row {e:.3}\n", .{ dv[0], dv[1], nf, K, dv[2] });
    // cut prompt, decode rows, a verify window
    const sc = try r.state(cap);
    defer sc.deinit();
    const cut0 = try k.pass(&pb, sc, 0, 300, null);
    const cut1 = try k.pass(&pb, sc, 300, T, null);
    try check.sameBytes("MTP rows cut at 300 == uncut", std.mem.sliceAsBytes(cut0), std.mem.sliceAsBytes(got[0 .. 300 * D]));
    try check.sameBytes("MTP rows cut at 300 == uncut (the rest)", std.mem.sliceAsBytes(cut1), std.mem.sliceAsBytes(got[300 * D ..]));
    const P = 400;
    const N = F.window_rows;
    const sd = try r.state(cap);
    defer sd.deinit();
    const sw = try r.state(cap);
    defer sw.deinit();
    _ = try k.pass(&pb, sd, 0, P, null);
    _ = try k.pass(&pb, sw, 0, P, null);
    const dec = try a.alloc(u16, N * D);
    for (0..N) |i| @memcpy(dec[i * D ..][0..D], try k.pass(&db, sd, P + i, P + i + 1, null));
    try check.sameBytes("16-row MTP window == 16 decode rows", std.mem.sliceAsBytes(try k.pass(&db, sw, P, P + N, null)), std.mem.sliceAsBytes(dec));
    const dd = dist(dec, want[P * D ..], N, D, skip[P..]);
    std.debug.print("RESULT   decode rows {d}-{d}: rel L2 {e:.3} vs HF\n", .{ P, P + N - 1, dd[0] });
    try check.expect(dv[1] <= 1e-2 and dd[1] <= 1e-2, "the MTP layer within rel L2 1e-2 of HF (rows whose picks agree)", .{});
    check.pass("ling-mtp-real: rel L2 {e:.3} of HF ({d} flipped rows apart); cut == uncut, window == decode rows; peak {d} MiB on the device", .{ dv[1], nf, cuda.usage(false).peak >> 20 });
}

/// Relative L2 of bf16 rows against fp32 rows, all and without `skip` rows, and the worst kept row's.
fn dist(got: []const u16, want: []const f32, rows: usize, D: usize, skip: []const bool) [3]f64 {
    var dd: f64 = 0;
    var ww: f64 = 0;
    var kd: f64 = 0;
    var kw: f64 = 0;
    var worst: f64 = 0;
    for (0..rows) |t| {
        var rd: f64 = 0;
        var rw: f64 = 0;
        for (0..D) |j| {
            const e: f64 = want[t * D + j];
            const g = rig.bfOf(got[t * D + j]);
            rd += (g - e) * (g - e);
            rw += e * e;
        }
        dd += rd;
        ww += rw;
        if (!skip[t]) {
            kd += rd;
            kw += rw;
            worst = @max(worst, @sqrt(rd / rw));
        }
    }
    return .{ @sqrt(dd / ww), @sqrt(kd / kw), worst };
}
