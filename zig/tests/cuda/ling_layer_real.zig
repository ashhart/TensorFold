//! `ling-layer-real DIR FIXTURE [DUMP]`: one real Ling-3.1 layer through the graph vs HF's output; exactness on it.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_forward_rig.zig");

const F = rig.F;
const Rig = rig.Rig;

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. n * @sizeOf(T)]);
    at.* += n * @sizeOf(T);
    return out;
}

/// Rows lo .. hi of x through the stack on `st` (prompt or decode buffers), committed; the output rows.
fn pass(r: *Rig, b: *const F.Buffers, st: *F.State, x: []const u16, D: usize, lo: usize, hi: usize, picks: ?u64) ![]u16 {
    const d = r.gpu.d;
    const n = hi - lo;
    try d.check(d.api.cuMemcpyHtoDAsync_v2(b.x, x[lo * D ..].ptr, n * D * 2, r.s.handle), "cuMemcpyHtoDAsync");
    const f = r.fwd(b, picks);
    try f.stack(&.{.{ .st = st, .row0 = 0, .rows = n }});
    const out = try r.a.alloc(u16, n * D);
    @memcpy(std.mem.sliceAsBytes(out), try r.back(b.x, n * D * 2));
    try f.commit(st, n);
    return out;
}

/// Relative L2 of bf16 rows against fp32 rows, all and without `skip` rows, and the worst row's.
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

/// `ling-layer-real DIR FIXTURE` (oracle/ling_layer.py's file for layer L of the checkpoint at DIR).
pub fn run(gpu: check.Gpu, dir: []const u8, path: []const u8, dump: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 30));
    if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], "TFLINGL1")) return error.BadFixture;
    var at: usize = 8;
    const dims = try take(u32, a, bytes, &at, 4);
    const L = dims[0];
    const T: usize = dims[1];
    const D: usize = dims[2];
    const K: usize = dims[3];
    const x = try take(u16, a, bytes, &at, T * D);
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
    const t0 = check.now(gpu.io);
    var w = try rig.W.load(gpu.gpa, gpu.d, s, &pack, &o.ck, o.c, &.{L}, .{ .globals = false });
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    var g = try F.Graph.init(gpu.gpa, gpu.d, &m, s, &w);
    defer g.deinit();
    const c = &w.config;
    try check.expect(D == c.hidden and K == (if (c.isMoe(L)) @as(usize, c.topk) else 0), "fixture: layer {d}, {d} wide, top {d}", .{ L, D, K });
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .m = &m, .g = &g };
    defer r.deinit();
    const cap = T + 32;
    var pb = try F.Buffers.init(gpu.d, &m, &w, T, true, cap);
    defer pb.deinit();
    var db = try F.Buffers.init(gpu.d, &m, &w, F.window_rows, false, cap);
    defer db.deinit();
    // the whole prompt against HF, picks against HF's
    const top: usize = c.topk;
    const pk = try r.zeros(T * (top + 1) * 4);
    const st = try r.state(cap);
    defer st.deinit();
    const got = try pass(&r, &pb, st, x, D, 0, T, if (K > 0) pk else null);
    const skip = try a.alloc(bool, T);
    @memset(skip, false);
    var nf: usize = 0;
    if (K > 0) {
        const ours = try rig.sortedPicks(a, try r.back(pk, T * (top + 1) * 4), 1, T, top);
        const theirs = try a.dupe(i32, topk);
        for (0..T) |t| std.mem.sort(i32, theirs[t * K ..][0..K], {}, std.sort.asc(i32));
        nf = rig.flips(ours, theirs, 1, T, 0, T, top, skip);
    }
    const dv = dist(got, want, T, D, skip);
    std.debug.print("RESULT ling-layer-real layer {d} ({t} {s}, {t}): {d} tokens, loaded in {d:.1} s\n", .{ L, c.kind(L), if (K > 0) "MoE" else "dense", c.quant, T, load_s });
    std.debug.print("RESULT   vs HF: rel L2 {e:.3}, {e:.3} without the {d} rows whose top-{d} differs, worst kept row {e:.3}\n", .{ dv[0], dv[1], nf, top, dv[2] });
    // cut prompt, decode rows, a verify window
    const sc = try r.state(cap);
    defer sc.deinit();
    const cut0 = try pass(&r, &pb, sc, x, D, 0, 300, null);
    const cut1 = try pass(&r, &pb, sc, x, D, 300, T, null);
    try check.sameBytes("prompt cut at 300 == uncut", std.mem.sliceAsBytes(cut0), std.mem.sliceAsBytes(got[0 .. 300 * D]));
    try check.sameBytes("prompt cut at 300 == uncut (the rest)", std.mem.sliceAsBytes(cut1), std.mem.sliceAsBytes(got[300 * D ..]));
    const P = 400;
    const N = F.window_rows;
    const sd = try r.state(cap);
    defer sd.deinit();
    const sw = try r.state(cap);
    defer sw.deinit();
    _ = try pass(&r, &pb, sd, x, D, 0, P, null);
    _ = try pass(&r, &pb, sw, x, D, 0, P, null);
    const dec = try a.alloc(u16, N * D);
    for (0..N) |i| @memcpy(dec[i * D ..][0..D], try pass(&r, &db, sd, x, D, P + i, P + i + 1, null));
    try check.sameBytes("16-row verify window == 16 decode rows", std.mem.sliceAsBytes(try pass(&r, &db, sw, x, D, P, P + N, null)), std.mem.sliceAsBytes(dec));
    const dd = dist(dec, want[P * D ..], N, D, skip[P..]);
    const dp = dist(dec, @as([]const f32, try widen(a, got[P * D .. (P + N) * D])), N, D, skip[P..]);
    std.debug.print("RESULT   decode rows {d}-{d}: rel L2 {e:.3} vs HF, {e:.3} vs the prompt's rows\n", .{ P, P + N - 1, dd[0], dp[0] });
    if (dump) |d| { // the prompt's and the decode rows' bytes, for comparing two builds
        const all = try a.alloc(u16, got.len + dec.len);
        @memcpy(all[0..got.len], got);
        @memcpy(all[got.len..], dec);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = d, .data = std.mem.sliceAsBytes(all) });
    }
    try check.expect(dv[1] <= 1e-2 and dd[1] <= 1e-2, "layer {d} within rel L2 1e-2 of HF (rows whose picks agree)", .{L});
    check.pass("ling-layer-real layer {d}: rel L2 {e:.3} of HF ({d} flipped rows apart); cut == uncut, window == decode rows; peak {d} MiB on the device", .{ L, dv[1], nf, cuda.usage(false).peak >> 20 });
}

fn widen(a: std.mem.Allocator, x: []const u16) ![]f32 {
    const out = try a.alloc(f32, x.len);
    for (out, x) |*o, v| o.* = @floatCast(rig.bfOf(v));
    return out;
}
