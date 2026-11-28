//! Exact BiasCorr zero-input and fused-SwiGLU production-oracle checks for the XPU FP64 probe.
const std = @import("std");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const freeze = @import("fp64_freeze.zig");
const ze = q.ze;
const Dev = q.Dev;
const Acts = q.Acts;
const Proj = q.Proj;
const alloc = q.alloc;
const NS = 24;

fn readRow(d: *Dev, dst: []f16, src: *anyopaque, row: usize, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, q.off(src, row * n * 2), n * 2, null, 0, null));
}

fn copyOutRows(d: *Dev, dst: []f16, src: *anyopaque, rows: []const usize, n: usize) !void {
    for (rows, 0..) |r, i| try readRow(d, dst[i * n ..][0..n], src, r, n);
}

fn auditTagsEnabled() bool {
    const v = std.c.getenv("XPU_BIASCORR_TAG_AUDIT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn auditFp16Value(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

fn auditLogicalWidth(layer: usize, site: u32) usize {
    return switch (site) {
        0 => if (q.isAttention(layer)) 14336 else 16480,
        2 => 34816,
        1, 3 => 5120,
        else => 0,
    };
}

/// Deterministic exact-FP16 tags for the unique-index audit sidecar, in device column order.
fn auditExpected(layer: usize, site: u32, c: usize) f32 {
    if (c >= auditLogicalWidth(layer, site)) return 0;
    var bits: u16 = undefined;
    if (site == 2) {
        const lane = c % 32;
        const block = c / 32;
        const ch = block * 16 + (lane % 16);
        bits = @intCast(ch + 1);
        if (lane >= 16) bits |= 0x8000;
    } else {
        bits = @intCast(c + 1);
    }
    return auditFp16Value(bits);
}

/// Zero-input audit: with x and both correction streams zero, every GEMM output element must be exactly fp16(bias[c]).
pub fn zeroBiasCheck(d: *Dev, a: *Acts, w: Proj, bias: []const f32, layer: usize, site: u32) !void {
    const N: usize = w.n;
    const K: usize = w.k;
    const sh = q.tpShape();
    const h_cols: usize = @max(@as(usize, q.D), @max(@as(usize, sh.ff), @as(usize, sh.od)));
    const corr_bytes = @as(usize, q.maxRows()) * fwd.gStride(h_cols / 64) * 2;
    const zx = try d.g.deviceAlloc(@as(usize, q.maxRows()) * K * 2);
    defer d.g.free(zx);
    const zhi = try d.g.deviceAlloc(corr_bytes);
    defer d.g.free(zhi);
    const zlo = try d.g.deviceAlloc(corr_bytes);
    defer d.g.free(zlo);
    try d.fill(zx, @as(u16, 0), @as(usize, q.maxRows()) * K * 2);
    try d.fill(zhi, @as(u16, 0), corr_bytes);
    try d.fill(zlo, @as(u16, 0), corr_bytes);
    const scratch = try d.g.deviceAlloc(@as(usize, NS) * N * 2);
    defer d.g.free(scratch);
    var bad: usize = 0;
    var total: usize = 0;
    var fp32_exact: usize = 0;
    const tag_audit = auditTagsEnabled();
    for ([_]u32{ 1, 8, NS }, 0..) |m, pi| {
        try fwd.gemmRows(d, a.pscr, zx, zhi, zlo, w, scratch, m, m > 16);
        try d.sync();
        const got = try alloc.alloc(f16, N);
        defer alloc.free(got);
        try readRow(d, got, scratch, 0, N);
        try d.sync();
        for (0..N) |c| {
            const want32 = if (tag_audit) auditExpected(layer, site, c) else bias[c];
            const want: f16 = @floatCast(want32);
            total += 1;
            const got_bits = @as(u16, @bitCast(got[c]));
            if (got_bits == @as(u16, @bitCast(want))) {
                const got32: f32 = @floatCast(got[c]);
                if (got32 == want32) fp32_exact += 1 else if (tag_audit) bad += 1;
            } else bad += 1;
        }
        if (bad != 0 and pi == 2) break;
    }
    std.debug.print("fp64 ref: zero-input exact fp16(bias) audit: {d}/{d} exact ({d} mismatches), direct fp32 exact {d}/{d}\n", .{ total - bad, total, bad, fp32_exact, total });
    if (bad != 0) return error.BiasCorrZeroInputMismatch;
}

fn swigluMaxRelL2() !f64 {
    const v = std.c.getenv("XPU_FP64_SWIGLU_MAX_REL_L2") orelse return 5.0e-3;
    const bound = std.fmt.parseFloat(f64, std.mem.span(v)) catch return error.BadSwigluRelL2Bound;
    if (!std.math.isFinite(bound) or bound <= 0.0) return error.BadSwigluRelL2Bound;
    return bound;
}

/// Compare the production fused SwiGLU kernel with the independent FP64 gate/up reference (including logical bias).
pub fn fusedSwigluCheck(d: *Dev, a: *Acts, w: Proj, T: u32, rows: []const usize, ref: []const f64, replay: ?freeze.Replay) !void {
    if (!w.ilv or !w.wide or fwd.gemm_hook != null) return error.FusedSwigluPathNotRun;
    const xb = @as(usize, T) * q.D * 2;
    const cb = freeze.corrBytes();
    const bx = try d.g.deviceAlloc(xb);
    defer d.g.free(bx);
    const bhi = try d.g.deviceAlloc(cb);
    defer d.g.free(bhi);
    const blo = try d.g.deviceAlloc(cb);
    defer d.g.free(blo);
    try q.copy(d, bx, a.h, xb);
    try q.copy(d, bhi, a.ahi, cb);
    try q.copy(d, blo, a.alo, cb);
    if (replay) |rr| {
        try q.copy(d, a.h, rr.x, xb);
        try q.copy(d, a.ahi, rr.hi, cb);
        try q.copy(d, a.alo, rr.lo, cb);
    }
    const outdev = try fwd.mlpIn(d, a, w, T, @intCast(q.GS), false, true);
    try q.copy(d, a.h, bx, xb);
    try q.copy(d, a.ahi, bhi, cb);
    try q.copy(d, a.alo, blo, cb);
    if (outdev != a.p) return error.FusedSwigluPathNotRun;
    try d.sync();
    const F: usize = w.n / 2;
    const got = try alloc.alloc(f16, rows.len * F);
    defer alloc.free(got);
    try copyOutRows(d, got, outdev, rows, F);
    try d.sync();
    var se: f64 = 0;
    var sr: f64 = 0;
    var ma: f64 = 0;
    for (0..rows.len) |r| {
        for (0..F) |c| {
            const gc = (c / 16) * 32 + c % 16;
            const uc = gc + 16;
            const g = ref[r * w.n + gc];
            const u = ref[r * w.n + uc];
            const expect = g / (1 + @exp(-g)) * u;
            const got64: f64 = @floatCast(got[r * F + c]);
            if (!std.math.isFinite(got64)) return error.NonFiniteFusedSwigluOutput;
            if (!std.math.isFinite(expect)) return error.NonFiniteSwigluOracle;
            const e = got64 - expect;
            se += e * e;
            sr += expect * expect;
            ma = @max(ma, @abs(e));
        }
    }
    var rel: f64 = std.math.inf(f64);
    if (sr != 0) rel = @sqrt(se / sr) else if (se == 0) rel = 0;
    const bound = try swigluMaxRelL2();
    std.debug.print("fp64 ref: fused SwiGLU production output vs independent FP64 oracle: rel L2 {e:.2}, max abs {e:.2}, bound {e:.2}\n", .{ rel, ma, bound });
    if (!std.math.isFinite(rel) or rel > bound) return error.FusedSwigluOracleMismatch;
}
