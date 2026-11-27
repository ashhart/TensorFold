//! Directed GB10 receipts for `hy4_engram_fp8_gate`: alt-masks, signed-sqrt g, and BF16-halfway.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ce = @import("hy4_engram_fp8");
const exact = @import("contract_hy4_engram_fp8_exact.zig");
const c = core.contract.spec;

fn isFiniteBf16(bits: u16) bool {
    return (bits & 0x7f80) != 0x7f80;
}

fn hostExpectedGate(g: f32) f32 {
    const mag: f32 = @sqrt(@max(@abs(g), @as(f32, 1.0e-6)));
    const ssqrt: f32 = if (g > 0.0) mag else if (g < 0.0) -mag else 0.0;
    const e: f32 = @exp(-@abs(ssqrt));
    return if (ssqrt < 0.0) e / (1.0 + e) else 1.0 / (1.0 + e);
}

fn runCompareAll(
    self: anytype,
    gw: ce.GateWeights,
    sin: []const u16,
    proj: []const f32,
    mask_opt: ?[]const u8,
    rows: usize,
    hidden: usize,
    label: []const u8,
    out_eo: ?[]u16,
    out_st: ?[]u16,
) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    const n = rows * 4 * hidden;
    const nbytes = n * 2;
    var d_sin = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(sin));
    defer d_sin.free();
    var d_proj = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(proj));
    defer d_proj.free();
    var d_mask: ?cuda.DeviceBuffer = if (mask_opt) |m| try cuda.DeviceBuffer.fromHost(d, m) else null;
    defer if (d_mask) |*dm| dm.free();
    const mask_ptr: u64 = if (d_mask) |dm| dm.ptr else 0;
    var r_eo = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer r_eo.free();
    var r_st = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer r_st.free();
    var p_eo = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer p_eo.free();
    var p_st = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer p_st.free();
    var o_eo = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer o_eo.free();
    var o_st = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer o_st.free();
    defer self.rig.s.synchronize() catch {};

    try r_eo.fill8(0xa5, self.rig.s.handle);
    try r_st.fill8(0xa5, self.rig.s.handle);
    try exact.launchSerialGate(self.ref_fn, self.rig.s, gw, d_sin.ptr, d_proj.ptr, mask_ptr, r_eo.ptr, r_st.ptr, rows, hidden);
    try self.rig.s.synchronize();

    const h_re = try a.alloc(u8, nbytes);
    defer a.free(h_re);
    const h_rs = try a.alloc(u8, nbytes);
    defer a.free(h_rs);
    const h_pe = try a.alloc(u8, nbytes);
    defer a.free(h_pe);
    const h_ps = try a.alloc(u8, nbytes);
    defer a.free(h_ps);
    try r_eo.download(0, h_re);
    try r_st.download(0, h_rs);

    // Repeat production and old-local launches twice each from fresh 0xa5 poison (repeat=0, both outputs non-null).
    for (0..2) |_| {
        try p_eo.fill8(0xa5, self.rig.s.handle);
        try p_st.fill8(0xa5, self.rig.s.handle);
        try ce.launchGate(self.func, self.rig.s, gw, d_sin.ptr, d_proj.ptr, mask_ptr, p_eo.ptr, p_st.ptr, rows, hidden, 0);
        try self.rig.s.synchronize();
        try p_eo.download(0, h_pe);
        try p_st.download(0, h_ps);
        if (exact.firstMismatchU16(label, h_pe, h_re, &self.msg)) |why| return why;
        if (exact.firstMismatchU16(label, h_ps, h_rs, &self.msg)) |why| return why;

        try o_eo.fill8(0xa5, self.rig.s.handle);
        try o_st.fill8(0xa5, self.rig.s.handle);
        try ce.launchGate(self.old_reduce_fn, self.rig.s, gw, d_sin.ptr, d_proj.ptr, mask_ptr, o_eo.ptr, o_st.ptr, rows, hidden, 0);
        try self.rig.s.synchronize();
        try o_eo.download(0, h_pe);
        try o_st.download(0, h_ps);
        if (exact.firstMismatchU16(label, h_pe, h_re, &self.msg)) |why| return why;
        if (exact.firstMismatchU16(label, h_ps, h_rs, &self.msg)) |why| return why;
    }
    const eo_w: []align(1) const u16 = std.mem.bytesAsSlice(u16, h_re);
    const st_w: []align(1) const u16 = std.mem.bytesAsSlice(u16, h_rs);
    if (mask_opt) |m| {
        for (0..rows) |r| if (m[r] == 0) {
            const base = r * 4 * hidden;
            for (0..4 * hidden) |j| {
                if (eo_w[base + j] != 0 or st_w[base + j] != sin[base + j]) return "masked row modified output";
            }
        };
    }
    if (out_eo) |dst| for (dst, 0..) |*v, i| {
        v.* = eo_w[i];
    };
    if (out_st) |dst| for (dst, 0..) |*v, i| {
        v.* = st_w[i];
    };
    return null;
}

fn checkAltMask(self: anytype, hidden: usize, label: []const u8) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    const rows: usize = 4;
    const n = rows * 4 * hidden;
    var prng = std.Random.DefaultPrng.init(0x416c744d61736b ^ hidden);
    const rng = prng.random();
    const sin = try a.alloc(u16, n);
    defer a.free(sin);
    const proj = try a.alloc(f32, rows * 5 * hidden);
    defer a.free(proj);
    const nk = try a.alloc(u16, 4 * hidden);
    defer a.free(nk);
    const nq = try a.alloc(u16, 4 * hidden);
    defer a.free(nq);
    const pg = try a.alloc(u16, hidden);
    defer a.free(pg);
    const ss = try a.alloc(u16, 4 * hidden);
    defer a.free(ss);
    for (sin) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * 1.5);
    for (proj) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 1.5;
    for (nk) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
    for (nq) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
    for (pg) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
    for (ss) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * 0.25);
    var d_nk = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nk));
    defer d_nk.free();
    var d_nq = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nq));
    defer d_nq.free();
    var d_pg = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pg));
    defer d_pg.free();
    var d_ss = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ss));
    defer d_ss.free();
    defer self.rig.s.synchronize() catch {};
    const gw: ce.GateWeights = .{ .norm_keys = d_nk.ptr, .norm_queries = d_nq.ptr, .pre_gamma = d_pg.ptr, .stream_scales = d_ss.ptr, .eps = ce.norm_eps };
    const alt = [_]u8{ 1, 0, 1, 0 };
    return runCompareAll(self, gw, sin, proj, &alt, rows, hidden, label, null, null);
}

fn fillUnitRmsKey(key: []f32, k0: f32) void {
    // 4.0 + 254 * 1.0 == 258.0 exactly in f32 (k0^2 <= 4.2e-9 vanishes into 1.0/4.0), so inv_k == 1.0 with eps == 0.
    key[0] = k0;
    key[1] = 2.0;
    key[2] = 0.0;
    key[3] = 0.0;
    for (key[4..]) |*v| v.* = 1.0;
}

fn checkDirected258(self: anytype) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    const rows: usize = 4;
    const hidden: usize = 258;
    const n = rows * 4 * hidden;
    const sin = try a.alloc(u16, n);
    defer a.free(sin);
    const proj = try a.alloc(f32, rows * 5 * hidden);
    defer a.free(proj);
    const nk = try a.alloc(u16, 4 * hidden);
    defer a.free(nk);
    const nq = try a.alloc(u16, 4 * hidden);
    defer a.free(nq);
    const pg = try a.alloc(u16, hidden);
    defer a.free(pg);
    const ss = try a.alloc(u16, 4 * hidden);
    defer a.free(ss);
    const got_eo = try a.alloc(u16, n);
    defer a.free(got_eo);
    const got_st = try a.alloc(u16, n);
    defer a.free(got_st);
    @memset(sin, 0);
    @memset(proj, 0.0);
    @memset(nk, 0);
    @memset(nq, 0);
    for (pg) |*v| v.* = c.toBf16(1.0);
    @memset(ss, 0);

    // Construct exact k_exact so k_exact * (1 / sqrt(258)) == +1e-6f and -k_exact * (1 / sqrt(258)) == -1e-6f.
    const inv_sqrt_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
    var k_exact: f32 = 1.0e-6 / inv_sqrt_h;
    if (k_exact * inv_sqrt_h < 1.0e-6) k_exact = std.math.nextAfter(f32, k_exact, std.math.inf(f32));
    if (k_exact * inv_sqrt_h > 1.0e-6) k_exact = std.math.nextAfter(f32, k_exact, -std.math.inf(f32));
    if (k_exact * inv_sqrt_h != 1.0e-6 or (-k_exact) * inv_sqrt_h != -1.0e-6) return "directed k_exact failed +/-1e-6";
    const k_sub: f32 = 0.5 * k_exact;
    const k_hi: f32 = 4.0 * k_exact;
    const g_cases = [4][3]f32{
        .{ k_sub, k_exact, k_exact },
        .{ -k_sub, -k_exact, -k_exact },
        .{ 0.0, k_hi, 0.0 },
        .{ -0.0, -k_hi, -0.0 },
    };
    const g_pos_sub = k_sub * inv_sqrt_h;
    const g_neg_sub = (-k_sub) * inv_sqrt_h;
    const g_pos_hi = k_hi * inv_sqrt_h;
    const g_neg_hi = (-k_hi) * inv_sqrt_h;
    if (!(g_pos_sub > 0.0 and g_pos_sub < 1.0e-6)) return "directed g_pos_sub out of (0, 1e-6)";
    if (!(g_neg_sub < 0.0 and g_neg_sub > -1.0e-6)) return "directed g_neg_sub out of (-1e-6, 0)";
    if (!(g_pos_hi > 1.0e-6 and g_neg_hi < -1.0e-6)) return "directed g_hi out of +/-1e-6";

    // Streams 0-2: h=0 sets dot; h=8 uses stream_scales=-0.5 on streams 0-1 to expose gate - 0.5 in bf16.
    for (0..3) |s| {
        nk[s * hidden] = c.toBf16(1.0);
        nq[s * hidden] = c.toBf16(1.0);
        if (s < 2) ss[s * hidden + 8] = c.toBf16(-0.5);
    }

    // Stream 3 (finite inv_k=1, nk=nq=0 -> exact zero-dot, gate=0.5): 8 BF16-halfway, +/-1 ULP, and cancellation cases.
    const Dir = struct { pg: f32, sc: f32, sin: f32 };
    const dirs = [_]Dir{
        .{ .pg = 2.0, .sc = 0x1.0p-9, .sin = 1.0 + 0x1.0p-7 },
        .{ .pg = 2.0, .sc = 0x1.0p-8 + 0x1.0p-9, .sin = 1.0 + 0x1.0p-7 },
        .{ .pg = -2.0, .sc = 0x1.0p-9, .sin = -1.0 - 0x1.0p-7 },
        .{ .pg = -2.0, .sc = 0x1.0p-8 + 0x1.0p-9, .sin = -1.0 - 0x1.0p-7 },
        .{ .pg = 2.0, .sc = -0.5, .sin = 1.0 + 0x1.0p-7 },
        .{ .pg = 2.0, .sc = -0.5 + 0x1.0p-8, .sin = -0x1.0p-7 },
        .{ .pg = 2.0, .sc = 0x1.0p-9 + 0x1.0p-10, .sin = 0x1.0p-7 },
        .{ .pg = 2.0, .sc = 0x1.0p-9 - 0x1.0p-10, .sin = -0x1.0p-7 },
    };
    for (dirs, 0..) |dc, h| {
        pg[h] = c.toBf16(dc.pg);
        ss[3 * hidden + h] = c.toBf16(dc.sc);
    }

    for (0..rows) |r| {
        const pv = proj[r * 5 * hidden ..][0 .. 5 * hidden];
        for (0..hidden) |h| pv[h] = 1.0;
        for (0..3) |s| {
            fillUnitRmsKey(pv[(1 + s) * hidden ..][0..hidden], g_cases[r][s]);
            for (0..hidden) |h| sin[(r * 4 + s) * hidden + h] = c.toBf16(1.0);
        }
        // Stream 3: unit-RMS key (inv_k == 1.0) and nonzero sin (finite inv_q) with nk=nq=0 -> dot == 0, gate == 0.5.
        for (0..hidden) |h| pv[4 * hidden + h] = 1.0;
        for (dirs, 0..) |dc, h| sin[(r * 4 + 3) * hidden + h] = c.toBf16(if (r % 2 == 0) dc.sin else -dc.sin);
    }

    var d_nk = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nk));
    defer d_nk.free();
    var d_nq = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nq));
    defer d_nq.free();
    var d_pg = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pg));
    defer d_pg.free();
    var d_ss = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ss));
    defer d_ss.free();
    defer self.rig.s.synchronize() catch {};
    const gw: ce.GateWeights = .{ .norm_keys = d_nk.ptr, .norm_queries = d_nq.ptr, .pre_gamma = d_pg.ptr, .stream_scales = d_ss.ptr, .eps = 0.0 };
    const mask = [_]u8{ 1, 1, 1, 1 };
    if (try runCompareAll(self, gw, sin, proj, &mask, rows, hidden, "hy4_engram_fp8_gate directed 258 g/halfway", got_eo, got_st)) |why| return why;

    // Independently assert every output word is finite (no NaN/Inf from singular RMS denominators).
    for (0..n) |i| {
        if (!isFiniteBf16(got_eo[i]) or !isFiniteBf16(got_st[i])) return "directed 258 produced non-finite bf16";
    }
    // Independently assert exact finite bf16 bits for +/-sub-1e-6, exact +/-1e-6, +/-above-1e-6, and +/-0.0 at h=8.
    for (0..rows) |r| {
        for (0..3) |s| {
            const idx = (r * 4 + s) * hidden + 8;
            const g = g_cases[r][s] * inv_sqrt_h;
            const gate = hostExpectedGate(g);
            const sc: f32 = if (s < 2) -0.5 else 0.0;
            const want_eo = c.toBf16(gate + sc);
            const want_st = c.toBf16(1.0 + c.fromBf16(want_eo));
            if (got_eo[idx] != want_eo or got_st[idx] != want_st) {
                return std.fmt.bufPrint(&self.msg, "directed 258 g r={d} s={d}: eo=0x{x:0>4}/0x{x:0>4} st=0x{x:0>4}/0x{x:0>4}", .{ r, s, got_eo[idx], want_eo, got_st[idx], want_st }) catch "directed 258 g mismatch";
            }
        }
        // Independently assert exact finite bf16 bits for all 8 zero-dot (gate=0.5) halfway/ULP/cancellation cases.
        for (dirs, 0..) |dc, h| {
            const idx = (r * 4 + 3) * hidden + h;
            const want_eo = c.toBf16(0.5 * dc.pg + c.fromBf16(c.toBf16(dc.sc)) * dc.pg);
            const sin_v = c.fromBf16(sin[idx]);
            const want_st = c.toBf16(sin_v + c.fromBf16(want_eo));
            if (got_eo[idx] != want_eo or got_st[idx] != want_st) {
                return std.fmt.bufPrint(&self.msg, "directed 258 half r={d} h={d}: eo=0x{x:0>4}/0x{x:0>4} st=0x{x:0>4}/0x{x:0>4}", .{ r, h, got_eo[idx], want_eo, got_st[idx], want_st }) catch "directed 258 half mismatch";
            }
        }
    }
    std.debug.print("INFO hy4_engram_fp8 directed258: finite exact bits for g=+/-{e:.3}, +/-1.000e-6, +/-{e:.3}, +/-0 + 8 BF16-halfway/cancel zero-dot cases: old==prod==serial==host x2 PASS\n", .{ g_pos_sub, g_pos_hi });
    return null;
}

pub fn checkAll(self: anytype) !?[]const u8 {
    if (try checkAltMask(self, 258, "hy4_engram_fp8_gate alt-mask 4x258")) |why| return why;
    if (try checkAltMask(self, 3074, "hy4_engram_fp8_gate alt-mask 4x3074")) |why| return why;
    std.debug.print("INFO hy4_engram_fp8 alt-mask: 4x258 and 4x3074 [1,0,1,0] dual-output old==prod==serial x2 PASS\n", .{});
    return checkDirected258(self);
}
