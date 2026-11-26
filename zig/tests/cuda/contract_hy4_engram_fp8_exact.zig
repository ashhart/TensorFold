//! GPU serial FP32 oracle, exact byte comparator with hex diagnostics, and directed fixtures for `hy4_engram_fp8`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ce = @import("hy4_engram_fp8");
const c = core.contract.spec;
const g = core.contract.gates;

pub fn embedDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5 + 4;
}

pub fn zeroDepth(_: c.Shape) usize {
    return 0;
}

pub fn gateDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

pub fn sigmoid64(x: f64) f64 {
    const e = @exp(-@abs(x));
    return if (x < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
}

fn firstMismatchU16(label: []const u8, got: []const u8, want: []const u8, msg: *[200]u8) ?[]const u8 {
    if (std.mem.eql(u8, got, want)) return null;
    const gw: []align(1) const u16 = std.mem.bytesAsSlice(u16, got);
    const ww: []align(1) const u16 = std.mem.bytesAsSlice(u16, want);
    for (gw, ww, 0..) |a, b, i| {
        if (a != b) return std.fmt.bufPrint(msg, "{s}: elem {d} got=0x{x:0>4} want=0x{x:0>4}", .{ label, i, a, b }) catch label;
    }
    return label;
}

fn firstMismatchU32(label: []const u8, got: []const u8, want: []const u8, msg: *[200]u8) ?[]const u8 {
    if (std.mem.eql(u8, got, want)) return null;
    const gw: []align(1) const u32 = std.mem.bytesAsSlice(u32, got);
    const ww: []align(1) const u32 = std.mem.bytesAsSlice(u32, want);
    for (gw, ww, 0..) |a, b, i| {
        if (a != b) return std.fmt.bufPrint(msg, "{s}: elem {d} got=0x{x:0>8} want=0x{x:0>8}", .{ label, i, a, b }) catch label;
    }
    return label;
}

fn launchSerialEmbed(f: cuda.Function, s: cuda.Stream, raw: u64, norms: u64, eps: f32, out: u64, rows: usize, heads: usize, dim: usize) !void {
    const pairs = rows * heads;
    var a: cuda.Args = .{};
    inline for (.{ raw, norms, eps, out, @as(i32, @intCast(pairs)), @as(i32, @intCast(heads)), @as(i32, @intCast(dim)), @as(i32, 0) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(pairs) }, .block = .{ .x = 1 } }, s, &a);
}

fn launchSerialBias(f: cuda.Function, s: cuda.Stream, x: u64, b: u64, out: u64, rows: usize, cols: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, b, out, @as(i32, @intCast(rows)), @as(i32, @intCast(cols)), @as(i32, 0) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1 } }, s, &a);
}

fn launchSerialGate(f: cuda.Function, s: cuda.Stream, gw: ce.GateWeights, sin: u64, proj: u64, mask: u64, eo: u64, sout: u64, rows: usize, hidden: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ sin, proj, gw.norm_keys, gw.norm_queries, gw.pre_gamma, gw.stream_scales, mask, gw.eps, eo, sout, @as(i32, @intCast(hidden)), @as(i32, 0) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1 } }, s, &a);
}

pub fn checkFp64(spec: c.Spec, shape: c.Shape, out: []const u8, want: []const f64, mag: []const f64, custom_allow: ?[]const f64, gate: g.Gate) ?[]const u8 {
    if (gate == .nonfinite) {
        var planted: usize = 0;
        for (want, 0..) |w, i| {
            if (c.classOf(w) != .finite) planted += 1;
            if (c.classOf(c.read(spec.out, out, i)) != c.classOf(w)) return "non-finite class differs from float64";
        }
        if (planted == 0 or planted == want.len) return "non-finite test lacked mixed finite/non-finite outputs";
        return null;
    }
    for (want, mag, 0..) |w, m, i| {
        const limit = if (custom_allow) |al| al[i] else c.bound(spec, shape, w, m);
        const got = c.read(spec.out, out, i);
        const q = if (got == w) 0 else @abs(got - w) / limit;
        if (!(q <= 1)) return "output exceeded float64 bound";
    }
    return null;
}

fn checkDirectedEmbed(self: anytype) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    const rows: usize = 2;
    const heads: usize = 2;
    const dim: usize = 37;
    const pairs = rows * heads;
    const rec_bytes = dim + 4;
    const raw = try a.alloc(u8, pairs * rec_bytes);
    defer a.free(raw);
    const norms = try a.alloc(u16, heads * dim);
    defer a.free(norms);
    @memset(raw, 0);
    // Directed E4M3 max-finite (+/-448 = 0x7e/0xfe), subnormals (0x01/0x81), +/-0, and BF16-halfway norms.
    const codes = [_]u8{ 0x7e, 0xfe, 0x01, 0x81, 0x00, 0x80, 0x38, 0xb8, 0x7d, 0xfd };
    for (0..pairs) |p| {
        const rec = raw[p * rec_bytes ..][0..rec_bytes];
        for (0..dim) |j| rec[j] = codes[(p * 7 + j) % codes.len];
        const sc: f32 = if (p == 0) 1.0 else if (p == 1) 0.125 else if (p == 2) 0x1.0p-7 else 3.0;
        std.mem.writeInt(u32, rec[dim..][0..4], @bitCast(sc), .little);
    }
    for (norms, 0..) |*w, i| w.* = if (i % 3 == 0) 0x3f81 else if (i % 3 == 1) 0x3f80 else 0xbf81;
    var d_raw = try cuda.DeviceBuffer.fromHost(d, raw);
    defer d_raw.free();
    var d_norms = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(norms));
    defer d_norms.free();
    var d_got = try cuda.DeviceBuffer.alloc(d, pairs * dim * 2);
    defer d_got.free();
    var d_ref = try cuda.DeviceBuffer.alloc(d, pairs * dim * 2);
    defer d_ref.free();
    defer self.rig.s.synchronize() catch {};
    try d_got.fill8(0xa5, self.rig.s.handle);
    try d_ref.fill8(0xa5, self.rig.s.handle);
    try ce.launchEmbed(self.func, self.rig.s, d_raw.ptr, d_norms.ptr, self.eps, d_got.ptr, rows, heads, dim, 0);
    try launchSerialEmbed(self.ref_fn, self.rig.s, d_raw.ptr, d_norms.ptr, self.eps, d_ref.ptr, rows, heads, dim);
    try self.rig.s.synchronize();
    const h_got = try a.alloc(u8, pairs * dim * 2);
    defer a.free(h_got);
    const h_ref = try a.alloc(u8, pairs * dim * 2);
    defer a.free(h_ref);
    try d_got.download(0, h_got);
    try d_ref.download(0, h_ref);
    return firstMismatchU16("hy4_engram_fp8_embed directed e4m3/halfway", h_got, h_ref, &self.msg);
}

pub fn checkEmbed(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite and gate != .sums) return null;
    const a = self.rig.gpu.gpa;
    const got = try a.dupe(u8, try self.rig.outputBytes());
    defer a.free(got);
    const want = try a.alloc(f64, self.count());
    defer a.free(want);
    const mag = try a.alloc(f64, self.count());
    defer a.free(mag);
    try self.reference(want, mag);
    if (checkFp64(self.kernel().spec, self.shape, got, want, mag, null, gate)) |why| return why;
    if (self.ctl != null) return null;
    var ref_buf = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, got.len);
    defer ref_buf.free();
    defer self.rig.s.synchronize() catch {};
    try ref_buf.fill8(0xa5, self.rig.s.handle);
    try launchSerialEmbed(self.ref_fn, self.rig.s, self.dev_raw, self.dev_norms, self.eps, ref_buf.ptr, self.shape.rows, self.shape.n, self.shape.k);
    try self.rig.s.synchronize();
    const ref_h = try a.alloc(u8, got.len);
    defer a.free(ref_h);
    try ref_buf.download(0, ref_h);
    if (firstMismatchU16("hy4_engram_fp8_embed vs GPU serial oracle", got, ref_h, &self.msg)) |why| return why;
    if (gate == .shapes and self.shape.k == 37) {
        if (try checkDirectedEmbed(self)) |why| return why;
    }
    return null;
}

pub fn checkBias(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite and gate != .sums) return null;
    const a = self.rig.gpu.gpa;
    const got = try a.dupe(u8, try self.rig.outputBytes());
    defer a.free(got);
    const want = try a.alloc(f64, self.count());
    defer a.free(want);
    const mag = try a.alloc(f64, self.count());
    defer a.free(mag);
    try self.reference(want, mag);
    if (checkFp64(self.kernel().spec, self.shape, got, want, mag, null, gate)) |why| return why;
    if (self.ctl != null) return null;
    var ref_buf = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, got.len);
    defer ref_buf.free();
    defer self.rig.s.synchronize() catch {};
    try ref_buf.fill8(0xa5, self.rig.s.handle);
    try launchSerialBias(self.ref_fn, self.rig.s, self.dev_x, self.dev_b, ref_buf.ptr, self.shape.rows, self.shape.n);
    try self.rig.s.synchronize();
    const ref_h = try a.alloc(u8, got.len);
    defer a.free(ref_h);
    try ref_buf.download(0, ref_h);
    return firstMismatchU32("hy4_engram_fp8_bias vs GPU serial oracle", got, ref_h, &self.msg);
}

fn checkGateMode(self: anytype, mask_ptr: u64, label: []const u8, compare_old: bool) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    const nbytes = self.count() * 2;
    var ref_eo = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer ref_eo.free();
    var ref_st = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer ref_st.free();
    var both_eo = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer both_eo.free();
    var both_st = try cuda.DeviceBuffer.alloc(d, nbytes);
    defer both_st.free();
    defer self.rig.s.synchronize() catch {};
    try ref_eo.fill8(0xa5, self.rig.s.handle);
    try ref_st.fill8(0xa5, self.rig.s.handle);
    try launchSerialGate(self.ref_fn, self.rig.s, self.gw, self.dev_sin, self.dev_proj, mask_ptr, ref_eo.ptr, ref_st.ptr, self.shape.rows, self.shape.k);
    try self.rig.s.synchronize();
    const h_ref_eo = try a.alloc(u8, nbytes);
    defer a.free(h_ref_eo);
    const h_ref_st = try a.alloc(u8, nbytes);
    defer a.free(h_ref_st);
    const h_got_eo = try a.alloc(u8, nbytes);
    defer a.free(h_got_eo);
    const h_got_st = try a.alloc(u8, nbytes);
    defer a.free(h_got_st);
    try ref_eo.download(0, h_ref_eo);
    try ref_st.download(0, h_ref_st);
    // Repeat combined dual-output launch twice from freshly 0xa5-poisoned buffers.
    for (0..2) |_| {
        try both_eo.fill8(0xa5, self.rig.s.handle);
        try both_st.fill8(0xa5, self.rig.s.handle);
        try ce.launchGate(self.func, self.rig.s, self.gw, self.dev_sin, self.dev_proj, mask_ptr, both_eo.ptr, both_st.ptr, self.shape.rows, self.shape.k, 0);
        try self.rig.s.synchronize();
        try both_eo.download(0, h_got_eo);
        try both_st.download(0, h_got_st);
        if (firstMismatchU16(label, h_got_eo, h_ref_eo, &self.msg)) |why| return why;
        if (firstMismatchU16(label, h_got_st, h_ref_st, &self.msg)) |why| return why;
    }
    if (compare_old) {
        try both_eo.fill8(0xa5, self.rig.s.handle);
        try both_st.fill8(0xa5, self.rig.s.handle);
        try ce.launchGate(self.old_reduce_fn, self.rig.s, self.gw, self.dev_sin, self.dev_proj, mask_ptr, both_eo.ptr, both_st.ptr, self.shape.rows, self.shape.k, 0);
        try self.rig.s.synchronize();
        try both_eo.download(0, h_got_eo);
        try both_st.download(0, h_got_st);
        if (firstMismatchU16("hy4_engram_fp8_gate old-local vs shared warp_sum eo", h_got_eo, h_ref_eo, &self.msg)) |why| return why;
        if (firstMismatchU16("hy4_engram_fp8_gate old-local vs shared warp_sum st", h_got_st, h_ref_st, &self.msg)) |why| return why;
    }
    return null;
}

fn checkDirectedGate(self: anytype) !?[]const u8 {
    const a = self.rig.gpu.gpa;
    const d = self.rig.gpu.d;
    // Directed signed-sqrt (g>0, g<0, g==0, |g|<1e-6), warp-reduction, and BF16-halfway fixture.
    const rows: usize = 4;
    const hidden: usize = 3070;
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
    @memset(sin, 0);
    @memset(proj, 0.0);
    for (nk) |*v| v.* = c.toBf16(1.0);
    for (nq) |*v| v.* = c.toBf16(1.0);
    for (pg) |*v| v.* = c.toBf16(1.0);
    for (ss, 0..) |*v, i| v.* = if (i % 2 == 0) 0x3f81 else 0x3f80;
    for (0..rows) |r| {
        const pv = proj[r * 5 * hidden ..][0 .. 5 * hidden];
        for (0..hidden) |h| pv[h] = if (h % 4 == 0) 1.00390625 else 0.5;
        for (0..4) |s| for (0..hidden) |h| {
            const sign: f32 = if (r == 0) 1.0 else if (r == 1) -1.0 else if (r == 2) 0.0 else 1.0e-4;
            pv[(1 + s) * hidden + h] = if (h % 3 == 0) 1.0 else 0.25;
            sin[(r * 4 + s) * hidden + h] = c.toBf16(sign * (if (h % 5 == 0) @as(f32, 1.0) else 0.5));
        };
    }
    var d_sin = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(sin));
    defer d_sin.free();
    var d_proj = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(proj));
    defer d_proj.free();
    var d_nk = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nk));
    defer d_nk.free();
    var d_nq = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(nq));
    defer d_nq.free();
    var d_pg = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pg));
    defer d_pg.free();
    var d_ss = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ss));
    defer d_ss.free();
    var d_eo = try cuda.DeviceBuffer.alloc(d, n * 2);
    defer d_eo.free();
    var d_st = try cuda.DeviceBuffer.alloc(d, n * 2);
    defer d_st.free();
    var r_eo = try cuda.DeviceBuffer.alloc(d, n * 2);
    defer r_eo.free();
    var r_st = try cuda.DeviceBuffer.alloc(d, n * 2);
    defer r_st.free();
    defer self.rig.s.synchronize() catch {};
    const gw: ce.GateWeights = .{ .norm_keys = d_nk.ptr, .norm_queries = d_nq.ptr, .pre_gamma = d_pg.ptr, .stream_scales = d_ss.ptr, .eps = ce.norm_eps };
    try d_eo.fill8(0xa5, self.rig.s.handle);
    try d_st.fill8(0xa5, self.rig.s.handle);
    try r_eo.fill8(0xa5, self.rig.s.handle);
    try r_st.fill8(0xa5, self.rig.s.handle);
    try ce.launchGate(self.func, self.rig.s, gw, d_sin.ptr, d_proj.ptr, 0, d_eo.ptr, d_st.ptr, rows, hidden, 0);
    try launchSerialGate(self.ref_fn, self.rig.s, gw, d_sin.ptr, d_proj.ptr, 0, r_eo.ptr, r_st.ptr, rows, hidden);
    try self.rig.s.synchronize();
    const h_de = try a.alloc(u8, n * 2);
    defer a.free(h_de);
    const h_ds = try a.alloc(u8, n * 2);
    defer a.free(h_ds);
    const h_re = try a.alloc(u8, n * 2);
    defer a.free(h_re);
    const h_rs = try a.alloc(u8, n * 2);
    defer a.free(h_rs);
    try d_eo.download(0, h_de);
    try d_st.download(0, h_ds);
    try r_eo.download(0, h_re);
    try r_st.download(0, h_rs);
    if (firstMismatchU16("hy4_engram_fp8_gate directed ssqrt/halfway eo", h_de, h_re, &self.msg)) |why| return why;
    return firstMismatchU16("hy4_engram_fp8_gate directed ssqrt/halfway st", h_ds, h_rs, &self.msg);
}

pub fn checkGate(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite and gate != .sums) return null;
    const a = self.rig.gpu.gpa;
    const n = self.count();
    const got = try a.dupe(u8, try self.rig.outputBytes());
    defer a.free(got);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const allow = try a.alloc(f64, n);
    defer a.free(allow);
    try self.reference(want, mag);
    const spec = self.kernel().spec;
    for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m);
    try self.bounds(want, mag, allow);
    if (checkFp64(spec, self.shape, got, want, mag, allow, gate)) |why| return why;
    if (self.ctl != null and !self.check_exact) return null;
    // Exact byte check against GPU serial FP32 oracle (also proves wrong-order control fails!).
    var ref_eo = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 2);
    defer ref_eo.free();
    var ref_st = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 2);
    defer ref_st.free();
    defer self.rig.s.synchronize() catch {};
    try ref_eo.fill8(0xa5, self.rig.s.handle);
    try ref_st.fill8(0xa5, self.rig.s.handle);
    try launchSerialGate(self.ref_fn, self.rig.s, self.gw, self.dev_sin, self.dev_proj, self.dev_mask, ref_eo.ptr, ref_st.ptr, self.shape.rows, self.shape.k);
    try self.rig.s.synchronize();
    const ref_primary = try a.alloc(u8, n * 2);
    defer a.free(ref_primary);
    try (if (self.kind == .engram_out) ref_eo else ref_st).download(0, ref_primary);
    if (firstMismatchU16("hy4_engram_fp8_gate primary vs GPU serial oracle", got, ref_primary, &self.msg)) |why| return why;
    if (self.ctl != null) return null;
    if (try checkGateMode(self, self.dev_mask, "hy4_engram_fp8_gate dual-output repeat", true)) |why| return why;
    if (gate == .shapes and self.shape.k == 3072) {
        if (try checkGateMode(self, 0, "hy4_engram_fp8_gate null-mask dual-output", false)) |why| return why;
        const alt = try a.alloc(u8, self.shape.rows);
        defer a.free(alt);
        for (alt, 0..) |*m, r| m.* = @intCast((r + 1) & 1);
        var d_alt = try cuda.DeviceBuffer.fromHost(self.rig.gpu.d, alt);
        defer d_alt.free();
        defer self.rig.s.synchronize() catch {};
        if (try checkGateMode(self, d_alt.ptr, "hy4_engram_fp8_gate alternating-mask dual-output", false)) |why| return why;
        if (try checkDirectedGate(self)) |why| return why;
    }
    return null;
}
