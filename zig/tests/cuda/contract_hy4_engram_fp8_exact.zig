//! Exact serial FP32 oracle for `hy4_engram_fp8` (embed, bias, gate) matching `engram_fp8.cuh` bit-for-bit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ce = @import("hy4_engram_fp8");
const c = core.contract.spec;
const g = core.contract.gates;

const kStreams: usize = 4;
const kThreads: usize = 256;
const kWarps: usize = kThreads / 32;

pub fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn roundBf(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fff;
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn e4m3(b: u8) f32 {
    const s: u32 = @as(u32, b & 0x80) << 24;
    const e: u32 = (b >> 3) & 0x0f;
    const m: u32 = b & 0x07;
    if (e == 15 and m == 7) return @bitCast(s | 0x7fc00000);
    if (e == 0) {
        if (m == 0) return @bitCast(s);
        const mi: i32 = if ((b & 0x80) != 0) -@as(i32, @intCast(m)) else @as(i32, @intCast(m));
        return @as(f32, @floatFromInt(mi)) * 0x1.0p-9;
    }
    return @bitCast(s | ((e + 120) << 23) | (m << 20));
}

fn tfExp(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    if (x > 104.0) return std.math.inf(f32);
    if (x < -104.0) return 0.0;
    const j = @mulAdd(f32, x, 0x1.715476p+0, 0x1.8p+23) - 0x1.8p+23;
    var f = @mulAdd(f32, j, -0x1.62e4p-1, x);
    f = @mulAdd(f32, j, -0x1.7f7d1cp-20, f);
    var p: f32 = 0x1.a01a02p-13;
    p = @mulAdd(f32, p, f, 0x1.6c16c2p-10);
    p = @mulAdd(f32, p, f, 0x1.111112p-7);
    p = @mulAdd(f32, p, f, 0x1.555556p-5);
    p = @mulAdd(f32, p, f, 0x1.555556p-3);
    p = @mulAdd(f32, p, f, 0.5);
    p = @mulAdd(f32, p, f, 1.0);
    p = @mulAdd(f32, p, f, 1.0);
    const i: i32 = @intFromFloat(j);
    const a: f32 = @bitCast(@as(u32, if (i > 0) 0x7f000000 else 0x02000000));
    const b: f32 = @bitCast(@as(u32, @intCast(if (i > 0) i << 23 else (i + 250) << 23)));
    return (p * a) * b;
}

fn sigmoid32(x: f32) f32 {
    const e = tfExp(-@abs(x));
    return if (x < 0.0) e / (1.0 + e) else 1.0 / (1.0 + e);
}

fn warpSum32(lanes_in: [32]f32) f32 {
    var v = lanes_in;
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (0..32) |l| next[l] = v[l] + v[l ^ o];
        v = next;
    }
    return v[0];
}

fn blockReduce256(threads: [kThreads]f32) f32 {
    var wsum: [kWarps]f32 = undefined;
    for (0..kWarps) |w| {
        var lanes: [32]f32 = undefined;
        for (0..32) |l| lanes[l] = threads[w * 32 + l];
        wsum[w] = warpSum32(lanes);
    }
    var s = wsum[0];
    for (1..kWarps) |w| s = s + wsum[w];
    return s;
}

pub fn serialEmbed(raw: []const u8, norms: []const u16, eps: f32, rows: usize, heads: usize, dim: usize, out: []u16) void {
    const pairs = rows * heads;
    const rec_bytes = dim + 4;
    for (0..pairs) |pair| {
        const head = pair % heads;
        const rec = raw[pair * rec_bytes ..][0..rec_bytes];
        const scale: f32 = @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little));
        var lanes: [32]f32 = @splat(0.0);
        for (0..32) |lane| {
            var sq: f32 = 0.0;
            var d = lane;
            while (d < dim) : (d += 32) {
                const v = e4m3(rec[d]) * scale;
                sq = sq + (v * v);
            }
            lanes[lane] = sq;
        }
        const sq = warpSum32(lanes);
        const inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(dim)) + eps);
        for (0..dim) |d| {
            const v = e4m3(rec[d]) * scale;
            out[pair * dim + d] = roundBf((v * inv) * bf(norms[head * dim + d]));
        }
    }
}

pub fn serialBias(x: []const f32, bias: []const u16, rows: usize, cols: usize, out: []f32) void {
    for (0..rows * cols) |i| {
        out[i] = x[i] + bf(bias[i % cols]);
    }
}

pub fn serialGate(
    streams_in: []const u16,
    proj: []const f32,
    norm_keys: []const u16,
    norm_queries: []const u16,
    pre_gamma: []const u16,
    stream_scales: []const u16,
    mask: []const u8,
    eps: f32,
    rows: usize,
    hidden: usize,
    eo_out: []u16,
    st_out: []u16,
) void {
    const hf: f32 = @floatFromInt(hidden);
    for (0..rows) |r| {
        const active = mask[r] != 0;
        const pv = proj[r * (1 + kStreams) * hidden ..][0 .. (1 + kStreams) * hidden];
        const sin = streams_in[r * kStreams * hidden ..][0 .. kStreams * hidden];
        var acc: [9][kThreads]f32 = @splat(@as([kThreads]f32, @splat(0.0)));
        for (0..kThreads) |t| {
            var h = t;
            while (h < hidden) : (h += kThreads) {
                const vv = pv[h];
                acc[0][t] = acc[0][t] + (vv * vv);
                for (0..kStreams) |s| {
                    const kv = pv[(1 + s) * hidden + h];
                    acc[1 + s][t] = acc[1 + s][t] + (kv * kv);
                    const qv = bf(sin[s * hidden + h]);
                    acc[5 + s][t] = acc[5 + s][t] + (qv * qv);
                }
            }
        }
        var inv_s: [9]f32 = undefined;
        for (0..9) |j| inv_s[j] = 1.0 / @sqrt(blockReduce256(acc[j]) / hf + eps);

        var dot: [kStreams][kThreads]f32 = @splat(@as([kThreads]f32, @splat(0.0)));
        for (0..kThreads) |t| {
            var h = t;
            while (h < hidden) : (h += kThreads) {
                for (0..kStreams) |s| {
                    const kh = (pv[(1 + s) * hidden + h] * inv_s[1 + s]) * bf(norm_keys[s * hidden + h]);
                    const qh = (bf(sin[s * hidden + h]) * inv_s[5 + s]) * bf(norm_queries[s * hidden + h]);
                    dot[s][t] = dot[s][t] + (kh * qh);
                }
            }
        }
        var gate_s: [kStreams]f32 = undefined;
        for (0..kStreams) |s| {
            const gv = blockReduce256(dot[s]) * (1.0 / @sqrt(hf));
            const mag = @sqrt(@max(@abs(gv), 1.0e-6));
            const ssqrt = if (std.math.isNan(gv)) gv else if (gv > 0.0) mag else if (gv < 0.0) -mag else 0.0;
            gate_s[s] = sigmoid32(ssqrt);
        }

        const inv_v = inv_s[0];
        for (0..hidden) |h| {
            const main_h = bf(pre_gamma[h]) * (pv[h] * inv_v);
            for (0..kStreams) |s| {
                const idx = (r * kStreams + s) * hidden + h;
                const st_in_bits = streams_in[idx];
                if (active) {
                    const eo_f = (gate_s[s] * main_h) + (bf(stream_scales[s * hidden + h]) * main_h);
                    const eo_bits = roundBf(eo_f);
                    eo_out[idx] = eo_bits;
                    st_out[idx] = roundBf(bf(st_in_bits) + bf(eo_bits));
                } else {
                    eo_out[idx] = 0;
                    st_out[idx] = st_in_bits;
                }
            }
        }
    }
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

pub fn checkEmbed(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite) return null;
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
    const ref = try a.alloc(u16, self.count());
    defer a.free(ref);
    serialEmbed(self.raw, self.norms, self.eps, self.shape.rows, self.shape.n, self.shape.k, ref);
    if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(ref))) return "hy4_engram_fp8_embed bytes differ from serial FP32 oracle";
    return null;
}

pub fn checkBias(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite) return null;
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
    const ref = try a.alloc(f32, self.count());
    defer a.free(ref);
    serialBias(self.x, self.b, self.shape.rows, self.shape.n, ref);
    if (gate == .nonfinite) {
        const got_f: []align(1) const f32 = std.mem.bytesAsSlice(f32, got);
        for (got_f, ref) |gv, rv| {
            if (std.math.isNan(rv)) {
                if (!std.math.isNan(gv)) return "hy4_engram_fp8_bias dropped NaN vs serial FP32 oracle";
            } else if (@as(u32, @bitCast(gv)) != @as(u32, @bitCast(rv))) {
                return "hy4_engram_fp8_bias non-NaN bits differ from serial FP32 oracle";
            }
        }
    } else if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(ref))) {
        return "hy4_engram_fp8_bias bytes differ from serial FP32 oracle";
    }
    return null;
}

pub fn checkGate(self: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .shapes and gate != .nonfinite) return null;
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
    if (self.ctl != null) return null;
    const ref_eo = try a.alloc(u16, n);
    defer a.free(ref_eo);
    const ref_st = try a.alloc(u16, n);
    defer a.free(ref_st);
    serialGate(self.streams_in, self.proj, self.norm_keys, self.norm_queries, self.pre_gamma, self.stream_scales, self.mask, self.eps, self.shape.rows, self.shape.k, ref_eo, ref_st);
    const want_primary = std.mem.sliceAsBytes(if (self.kind == .engram_out) ref_eo else ref_st);
    if (!std.mem.eql(u8, got, want_primary)) return "hy4_engram_fp8_gate primary output bytes differ from serial FP32 oracle";
    // Also launch with BOTH engram_out and streams_out non-null into poisoned buffers and verify both byte-for-byte.
    var both_eo = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 2);
    defer both_eo.free();
    var both_st = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 2);
    defer both_st.free();
    try both_eo.fill8(0xa5, self.rig.s.handle);
    try both_st.fill8(0xa5, self.rig.s.handle);
    try ce.launchGate(self.func, self.rig.s, self.gw, self.dev_sin, self.dev_proj, self.dev_mask, both_eo.ptr, both_st.ptr, self.shape.rows, self.shape.k, 0);
    try self.rig.s.synchronize();
    const got_eo = try a.alloc(u8, n * 2);
    defer a.free(got_eo);
    const got_st = try a.alloc(u8, n * 2);
    defer a.free(got_st);
    try both_eo.download(0, got_eo);
    try both_st.download(0, got_st);
    if (!std.mem.eql(u8, got_eo, std.mem.sliceAsBytes(ref_eo))) return "hy4_engram_fp8_gate combined engram_out bytes differ from serial FP32 oracle";
    if (!std.mem.eql(u8, got_st, std.mem.sliceAsBytes(ref_st))) return "hy4_engram_fp8_gate combined streams_out bytes differ from serial FP32 oracle";
    return null;
}
