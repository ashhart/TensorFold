//! DeepSeek V4.1 glue on Metal: RMS norm, embedding, joins, RoPE and act-quant round trips, and the RoPE tables.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// One pipeline a kernel; every kernel's rows are independent, so no launch shape changes a bit.
pub const Pipelines = struct {
    rms_bf16: mtl.Pipeline,
    rms_f32: mtl.Pipeline,
    embed: mtl.Pipeline,
    join: mtl.Pipeline,
    mean: mtl.Pipeline,
    rope: mtl.Pipeline,
    fp8: mtl.Pipeline,
    fp4_e8m0: mtl.Pipeline,
    fp4_e4m3: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.glue);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.rms_bf16 = try mtl.Pipeline.init(device, lib, "tf_dsv41_rms_bf16w", false);
        p.rms_f32 = try mtl.Pipeline.init(device, lib, "tf_dsv41_rms_f32w", false);
        p.embed = try mtl.Pipeline.init(device, lib, "tf_dsv41_embed", false);
        p.join = try mtl.Pipeline.init(device, lib, "tf_dsv41_moe_join", false);
        p.mean = try mtl.Pipeline.init(device, lib, "tf_dsv41_hc_mean", false);
        p.rope = try mtl.Pipeline.init(device, lib, "tf_dsv41_rope", false);
        p.fp8 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_rt", false);
        p.fp4_e8m0 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp4_rt32", false);
        p.fp4_e4m3 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp4_rt16", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for ([_]mtl.Pipeline{ p.rms_bf16, p.rms_f32, p.embed, p.join, p.mean, p.rope, p.fp8, p.fp4_e8m0, p.fp4_e4m3 }) |x| x.deinit();
    }
};

pub const rms_threads = 256;

const RmsArgs = extern struct { D: u32, R: u32, eps: f32 };

/// y [rows][d] bf16 = RMS norm of x with weight w (bf16, or fp32 when `w32`); d any size, one threadgroup a row.
pub fn rms(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32, d: u32, eps: f32, w32: bool) void {
    e.setPipeline(if (w32) p.rms_f32 else p.rms_bf16);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.setValue(RmsArgs{ .D = d, .R = rows, .eps = eps }, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(rms_threads, 1, 1));
}

const EmbedArgs = extern struct { D: u32, R: u32, V: u32, reps: u32, lo: i32 };

/// out [n][reps][d] bf16 from the rank's slice table [v][d] (global ids lo .. lo + v); an id off it gives zeros.
pub fn embed(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, table: Ref, out: Ref, n: u32, d: u32, v: u32, lo: i32, reps: u32) void {
    e.setPipeline(p.embed);
    e.setBuffer(ids.buf, ids.off, 0);
    e.setBuffer(table.buf, table.off, 1);
    e.setBuffer(out.buf, out.off, 2);
    e.setValue(EmbedArgs{ .D = d, .R = n, .V = v, .lo = lo, .reps = reps }, 3);
    e.dispatchThreads(mtl.Size.of(d / 2, n, 1), mtl.Size.of(32, 1, 1));
}

/// y [n] bf16 = routed [n] fp32 + shared [n] bf16, one rounding.
pub fn moeJoin(p: *const Pipelines, e: mtl.ComputeEncoder, routed: Ref, shared: Ref, y: Ref, n: u32) void {
    e.setPipeline(p.join);
    e.setBuffer(routed.buf, routed.off, 0);
    e.setBuffer(shared.buf, shared.off, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.setValue(n, 3);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
}

const MeanArgs = extern struct { D: u32, R: u32, hc: u32 };

/// y [n][d] bf16 = mean over the hc copies of x [n][hc][d] bf16.
pub fn hcMean(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, n: u32, d: u32, hc: u32) void {
    e.setPipeline(p.mean);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(y.buf, y.off, 1);
    e.setValue(MeanArgs{ .D = d, .R = n, .hc = hc }, 2);
    e.dispatchThreads(mtl.Size.of(d, n, 1), mtl.Size.of(32, 1, 1));
}

/// Rows of `heads` heads, the rope tail at element `off` of each head; strides in bf16 elements, all even. x may be y.
pub const RopeCall = struct { rows: u32, heads: u32 = 1, pairs: u32 = 32, ldx: u32, hx: u32, ox: u32, ldy: u32, hy: u32, oy: u32 };

const RopeArgs = extern struct { R: u32, H: u32, pairs: u32, ldx: u32, hx: u32, ox: u32, ldy: u32, hy: u32, oy: u32 };

/// An in-place call on [rows][heads][head_dim] rows: the last 2 * pairs elements of each head.
pub fn ropeInPlace(rows: u32, heads: u32, head_dim: u32, pairs: u32) RopeCall {
    const tail = head_dim - 2 * pairs;
    return .{ .rows = rows, .heads = heads, .pairs = pairs, .ldx = heads * head_dim, .hx = head_dim, .ox = tail, .ldy = heads * head_dim, .hy = head_dim, .oy = tail };
}

pub fn ropeCheck(c: RopeCall) !void {
    for ([_]u32{ c.ldx, c.hx, c.ox, c.ldy, c.hy, c.oy }) |v| if (v % 2 != 0) return error.BadRopeStride;
    if (c.rows == 0 or c.heads == 0 or c.pairs == 0 or c.pairs > 32) return error.BadRopeShape;
}

/// pos [rows] i32 indexes tab [position][pairs] float2 (cos, sin); the tail of each head is rotated into y.
pub fn rope(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, pos: Ref, tab: Ref, c: RopeCall) !void {
    try ropeCheck(c);
    e.setPipeline(p.rope);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(y.buf, y.off, 1);
    e.setBuffer(pos.buf, pos.off, 2);
    e.setBuffer(tab.buf, tab.off, 3);
    e.setValue(RopeArgs{ .R = c.rows, .H = c.heads, .pairs = c.pairs, .ldx = c.ldx, .hx = c.hx, .ox = c.ox, .ldy = c.ldy, .hy = c.hy, .oy = c.oy }, 4);
    e.dispatchThreads(mtl.Size.of(c.pairs, c.heads, c.rows), mtl.Size.of(32, 1, 1));
}

/// fp8: act_quant blocks of 32; fp4_e8m0: indexer q and k, blocks of 32; fp4_e4m3: compressed KV, blocks of 16.
pub const Quant = enum { fp8, fp4_e8m0, fp4_e4m3 };

const QuantArgs = extern struct { N: u32, R: u32 };

/// y [rows][n] bf16 = the round trip of x [rows][n] bf16; n a multiple of 32. x may be y (a block is read whole first).
pub fn quant(p: *const Pipelines, e: mtl.ComputeEncoder, kind: Quant, x: Ref, y: Ref, rows: u32, n: u32) !void {
    if (rows == 0 or n == 0 or n % 32 != 0) return error.BadQuantShape;
    e.setPipeline(switch (kind) {
        .fp8 => p.fp8,
        .fp4_e8m0 => p.fp4_e8m0,
        .fp4_e4m3 => p.fp4_e4m3,
    });
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(y.buf, y.off, 1);
    e.setValue(QuantArgs{ .N = n, .R = rows }, 2);
    e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(32, 1, 1));
}

/// A RoPE spectrum (model.py:369): inverse frequencies theta^(-2i/dim), YaRN-blended when `original` is not 0.
pub const RopeSpec = struct {
    theta: f64,
    dim: u32 = 64,
    factor: f32 = 1,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    original: u32 = 0,

    /// Layers without compression: plain rope_theta, no YaRN.
    pub const sliding: RopeSpec = .{ .theta = 10000 };
    /// Compressed layers: compress_rope_theta with the release's rope_scaling.
    pub const compressed: RopeSpec = .{ .theta = 160000, .factor = 16, .original = 65536 };
};

fn correctedDim(s: RopeSpec, rotations: f64) f64 {
    const dim: f64 = @floatFromInt(s.dim);
    return dim * @log(@as(f64, @floatFromInt(s.original)) / (rotations * 2 * std.math.pi)) / (2 * @log(s.theta));
}

/// The spectrum's dim / 2 fp32 frequencies, each step rounded to fp32 as torch's eager ops do.
pub fn ropeFreqs(s: RopeSpec, out: []f32) void {
    const dim: f32 = @floatFromInt(s.dim);
    for (out, 0..) |*f, i| {
        const e = @as(f32, @floatFromInt(2 * i)) / dim;
        f.* = 1.0 / @as(f32, @floatCast(std.math.pow(f64, s.theta, e)));
    }
    if (s.original == 0) return;
    const low = @max(@floor(correctedDim(s, s.beta_fast)), 0);
    const high = @min(@ceil(correctedDim(s, s.beta_slow)), @as(f64, @floatFromInt(s.dim)) - 1);
    const span: f32 = @floatCast(@max(high - low, 1e-3));
    for (out, 0..) |*f, i| {
        const ramp = std.math.clamp((@as(f32, @floatFromInt(i)) - @as(f32, @floatCast(low))) / span, 0, 1);
        const smooth = 1 - ramp;
        f.* = f.* / s.factor * (1 - smooth) + f.* * smooth;
    }
}

/// Position `pos`'s (cos, sin) pairs: the fp32 angle pos * freq, then its cos and sin rounded once to fp32.
pub fn ropeRow(freqs: []const f32, pos: u32, out: [][2]f32) void {
    for (freqs, out) |f, *o| {
        const angle: f64 = @as(f32, @floatFromInt(pos)) * f;
        o.* = .{ @floatCast(@cos(angle)), @floatCast(@sin(angle)) };
    }
}

/// The table [positions][dim / 2] of (cos, sin) the rope kernel reads, built for one spectrum.
pub fn ropeTable(a: std.mem.Allocator, s: RopeSpec, positions: u32) ![][2]f32 {
    const freqs = try a.alloc(f32, s.dim / 2);
    defer a.free(freqs);
    ropeFreqs(s, freqs);
    const out = try a.alloc([2]f32, @as(usize, positions) * freqs.len);
    for (0..positions) |p| ropeRow(freqs, @intCast(p), out[p * freqs.len ..][0..freqs.len]);
    return out;
}

test "the two spectra: sliding is plain theta, compressed divides the slow end by the factor and keeps the fast end" {
    var plain: [32]f32 = undefined;
    var yarn: [32]f32 = undefined;
    ropeFreqs(RopeSpec.sliding, &plain);
    ropeFreqs(RopeSpec.compressed, &yarn);
    try std.testing.expectEqual(@as(f32, 1.0), plain[0]);
    try std.testing.expectEqual(@as(f32, 1.0), yarn[0]);
    try std.testing.expectApproxEqRel(@as(f32, 1.3335e-4), plain[31], 1e-3);
    var base: [32]f32 = undefined;
    ropeFreqs(.{ .theta = 160000 }, &base);
    try std.testing.expectApproxEqRel(base[31] / 16, yarn[31], 1e-6);
    try std.testing.expectEqual(base[0], yarn[0]);
}
