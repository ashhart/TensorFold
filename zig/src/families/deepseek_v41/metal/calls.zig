//! Launches the layer encoders share, over the registered kernels (the phase-1 glue's bits).
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const pipelines = @import("pipelines.zig");

const Ref = ctx_.Ref;
const cv = ctx_.cv;
const hcm = pipelines.hc;
const f = pipelines.fused;
const gk = pipelines.glue_k;
const Pipelines = pipelines.Pipelines;

/// The hyper-connection mixer's constants (hc.zig's, glue.zig's own field for field).
pub const Hc = hcm.Hc;

/// Element strides of an act_quant round trip: value (a, b, k) reads a*sa + b*sb + k and writes a*ta + b*tb + k.
pub const Strides = struct { k: u32, nb: u32, na: u32 = 1, sa: u32 = 0, sb: u32, ta: u32 = 0, tb: u32 };

fn refs3(r: [3]Ref, comptime T: type) [3]T {
    return .{ cv(T, r[0]), cv(T, r[1]), cv(T, r[2]) };
}

/// The mixes of each row's four streams: pre [rows][4], post [rows][4], comb [rows][4][4].
pub fn hcmix(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: [3]Ref, out: [3]Ref, rows: u32, h: Hc) void {
    if (rows == 0) return;
    hcm.mix(&p.hc, e, cv(hcm.Ref, x), refs3(w, hcm.Ref), refs3(out, hcm.Ref), rows, h);
}

/// hc_pre: x [rows][4][dim] collapsed by the previous sublayer's pre mix into y [rows][dim].
pub fn hcpre(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, pre: Ref, y: Ref, rows: u32, dim: u32) void {
    if (rows == 0) return;
    hcm.pre(&p.hc, e, cv(hcm.Ref, x), cv(hcm.Ref, pre), cv(hcm.Ref, y), rows, dim);
}

/// hc_post: the branch [rows][dim] and the residual [rows][4][dim] into y [rows][4][dim].
pub fn hcpost(p: *const Pipelines, e: mtl.ComputeEncoder, branch: Ref, res: Ref, post: Ref, comb: Ref, y: Ref, rows: u32, dim: u32) void {
    if (rows == 0) return;
    hcm.post(&p.hc, e, cv(hcm.Ref, branch), cv(hcm.Ref, res), cv(hcm.Ref, post), cv(hcm.Ref, comb), cv(hcm.Ref, y), rows, dim);
}

/// y [n] bf16 = routed [n] fp32 + shared [n] bf16, one rounding.
pub fn join(p: *const Pipelines, e: mtl.ComputeEncoder, routed: Ref, shared: Ref, y: Ref, n: u32) void {
    if (n == 0) return;
    gk.moeJoin(&p.glue_k, e, cv(gk.Ref, routed), cv(gk.Ref, shared), cv(gk.Ref, y), n);
}

/// act_quant (E4M3, 32-value blocks, power-of-two scales) as a bf16 round trip, permuting by `s`; k a multiple of 32.
pub fn actq(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, s: Strides) !void {
    if (s.nb == 0 or s.na == 0) return;
    try f.actq(&p.fused, e, cv(f.Ref, x), cv(f.Ref, y), s);
}

/// act_quant of `rows` contiguous rows of `k`.
pub fn actqRows(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, k: u32) !void {
    try actq(p, e, x, y, .{ .k = k, .nb = rows, .sb = k, .tb = k });
}

/// RMSNorm of `rows` bf16 rows of `dim` (fp32 weight w) into y, and its act_quant round trip into yq.
pub fn normq(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, yq: Ref, rows: u32, dim: u32, eps: f32) !void {
    if (rows == 0) return;
    try f.normq(&p.fused, e, cv(f.Ref, x), cv(f.Ref, w), cv(f.Ref, y), cv(f.Ref, yq), rows, dim, eps);
}

/// The shared expert's clamped SwiGLU of n values into y and its act_quant round trip into yq; n a multiple of 32.
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, g: Ref, u: Ref, y: Ref, yq: Ref, n: u32, limit: f32) void {
    if (n == 0) return;
    f.swiglu(&p.fused, e, cv(f.Ref, g), cv(f.Ref, u), cv(f.Ref, y), cv(f.Ref, yq), n, limit);
}

/// A key row's kv_norm, RoPE and act_quant, into its ring rows `dest` (reps a row); the keys' dim is 512.
pub fn key(p: *const Pipelines, e: mtl.ComputeEncoder, b: f.Kv, rows: u32, dim: u32, eps: f32) !void {
    if (rows == 0) return;
    try f.kv(&p.fused, e, b, rows, dim, eps);
}
