//! Explicit fast-prefill dispatch; live decode and verification never enter this module.
const cuda = @import("cuda");

pub fn affine(e: anytype, w: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !bool {
    const model = if (e.mo) |*m| m else return false;
    if (e.learning == null and @import("cuda_cublas_prompt.zig").owns(e, w.w)) if (model.vendor) |*v| {
        if (try v.affine4(e.stream, x, rows, w, out, .bf16)) return true;
    };
    const fast = if (model.fast8) |*n| n else return false;
    try fast.affine4(e.stream, x, rows, w, out, .bf16);
    return true;
}

pub fn swiglu(e: anytype, gate: anytype, up: @TypeOf(gate), x: u64, out: u64, rows: usize) !bool {
    if (gate != .ql or up != .ql or gate.ql != .nvfp4 or up.ql != .nvfp4) return false;
    const model = if (e.mo) |*m| m else return false;
    if (model.vendor != null and @import("cuda_cublas_prompt.zig").owns(e, gate.ql.nvfp4.codes) and
        cuda.cublaslt_prefill.supports(rows, gate.ql.nvfp4.n, gate.ql.nvfp4.k)) return false;
    const fast = if (model.native) |*n| n else return false;
    return fast.swiglu(e.stream, x, rows, gate.ql.nvfp4, up.ql.nvfp4, out);
}
