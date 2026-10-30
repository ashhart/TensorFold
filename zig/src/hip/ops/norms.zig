//! Norm launches and the decode tails that fuse a norm.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const nemotron_moe = @import("../launches/nemotron_moe.zig");
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const int = t.int;

pub fn rms(o: Ops, x: Tensor, weight: ?u64, y: Tensor, rows: usize, width: usize, eps: f32) Error!void {
    if (x.kind != y.kind or width < 1 or width > 8192) return error.BadShape;
    if (rows == 0) return;
    try o.l.tf_rms(p(x.ptr), if (weight) |w| f(w) else null, p(y.ptr), @backingInt(x.kind), int(rows), int(width), eps, o.stream);
}

/// Two sets of fp32 rows (width at most 1024), each normed with its own weight, in one launch.
pub fn rms2(o: Ops, x0: u64, w0: u64, y0: u64, x1: u64, w1: u64, y1: u64, rows: usize, width: usize, eps: f32) Error!void {
    const z = o.l;
    try z.tf_rms2(f(x0), f(w0), f(y0), f(x1), f(w1), f(y1), int(rows), int(width), eps, o.stream);
}

/// The gated norm's gate: SiLU (Qwen3.5) or sigmoid (Flash Next's `output_gate_type`).
pub const Gate = enum(c_int) { silu = 0, sigmoid = 1 };

/// The linear attention's gated norm: out = rms(y) * weight * round(gate(z)) in z's kind, `width` at most 1024.
pub fn gnormOut(
    o: Ops,
    y: u64,
    weight: u64,
    z: Tensor,
    out: Tensor,
    rows: usize,
    width: usize,
    eps: f32,
    gate: Gate,
) Error!void {
    if (z.kind != out.kind or z.kind == .f32) return error.BadShape;
    const zig = o.l;
    try zig.tf_gnorm_out(
        f(y),
        f(weight),
        p(z.ptr),
        p(out.ptr),
        @backingInt(z.kind),
        int(rows),
        int(width),
        eps,
        @backingInt(gate),
        o.stream,
    );
}

/// x = round(x + y) and normed = rms(x) * weight (fp32) in one launch: a residual and the norm after it.
pub fn addRms(o: Ops, x: Tensor, y: Tensor, weight: u64, normed: Tensor, rows: usize, width: usize, eps: f32) Error!void {
    if (x.kind != y.kind or x.kind != normed.kind or x.kind == .f32) return error.BadShape;
    const z = o.l;
    try z.tf_tail(p(x.ptr), p(y.ptr), null, f(weight), p(normed.ptr), @backingInt(x.kind), int(rows), 0, int(width), eps, o.stream);
}

/// `add` (or `addWide` for an fp32 y), then `rms` into `normed`: one launch where it takes the rows, the same bits.
pub fn residualRms(
    o: Ops,
    x: Tensor,
    y: Tensor,
    weight: u64,
    normed: Tensor,
    rows: usize,
    width: usize,
    eps: f32,
) Error!void {
    if (x.kind == .f32 or x.kind != normed.kind or (y.kind != .f32 and y.kind != x.kind)) return error.BadShape;
    if (rows == 0) return;
    const wide = y.kind == .f32;
    if (merges(o, width)) return addRms1(
        o,
        x,
        .{ .y = y.ptr, .mode = @intFromBool(wide) },
        weight,
        normed,
        rows,
        width,
        eps,
    );
    if (wide) try o.addWide(x, y.ptr, x, rows * width) else try o.add(x, y, x, rows * width);
    try rms(o, x, weight, normed, rows, width, eps);
}

/// A Nemotron MoE layer's outputs before their sum: `slots` fp32 expert rows a row, their weights, the shared row.
pub const MoeRows = struct { y: u64, wts: u64, shared: u64, slots: usize };

/// `residualRms` of the rows' MoE sum (tf_nemo_combine's): one launch where it takes the rows, the same bits.
pub fn residualMoeRms(
    o: Ops,
    x: Tensor,
    m: MoeRows,
    weight: u64,
    normed: Tensor,
    rows: usize,
    width: usize,
    eps: f32,
) Error!void {
    if (x.kind == .f32 or x.kind != normed.kind or m.slots == 0) return error.BadShape;
    if (rows == 0) return;
    if (merges(o, width)) return addRms1(
        o,
        x,
        .{ .y = m.y, .mode = 2, .wts = m.wts, .shared = m.shared, .slots = int(m.slots) },
        weight,
        normed,
        rows,
        width,
        eps,
    );
    const sum: Tensor = .{ .ptr = try o.arena.take(rows * width * x.kind.size()), .kind = x.kind };
    try nemotron_moe.combine(
        o.l,
        @ptrFromInt(m.y),
        @ptrFromInt(m.wts),
        @ptrFromInt(m.shared),
        p(sum.ptr),
        @backingInt(x.kind),
        int(rows),
        int(m.slots),
        int(width),
        o.stream,
    );
    try residualRms(o, x, sum, weight, normed, rows, width, eps);
}

fn merges(o: Ops, width: usize) bool {
    return o.fused() and width > 1024 and width <= 8192;
}

fn addRms1(
    o: Ops,
    x: Tensor,
    in: @import("../launches/norms.zig").AddRmsIn,
    weight: u64,
    normed: Tensor,
    rows: usize,
    width: usize,
    eps: f32,
) Error!void {
    try o.l.tf_add_rms(
        p(x.ptr),
        in,
        f(weight),
        p(normed.ptr),
        @backingInt(x.kind),
        int(rows),
        int(width),
        eps,
        o.stream,
    );
}

pub fn gnormSilu(o: Ops, y: u64, z: Tensor, out: Tensor, n: usize, gate: Gate) Error!void {
    try o.l.tf_gnorm_silu(f(y), p(z.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), @backingInt(gate), o.stream);
}
