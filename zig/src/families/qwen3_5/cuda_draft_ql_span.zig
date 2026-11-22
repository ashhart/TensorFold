//! Span views of a ModelOpt draft head: the stored tiles, on the whole head's K slices, for NVFP4 and plain FP8.
const cuda = @import("cuda");
const platform = @import("sys");
const qmmf = cuda.qmmf;
const Weight = cuda.qlinear.Weight;

/// TF_DFLASH_QL_SPAN=1 drafts a ModelOpt head over the spans; off by default.
pub fn enabled() bool {
    return platform.environment.read(bool, "TF_DFLASH_QL_SPAN", false) catch false;
}

/// Rows [from, from + n) of a packed FP4 or FP8 weight, `from` on a 64-row tile, summing in the whole weight's slices.
pub fn view(q: qmmf.Weight, from: usize, n: usize) ?qmmf.Weight {
    if (from % qmmf.group != 0 or n == 0 or from + n > q.n or q.k % qmmf.group != 0) return null;
    var out = q;
    switch (q.mode) {
        .fp4 => {
            out.codes = q.codes + from * q.k / 2;
            out.scales = q.scales + from * q.k / 16;
        },
        .fp8 => out.codes = q.codes + from * q.k,
        .mxfp8, .fp8g, .bf16 => return null, // a short view of these can select another tile kernel
    }
    out.n = @intCast(n);
    out.npad = @intCast((n + 127) / 128 * 128);
    return qmmf.pinned(out, qmmf.slices(q)) catch null;
}

pub const Views = struct { first: Weight, tail: Weight };

/// The first span and the tiles holding the tail span, or null when this head must stay whole.
pub fn views(w: Weight, first_n: usize, tail_at: usize, vocab: usize) ?Views {
    switch (w) {
        .nvfp4 => |q| {
            if (q.n != vocab) return null;
            const first = view(q, 0, first_n) orelse return null;
            const tail = view(q, tail_at, q.n - tail_at) orelse return null;
            return .{ .first = .{ .nvfp4 = first }, .tail = .{ .nvfp4 = tail } };
        },
        .fp8 => |q| {
            if (q.n != vocab or q.act != 0) return null;
            const first = view(q, 0, first_n) orelse return null;
            const tail = view(q, tail_at, q.n - tail_at) orelse return null;
            return .{ .first = .{ .fp8 = first }, .tail = .{ .fp8 = tail } };
        },
        else => return null,
    }
}
