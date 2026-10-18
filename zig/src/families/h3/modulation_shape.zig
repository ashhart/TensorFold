//! H3 modulation dimensions checked on the host before division, indexing or GPU allocation.
const std = @import("std");
const st = @import("safetensors");
const geometry = @import("geometry.zig");

pub fn linear(x: usize, w: usize, n: usize, rows: usize, k: usize) !usize {
    try geometry.dimension(n);
    try geometry.dimension(k);
    if (x != try geometry.product(&.{ rows, k }) or w != try geometry.product(&.{ n, k }))
        return error.H3BadEmbeddingShape;
    const count = try geometry.product(&.{ rows, n });
    _ = try geometry.product(&.{ count, 4 });
    return count;
}

pub fn embedding(w: st.Entry, b: st.Entry, out: st.Entry, bias: st.Entry, times: usize) !usize {
    if (w.rank != 2 or out.rank != 2 or b.rank != 1 or bias.rank != 1 or
        w.shape[0] != b.shape[0] or out.shape[1] != b.shape[0] or out.shape[0] != bias.shape[0])
        return error.H3BadEmbeddingShape;
    for ([_]st.Entry{ w, b, out, bias }) |e| {
        if (e.dtype != .bf16 and e.dtype != .f32) return error.H3BadEmbeddingShape;
        for (e.shape[0..e.rank]) |n| try geometry.dimension(n);
    }
    const dim = w.shape[1];
    if (dim % 2 != 0) return error.H3BadEmbeddingShape;
    _ = try geometry.product(&.{ times, dim, 4 });
    _ = try geometry.product(&.{ times, b.shape[0], 4 });
    _ = try geometry.product(&.{ times, bias.shape[0], 4 });
    return dim;
}

pub fn projection(t: usize, k: usize, groups: usize, parts: usize, width: usize) !usize {
    for ([_]usize{ t, k, groups, parts, width }) |n| try geometry.dimension(n);
    const n = try geometry.product(&.{ groups, parts, width });
    try geometry.dimension(n);
    _ = try geometry.product(&.{ n, k, 2 });
    _ = try geometry.product(&.{ t, n, 2 });
    return n;
}

pub fn tables(t: usize, hidden: usize) !usize {
    try geometry.dimension(t);
    const lines = try geometry.product(&.{ t, 3 });
    try geometry.dimension(lines);
    _ = try geometry.product(&.{ 6, lines, hidden, 2 });
    _ = try geometry.product(&.{ t, 2, hidden, 2 });
    return lines;
}
