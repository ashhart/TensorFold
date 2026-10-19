//! A training pass's hooks in the 27B's batched prompt pass: per-layer keeps, the MLP hook and every row's logits.
const std = @import("std");
const cuda = @import("cuda");
const c = @import("cuda_learn_shape.zig");
const to = @import("cuda_train_ops.zig");

/// After layer `layer`'s MLP: out gains what the hook adds; `taught` (f32 a row, 0: every row) zeroes untaught rows.
pub const MlpAdd = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, layer: usize, y: u64, out: u64, rows: usize, taught: u64) anyerror!void,
};

/// One layer's kept bf16 rows (0: not kept), named as the backward reads them (train.zig).
pub const LayerKeep = struct { h_in: u64 = 0, proj: u64 = 0, pq: u64 = 0, pk: u64 = 0, pv: u64 = 0, gated: u64 = 0, h_mid: u64 = 0, y: u64 = 0, gate: u64 = 0, up: u64 = 0 };

pub const Keep = struct {
    t: to.Ops, // the trainer's kernels (column copies)
    d: *const cuda.Driver,
    g: c.Geometry,
    layers: []const LayerKeep,
    at: usize = 0,
    final_h: u64 = 0, // [rows, hidden] the residual into the final norm
    last: u64 = 0, // [rows, hidden] every row normed, then the head's input
    last_xs: u64 = 0, // [rows, hidden / 64] fp32 its group sums
    logits: u64 = 0, // [rows, vocab] every row's logits (0: none)

    fn copy(k: *const Keep, dst: u64, src: u64, bytes: usize) !void {
        if (dst == 0 or bytes == 0) return;
        try k.d.check(k.d.api.cuMemcpyDtoDAsync_v2(dst + k.at * bytes, src, bytes, k.t.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn input(k: *const Keep, i: usize, h: u64, rows: usize) !void {
        try k.copy(k.layers[i].h_in, h, rows * k.g.hidden * 2);
    }

    /// The DeltaNet in-projection's four outputs side by side, as gdn_train's P reads them.
    pub fn delta(k: *const Keep, i: usize, qkv: u64, z: u64, b: u64, a: u64, rows: usize) !void {
        const base = k.layers[i].proj;
        const p = if (base == 0) 0 else base + k.at * (k.g.convDim() + k.g.vInner() + 2 * k.g.linear_v_heads) * 2;
        if (p == 0) return;
        const g = k.g;
        const conv = g.convDim();
        const width = conv + g.vInner() + 2 * g.linear_v_heads;
        try k.t.cols(qkv, conv, p, width, 0, rows);
        try k.t.cols(z, g.vInner(), p, width, conv, rows);
        try k.t.cols(b, g.linear_v_heads, p, width, conv + g.vInner(), rows);
        try k.t.cols(a, g.linear_v_heads, p, width, conv + g.vInner() + g.linear_v_heads, rows);
    }

    pub fn attention(k: *const Keep, i: usize, qg: u64, key: u64, v: u64, rows: usize) !void {
        const l = k.layers[i];
        try k.copy(l.pq, qg, rows * 2 * k.g.qInner() * 2);
        try k.copy(l.pk, key, rows * k.g.kvInner() * 2);
        try k.copy(l.pv, v, rows * k.g.kvInner() * 2);
    }

    pub fn gated(k: *const Keep, i: usize, x: u64, rows: usize) !void {
        try k.copy(k.layers[i].gated, x, rows * k.g.qInner() * 2);
    }

    pub fn post(k: *const Keep, i: usize, h: u64, y: u64, rows: usize) !void {
        try k.copy(k.layers[i].h_mid, h, rows * k.g.hidden * 2);
        try k.copy(k.layers[i].y, y, rows * k.g.hidden * 2);
    }

    pub fn mlp(k: *const Keep, i: usize, gate: u64, up: u64, rows: usize) !void {
        try k.copy(k.layers[i].gate, gate, rows * k.g.intermediate * 2);
        try k.copy(k.layers[i].up, up, rows * k.g.intermediate * 2);
    }

    pub fn final(k: *const Keep, h: u64, rows: usize) !void {
        try k.copy(k.final_h, h, rows * k.g.hidden * 2);
    }
};

/// The owned engine's logits, before the next row overwrites its scratch.
pub fn logits(k: *const Keep, src: u64, stream: cuda.Stream) !void {
    if (k.logits == 0) return;
    try k.d.check(k.d.api.cuMemcpyDtoDAsync_v2(k.logits + k.at * c.vocab * 2, src, c.vocab * 2, stream.handle), "keep logits");
}
