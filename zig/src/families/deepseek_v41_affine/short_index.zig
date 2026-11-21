//! Finite bounds and ordered selection for a verifier's known-visible prefix.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const Enc = @import("forward.zig").Enc;
const Model = @import("model.zig").Model;

const Args = extern struct {
    rows: u32,
    base: u32,
    ratio: u32,
    fill_keep: u32,
    force_replay: u32,
};

// Guard all activations at 2^8. A group-64 affine weight is bounded

// by 2^16 even at q8. Across at most 8192 terms the query projection

// is below 2^38 including rounding. RoPE and FP4 raise this bound to

// 2^43. Keys below 2^8 give a dot below 2^59. The separate head

// weight is below 2^30, so the rounded head sum stays below 2^96.
fn within(values: []const u16) bool {
    for (values) |v| if ((v & 0x7fff) > 0x4380) return false;
    return true;
}

pub fn compatible(m: *const Model) bool {
    comptime std.debug.assert(c.index_heads == 32 and c.index_dim == 128 and
        st.picks == 512 and c.candidate_blocks == 2048 and c.candidate_block == 8 and
        c.hidden == 5120 and c.q_rank == 1280 and st.ring > st.picks * 2);
    for (m.freq_compressed.slice(f32, c.rope_dim / 2)) |f| {
        if (!std.math.isFinite(f) or @abs(f) > 256.0) return false;
    }
    for (&m.weights.layers) |*layer| {
        const ix = layer.attn.indexer orelse continue;
        const w = ix.wq_b;
        if (w.bits > 8 or w.k != c.q_rank or w.group != 64) return false;
        const n = w.n * (w.k / w.group);
        const s0 = w.s_off / 2;
        const b0 = w.b_off / 2;
        if (!within(w.scales.slice(u16, s0 + n)[s0..]) or
            !within(w.biases.slice(u16, b0 + n)[b0..]) or
            !within(ix.weights_proj.host(u16))) return false;
    }
    return true;
}

pub fn emit(x: Enc, source: usize, ratio: usize, layer: usize) void {
    const s = x.s;
    const cache = x.segs[0].cache;
    const keys = (x.base + x.rows) / ratio;
    const keep = layer == c.candidate_source;
    const audit = x.m.short_index_audit;
    const force = x.m.short_index_force != null and x.base >= x.m.short_index_force.? and cache.short_index_replays == 0;
    if (audit) x.pipe("tf_short_index_audit") else x.pipe("tf_short_index_fill");
    x.buf(s.qr, 0, 0);
    x.buf(s.normed, 0, 1);
    x.buf(cache.sources[source].keys, 0, 2);
    x.buf(s.picks, 0, 3);
    x.buf(s.keep, 0, 4);
    x.buf(s.short_index_status, 0, 5);
    x.args(Args{
        .rows = @intCast(x.rows),
        .base = @intCast(x.base),
        .ratio = @intCast(ratio),
        .fill_keep = @intFromBool(keep),
        .force_replay = @intFromBool(force),
    }, 6);
    const scans = (x.rows * (c.q_rank + c.hidden) + keys * c.index_dim) / 8;
    const outputs_per_row: usize = if (keep) c.candidate_blocks else st.picks;
    const output = x.rows * outputs_per_row;
    x.threads(.{ @max(scans, output), 1, 1 }, .{ 256, 1, 1 });
    cache.short_index_layers += 1;
    if (audit) cache.short_index_checked += x.rows * (st.picks + if (keep) c.candidate_blocks else @as(usize, 0));
}
