//! The linear attention: the conv, the gates and the DeltaNet recurrence.

const t = @import("types.zig");
const launches = @import("../launches.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const int = t.int;

/// Rows from which the DeltaNet prefill runs chunked, and the rows of one chunked segment (a multiple of the chunk).
const gdn_min_rows = 64;
const gdn_segment = 4096;

pub fn convPrefill(o: Ops, x: Tensor, weight: u64, state: ?u64, out: u64, new_state: u64, len: usize, channels: usize, kernel: usize) Error!void {
    if (kernel < 1 or kernel > 8) return error.BadShape;
    try o.l.tf_conv_prefill(p(x.ptr), @backingInt(x.kind), f(weight), if (state) |s| f(s) else null, f(out), f(new_state), int(len), int(channels), int(kernel), o.stream);
}

pub fn convDecode(o: Ops, x: u64, weight: u64, state: u64, y: u64, channels: usize, kernel: usize) Error!void {
    try o.l.tf_conv_decode(f(x), f(weight), f(state), f(y), 1, int(channels), int(kernel), o.stream);
}

pub fn convRows(o: Ops, x: u64, weight: u64, state: u64, y: u64, states: ?u64, rows: usize, channels: usize, kernel: usize) Error!void {
    try o.l.tf_conv_rows(f(x), f(weight), f(state), f(y), if (states) |s| f(s) else null, int(rows), int(channels), int(kernel), o.stream);
}

pub fn gdnGatePrefill(o: Ops, a: Tensor, b: Tensor, a_log: u64, dt_bias: u64, gate: u64, beta: u64, count: usize, heads: usize) Error!void {
    try o.l.tf_gdn_gate_prefill(p(a.ptr), p(b.ptr), @backingInt(a.kind), f(a_log), f(dt_bias), f(gate), f(beta), int(count), int(heads), o.stream);
}

/// The fused gate over `elements` values of a and b (rows times heads).
pub fn gdnGate(o: Ops, a: Tensor, b: Tensor, a_log: u64, dt_bias: u64, gate: u64, beta: u64, elements: usize, heads: usize) Error!void {
    try o.l.tf_gdn_gate(p(a.ptr), p(b.ptr), @backingInt(a.kind), f(a_log), f(dt_bias), f(gate), f(beta), int(elements), int(heads), o.stream);
}

/// The chunked recurrence in segments of `gdn_segment` rows, whose scratch the arena hands back after each.
fn gatedDeltaChunked(o: Ops, z: *const launches.Launcher, q: u64, k: u64, v: u64, gate: u64, beta: u64, state: u64, y: u64, length: usize, key_heads: usize, value_heads: usize) Error!void {
    var start: usize = 0;
    while (start < length) : (start += gdn_segment) {
        const rows = @min(gdn_segment, length - start);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const scratch = try o.arena.take(launches.gdnScratch(rows, key_heads, value_heads).total);
        const qk = start * key_heads * 128 * 4;
        const vy = start * value_heads * 128 * 4;
        const gb = start * value_heads * 4;
        try z.gdnChunked(f(q + qk), f(k + qk), f(v + vy), f(gate + gb), f(beta + gb), f(state), f(y + vy), rows, key_heads, value_heads, scratch, o.stream);
    }
}

/// The DeltaNet recurrence, state in place; a prefill of `gdn_min_rows` rows or more runs chunked unless reference.
pub fn gatedDelta(o: Ops, q: u64, k: u64, v: u64, gate: u64, beta: u64, state: u64, y: u64, length: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize, states: ?u64) Error!void {
    if (states == null and (length >= gdn_min_rows or o.prefill) and dk == 128 and dv == 128 and value_heads % key_heads == 0) {
        if (o.l.chunked) return gatedDeltaChunked(o, o.l, q, k, v, gate, beta, state, y, length, key_heads, value_heads);
    }
    try o.l.tf_gated_delta(f(q), f(k), f(v), f(gate), f(beta), f(state), f(y), 1, int(length), int(key_heads), int(value_heads), int(dk), int(dv), o.stream, if (states) |s| f(s) else null);
}
