//! The linear attention: the conv, the gates and the DeltaNet recurrence.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const int = t.int;

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

/// The DeltaNet recurrence in fp32, state in place, the same walk at every length.
pub fn gatedDelta(o: Ops, q: u64, k: u64, v: u64, gate: u64, beta: u64, state: u64, y: u64, length: usize, key_heads: usize, value_heads: usize, dk: usize, dv: usize, states: ?u64) Error!void {
    try o.l.tf_gated_delta(f(q), f(k), f(v), f(gate), f(beta), f(state), f(y), 1, int(length), int(key_heads), int(value_heads), int(dk), int(dv), o.stream, if (states) |s| f(s) else null);
}
