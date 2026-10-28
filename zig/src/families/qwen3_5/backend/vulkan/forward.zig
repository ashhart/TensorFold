//! The Qwen3.5-2B Vulkan forward composition: one single-sequence pass over
//! the pinned 24-layer tower (eighteen linear GatedDeltaNet layers, six full
//! attention layers every 4th), matching the unchanged Ash8c Metal oracle
//! (zig/src/families/qwen3_5/forward.zig in the ash tree) op for op.
//!
//! The oracle's fused norm kernel adds first (`h = bf16(H + R)` written back,
//! then `x = bf16(gain * rmsnorm(h))`), so this composition is an explicit
//! bf16 residual add followed by the raw-BF16-gain RMSNorm — never a `1 + w`
//! gain and never a norm-then-add. Every mixer writes its output into the
//! residual companion the next add consumes; the residual ring never aliases
//! a read with a write. The final norm touches only the last row; the tied
//! or untied head projects that one row onto the full vocab.
//!
//! Full attention reads the oracle's interleaved Q layout: the Q projection
//! holds [M, 4096] rows of 8 interleaved `[q 256 | gate 256]` halves; Q is
//! extracted strided on the GPU through the existing copy_u16 pipeline before
//! the per-head RMS gamma (256). RoPE, the flat KV cache write, causal GQA
//! SDPA and the sigmoid output gate ride the admitted attention operator.
//!
//! The linear layers ride the admitted GDN operator: the 4-tap conv over the
//! raw BF16 taps (both paths), the gate split (g F32, beta BF16), the chunked
//! prefill scan or the raw-gates decode step, and the gated RMSNorm epilogue.
//! Conv state and recurrence state are the caller's ping-pong halves; the
//! caller flips only after the writes' fences and commits visible length only
//! after the whole forward.
//!
//! Every GPU call is routed through `vk.orExit` at the call site: a Timeout
//! or InFlight ends the process before any destructor can free a resource the
//! GPU may still be using. Nothing here falls back to the CPU.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");

const Arena = vk.arena.Arena;
const View = vk.tensor.View;

/// Whole-prefill cap: every prefill is one chunk of at most this many rows.
pub const max_prefill_rows: usize = 32;
/// The GDN prefill scan's chunk cap (chunks of 1 when a prefill fits one).
pub const max_gdn_chunk: u64 = 32;

comptime {
    if (2 * pinned.query_heads * pinned.head_dim != 4096)
        @compileError("the interleaved Q layout drifted from [q 256 | gate 256] x 8");
    if (pinned.conv_dim != 2 * pinned.linear_heads * pinned.linear_dim + pinned.linear_heads * pinned.linear_dim)
        @compileError("GDN channel order drifted from q|k|v");
}

/// The admitted operators the composition drives.
pub const Acc = struct {
    ops: *vk.ops.Ops,
    norm: *vk.norm_mlp.NormMlp,
    attn: *vk.Attention,
    gdn: *vk.Gdn,
};

/// Everything one forward pass needs beyond the weights and state: the
/// command stream, the shared staging buffer, the operators, the scratch
/// arena and the fence timeout. The arena and every view are the owner's;
/// this struct only borrows them.
pub const Core = struct {
    cmds: *vk.Commands,
    staging: vk.Buffer,
    acc: Acc,
    scratch: *Arena,
    timeout_ns: u64,

    /// One affine projection through the admitted Q4 closure.
    fn project(c: *const Core, out: View, x: View, q: weights.Quant) !void {
        try vk.orExit(vk.ops_q4.project(c.acc.ops, c.cmds, out, x, q.w, q.scales, q.biases, q.k, c.scratch, c.timeout_ns));
    }

    /// The oracle's fused norm, split into its two exact roundings: first
    /// the bf16 residual add (`h = bf16(H + R)`, written out of place into
    /// `sum`), then the raw-gain RMSNorm into `out`.
    pub fn addNorm(c: *const Core, out: View, sum: View, h: View, r: View, gain: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.add(c.cmds, sum, h, r, c.timeout_ns));
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, out, sum, gain, rows, pinned.hidden, pinned.eps, c.timeout_ns));
    }

    /// The layer-0 norm: no residual, the gain over the raw embedding.
    pub fn plainNorm(c: *const Core, out: View, h: View, gain: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, out, h, gain, rows, pinned.hidden, pinned.eps, c.timeout_ns));
    }

    /// Q extraction from the interleaved qraw: out element (d, k) reads
    /// qraw[d + k * 512], k = row * 8 + head, exactly the oracle's
    /// qwen35_head_norm source indexing, as one 2D strided copy.
    pub fn extractQueries(c: *const Core, q_out: View, qraw: View, rows: usize) !void {
        const plan = extractQPlan(rows, q_out.elements, qraw.elements) orelse return error.Invalid;
        try c.acc.ops.copy_u16.bind(&.{ qraw.bind(), c.acc.ops.dummyBinding(), q_out.bind(), c.acc.ops.dummyBinding() });
        var p: vk.params.Params = .{
            .count = plan.count,
            .dims = 2,
            .shape = .{ plan.dim, plan.heads_rows, 0, 0 },
            .in_strides = .{ 1, plan.in_head_stride, 0, 0 },
            .out_strides = .{ 1, plan.dim, 0, 0 },
        };
        try vk.orExit(c.acc.ops.copy_u16.run(c.cmds, std.mem.asBytes(&p), vk.ops.dispatchGroups(plan.count, c.acc.ops.copy_u16.local[0]), c.timeout_ns));
    }

    /// One full-attention layer: Q/K/V projections, interleaved-Q extraction,
    /// per-head RMS gamma (256), RoPE, the flat KV write at `base`, causal
    /// GQA SDPA over the visible prefix, the output gate and the O projection
    /// into `r_out`. `r_out` must not alias anything read here.
    pub fn fullLayer(c: *const Core, x: View, layer: weights.FullLayer, s: *state.State, slot: usize, rows: usize, r_out: View, base: u64) !void {
        const mark = c.scratch.mark();
        errdefer c.scratch.reset(mark) catch {};
        const hidden = pinned.hidden;
        const qraw = try c.scratch.alloc(.bf16, rows * 2 * pinned.query_heads * pinned.head_dim);
        try c.project(qraw, x, layer.q);
        const q = try c.scratch.alloc(.bf16, rows * pinned.query_heads * pinned.head_dim);
        try c.extractQueries(q, qraw, rows);
        const qn = try c.scratch.alloc(.bf16, q.elements);
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, qn, q, layer.q_norm, rows * pinned.query_heads, pinned.head_dim, pinned.eps, c.timeout_ns));
        const k = try c.scratch.alloc(.bf16, rows * pinned.kv_heads * pinned.head_dim);
        try c.project(k, x, layer.k);
        const kn = try c.scratch.alloc(.bf16, k.elements);
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, kn, k, layer.k_norm, rows * pinned.kv_heads, pinned.head_dim, pinned.eps, c.timeout_ns));
        const v = try c.scratch.alloc(.bf16, k.elements);
        try c.project(v, x, layer.v);
        const queries = try c.scratch.alloc(.bf16, q.elements);
        try vk.orExit(c.acc.attn.setBasePosition(c.cmds, c.staging, @intCast(base), c.timeout_ns));
        try vk.orExit(c.acc.attn.ropeQueries(c.cmds, queries, qn, rows, c.timeout_ns));
        try vk.orExit(c.acc.attn.cacheKV(c.cmds, s.keysFor(slot), s.valuesFor(slot), kn, v, rows, base, s.capacity, c.timeout_ns));
        const y = try c.scratch.alloc(.bf16, q.elements);
        try vk.orExit(c.acc.attn.attend(c.cmds, y, queries, s.keysFor(slot), s.valuesFor(slot), base, rows, s.capacity, c.timeout_ns));
        const mix = try c.scratch.alloc(.bf16, rows * hidden);
        try vk.orExit(c.acc.attn.gateOutput(c.cmds, mix, y, qraw, rows, c.timeout_ns));
        try c.project(r_out, mix, layer.o);
        c.scratch.reset(mark) catch {};
    }

    /// One linear GatedDeltaNet layer: the qkv/z/a/b projections, the conv
    /// (prefill assemble or decode step) over the raw BF16 taps, the gate
    /// split, the prefill scan or the raw-gates decode step, the gated
    /// RMSNorm epilogue and the out projection. Conv and recurrence halves
    /// are the state's ping-pong pair; the slot flips after the writes.
    pub fn linearLayer(c: *const Core, x: View, layer: weights.LinearLayer, s: *state.State, slot: usize, rows: usize, r_out: View) !void {
        const mark = c.scratch.mark();
        errdefer c.scratch.reset(mark) catch {};
        const dim = pinned.linear_heads * pinned.linear_dim;
        const qkv = try c.scratch.alloc(.bf16, rows * pinned.conv_dim);
        try c.project(qkv, x, layer.qkv);
        const z = try c.scratch.alloc(.bf16, rows * dim);
        try c.project(z, x, layer.z);
        const a = try c.scratch.alloc(.bf16, rows * pinned.linear_heads);
        try c.project(a, x, layer.a);
        const b = try c.scratch.alloc(.bf16, rows * pinned.linear_heads);
        try c.project(b, x, layer.b);
        const soa = try c.scratch.alloc(.bf16, rows * 3 * dim);
        const g = try c.scratch.alloc(.f32, rows * pinned.linear_heads);
        const beta = try c.scratch.alloc(.bf16, rows * pinned.linear_heads);
        if (rows == 1) {
            const activated = try c.scratch.alloc(.bf16, pinned.conv_dim);
            try vk.orExit(c.acc.gdn.convDecodeStep(c.cmds, soa, g, beta, activated, s.convInput(slot), qkv, layer.conv, a, b, layer.a_log, layer.dt_bias, s.convOutput(slot), c.timeout_ns));
            const y = try c.scratch.alloc(.bf16, dim);
            try vk.orExit(c.acc.gdn.decodeRecur(c.cmds, soa, a, b, try aLogU16(layer.a_log), layer.dt_bias, s.gdnInput(slot), s.gdnOutput(slot), y, c.timeout_ns));
            const mix = try c.scratch.alloc(.bf16, dim);
            try vk.orExit(c.acc.gdn.gatedNorm(c.cmds, mix, y, z, layer.gnorm, rows, pinned.eps, c.timeout_ns));
            try c.project(r_out, mix, layer.out);
        } else {
            const conv_out = try c.scratch.alloc(.bf16, rows * pinned.conv_dim);
            const conv_in = try c.scratch.alloc(.bf16, (rows + state.conv_rows) * pinned.conv_dim);
            try vk.orExit(c.acc.gdn.convPrefill(c.cmds, soa, g, beta, conv_out, conv_in, s.convInput(slot), qkv, layer.conv, a, b, layer.a_log, layer.dt_bias, s.convOutput(slot), rows, c.timeout_ns));
            const y = try c.scratch.alloc(.bf16, rows * dim);
            try vk.orExit(c.acc.gdn.prefillRecur(c.cmds, soa, g, beta, s.gdnInput(slot), y, s.gdnOutput(slot), null, rows, @min(rows, max_gdn_chunk), c.timeout_ns));
            const mix = try c.scratch.alloc(.bf16, rows * dim);
            try vk.orExit(c.acc.gdn.gatedNorm(c.cmds, mix, y, z, layer.gnorm, rows, pinned.eps, c.timeout_ns));
            try c.project(r_out, mix, layer.out);
        }
        s.flip(slot);
        c.scratch.reset(mark) catch {};
    }

    /// The dense MLP: Q4 gate and up projections, the ASH single-round
    /// SwiGLU, and the Q4 down projection into `r_out`.
    pub fn mlp(c: *const Core, x: View, rows: usize, m: weights.DenseMlp, r_out: View) !void {
        const mark = c.scratch.mark();
        errdefer c.scratch.reset(mark) catch {};
        const gate = try c.scratch.alloc(.bf16, rows * pinned.intermediate);
        try c.project(gate, x, m.gate);
        const up = try c.scratch.alloc(.bf16, rows * pinned.intermediate);
        try c.project(up, x, m.up);
        const act = try c.scratch.alloc(.bf16, rows * pinned.intermediate);
        try vk.orExit(c.acc.norm.swiglu(c.cmds, act, gate, up, c.timeout_ns));
        try c.project(r_out, act, m.down);
        c.scratch.reset(mark) catch {};
    }

    /// The last row only: the final residual add, the final raw-gain norm,
    /// and the head projection (tied embedding or separate lm_head) onto the
    /// full vocab row the greedy draw reads.
    pub fn finish(c: *const Core, sum: View, h: View, r: View, x: View, final_norm: View, head: weights.Quant, logits: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.add(c.cmds, sum, h, r, c.timeout_ns));
        const last = (rows - 1) * pinned.hidden;
        const x_last = try x.slice(last, pinned.hidden);
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, x_last, try sum.slice(last, pinned.hidden), final_norm, 1, pinned.hidden, pinned.eps, c.timeout_ns));
        try c.project(logits, x_last, head);
    }
};

/// The f32 A_log view as u16 pairs, the decode kernel's sanctioned transport
/// (32 elements over the same 64 bytes; never a value reinterpretation).
fn aLogU16(a_log: View) !View {
    if (a_log.kind != .f32 or a_log.elements != pinned.linear_heads) return error.Invalid;
    return View.init(a_log.buffer, .u16, a_log.byte_offset, 2 * pinned.linear_heads);
}

/// The Q extraction's derived geometry, or null when any view is torn; pure
/// and host-tested. `dim` = head_dim, `heads_rows` = rows * query_heads,
/// `in_head_stride` = the interleaved half pitch 2 * head_dim.
pub const ExtractQPlan = struct { count: u32, dim: u32, heads_rows: u32, in_head_stride: u32 };

pub fn extractQPlan(rows: usize, out_elements: u64, in_elements: u64) ?ExtractQPlan {
    if (rows == 0 or rows > max_prefill_rows) return null;
    const dim: u64 = pinned.head_dim;
    const heads: u64 = pinned.query_heads;
    const per_row_out = heads * dim;
    const per_row_in = 2 * heads * dim;
    if (out_elements != rows * per_row_out or in_elements != rows * per_row_in) return null;
    return .{
        .count = std.math.cast(u32, rows * per_row_out) orelse return null,
        .dim = @intCast(dim),
        .heads_rows = std.math.cast(u32, rows * heads) orelse return null,
        .in_head_stride = @intCast(2 * dim),
    };
}

const testing = std.testing;

test "extractQPlan derives the interleaved geometry and refuses torn views" {
    const plan = extractQPlan(3, 3 * 8 * 256, 3 * 4096).?;
    try testing.expectEqual(@as(u32, 3 * 2048), plan.count);
    try testing.expectEqual(@as(u32, 256), plan.dim);
    try testing.expectEqual(@as(u32, 24), plan.heads_rows);
    try testing.expectEqual(@as(u32, 512), plan.in_head_stride);
    try testing.expectEqual(@as(?ExtractQPlan, null), extractQPlan(0, 0, 0));
    try testing.expectEqual(@as(?ExtractQPlan, null), extractQPlan(33, 33 * 2048, 33 * 4096)); // past the prefill cap
    try testing.expectEqual(@as(?ExtractQPlan, null), extractQPlan(2, 2 * 2048, 2 * 4095));
    try testing.expectEqual(@as(?ExtractQPlan, null), extractQPlan(2, 2 * 2047, 2 * 4096));
}
