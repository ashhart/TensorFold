// Single-sequence Qwen3.5-2B Vulkan matches Ash8c Metal (24 layers: 18 GDN, 6 attention); GPU calls via vk.orExit.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");

const Arena = vk.arena.Arena;
const View = vk.tensor.View;

// Whole-prompt rows are split into dispatch chunks capped here; persistent windows use this cap.
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

// `Core` borrows commands, staging, operators, scratch and fence timeout; the owner owns the arena and every view.
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

    // Fused norm: bf16 residual add (`h = bf16(H + R)`) out of place into `sum`, then raw-gain RMSNorm into `out`.
    pub fn addNorm(c: *const Core, out: View, sum: View, h: View, r: View, gain: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.add(c.cmds, sum, h, r, c.timeout_ns));
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, out, sum, gain, rows, pinned.hidden, pinned.eps, c.timeout_ns));
    }

    /// The layer-0 norm: no residual, the gain over the raw embedding.
    pub fn plainNorm(c: *const Core, out: View, h: View, gain: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, out, h, gain, rows, pinned.hidden, pinned.eps, c.timeout_ns));
    }

    // `extractQueries` reads `qraw[d + k * 512]`, `k = row * 8 + head`, matching `qwen35_head_norm` source indexing.
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

    // Attention uses Q/K/V, gamma (256), RoPE, KV at `base`, causal GQA SDPA and gate/O; `r_out` cannot alias reads.
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

    // GDN runs qkv/z/a/b, raw-BF16 conv, gates/recurrence; one-row prefill preserves rounding; slot flips after writes.
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
        const conv_out = try c.scratch.alloc(.bf16, rows * pinned.conv_dim);
        const conv_in = try c.scratch.alloc(.bf16, (rows + state.conv_rows) * pinned.conv_dim);
        try vk.orExit(c.acc.gdn.convPrefill(c.cmds, soa, g, beta, conv_out, conv_in, s.convInput(slot), qkv, layer.conv, a, b, layer.a_log, layer.dt_bias, s.convOutput(slot), rows, c.timeout_ns));
        const y = try c.scratch.alloc(.bf16, rows * dim);
        try vk.orExit(c.acc.gdn.prefillRecur(c.cmds, soa, g, beta, s.gdnInput(slot), y, s.gdnOutput(slot), null, rows, @min(rows, max_gdn_chunk), c.timeout_ns));
        const mix = try c.scratch.alloc(.bf16, rows * dim);
        try vk.orExit(c.acc.gdn.gatedNorm(c.cmds, mix, y, z, layer.gnorm, rows, pinned.eps, c.timeout_ns));
        try c.project(r_out, mix, layer.out);
        s.flip(slot);
        c.scratch.reset(mark) catch {};
    }

    // The dense MLP: Q4 gate and up projections, the ASH single-round SwiGLU, and the Q4 down projection into `r_out`.
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

    // The last row gets residual add, raw-gain norm, and tied embedding or separate `lm_head` for greedy draw.
    pub fn finish(c: *const Core, sum: View, h: View, r: View, x: View, final_norm: View, head: weights.Quant, logits: View, rows: usize) !void {
        try vk.orExit(c.acc.norm.add(c.cmds, sum, h, r, c.timeout_ns));
        const last = (rows - 1) * pinned.hidden;
        const x_last = try x.slice(last, pinned.hidden);
        try vk.orExit(c.acc.norm.rmsNorm(c.cmds, x_last, try sum.slice(last, pinned.hidden), final_norm, 1, pinned.hidden, pinned.eps, c.timeout_ns));
        const mark = c.scratch.mark();
        defer c.scratch.reset(mark) catch {};
        try c.project(logits, x_last, head);
    }
};

// `extractQPlan` is null on torn views; pure, host-tested; `heads_rows=rows*query_heads`, `in_head_stride=2*head_dim`.
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
