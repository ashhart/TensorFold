//! Mixed and contiguous attention entry points with optional output group sums.
const mtl = @import("metal");
const attn = @import("attention.zig");
const views = @import("attention_views.zig");
const Ref = @import("projection.zig").Ref;
const Sums = @import("quant_gpu.zig").Sums;

pub fn encode(a: *const attn.Attention, e: mtl.ComputeEncoder, scratch: *const attn.Scratch, binding: *const attn.Binding, plan: attn.Plan, q: Ref, k: Ref, mixed: Ref, out: Ref, caches: []const attn.Cache, gate: ?Ref, prompt: bool, split: bool) !void {
    return encodeSums(a, e, scratch, binding, plan, q, k, mixed, out, caches, gate, prompt, split, null);
}

pub fn encodeContiguous(a: *const attn.Attention, e: mtl.ComputeEncoder, scratch: *const attn.Scratch, binding: *const attn.Binding, plan: attn.Plan, q: Ref, k: Ref, v: Ref, out: Ref, caches: []const attn.Cache, gate: ?Ref, prompt: bool, split: bool) !void {
    return encodeContiguousSums(a, e, scratch, binding, plan, q, k, v, out, caches, gate, prompt, split, null);
}

pub fn encodeSums(a: *const attn.Attention, e: mtl.ComputeEncoder, scratch: *const attn.Scratch, binding: *const attn.Binding, plan: attn.Plan, q: Ref, k: Ref, mixed: Ref, out: Ref, caches: []const attn.Cache, gate: ?Ref, prompt: bool, split: bool, out_sums: ?Sums) !void {
    try plan.geometry.check();
    if (plan.rows == 0) return error.TargetAttentionShape;
    const kv = plan.geometry.kv_heads * plan.geometry.dim;
    try views.span(mixed, @as(usize, plan.rows) * kv * 4);
    try views.span(out, @as(usize, plan.rows) * plan.geometry.heads * plan.geometry.dim * 2);
    try views.disjoint(mixed, @as(usize, plan.rows) * kv * 4, out, @as(usize, plan.rows) * plan.geometry.heads * plan.geometry.dim * 2);
    for (caches) |cache| try views.apart(cache, plan.geometry, mixed, @as(usize, plan.rows) * kv * 4);
    const v = Ref{ .buffer = mixed.buffer, .offset = mixed.offset + @as(usize, kv) * 2 };
    return a.encodeRows(e, scratch, binding, plan, q, k, v, 2 * kv, out, caches, gate, prompt, split, out_sums);
}

pub fn encodeContiguousSums(a: *const attn.Attention, e: mtl.ComputeEncoder, scratch: *const attn.Scratch, binding: *const attn.Binding, plan: attn.Plan, q: Ref, k: Ref, v: Ref, out: Ref, caches: []const attn.Cache, gate: ?Ref, prompt: bool, split: bool, out_sums: ?Sums) !void {
    return a.encodeRows(e, scratch, binding, plan, q, k, v, plan.geometry.kv_heads * plan.geometry.dim, out, caches, gate, prompt, split, out_sums);
}

pub fn check(a: *const attn.Attention, plan: attn.Plan, q: Ref, k: Ref, v: Ref, out: Ref, q_bytes: usize, kv_bytes: usize, v_bytes: usize, caches: []const attn.Cache, raw_gate: ?Ref, scratch: *const attn.Scratch, binding: *const attn.Binding, prompt: bool, split: bool, out_sums: ?Sums) !void {
    if (out_sums) |xs| {
        if (!a.units or prompt or split or xs.stride < plan.rows) return error.AttentionSums;
        const bytes = ((@as(usize, plan.geometry.heads) * plan.geometry.dim / 64 - 1) * xs.stride + plan.rows) * 4;
        try views.spanAligned(xs.ref, bytes, 4);
        try views.disjoint(xs.ref, bytes, out, q_bytes);
        try views.disjoint(xs.ref, bytes, q, q_bytes);
        try views.disjoint(xs.ref, bytes, k, kv_bytes);
        try views.disjoint(xs.ref, bytes, v, v_bytes);
        if (raw_gate) |gate| try views.disjoint(xs.ref, bytes, gate, 2 * q_bytes);
        for (caches) |cache| try views.apart(cache, plan.geometry, xs.ref, bytes);
        for (scratch.buffers) |buffer| if (buffer.id == xs.ref.buffer.id) return error.AttentionAlias;
        for (binding.buffers) |buffer| if (buffer.id == xs.ref.buffer.id) return error.AttentionAlias;
    }
}
