//! Parse's causal mBART target: biased pre-norm GQA, exact GELU and tied BF16 logits, without text RoPE.
const std = @import("std");
const hip = @import("hip");
const c = @import("config.zig");
const Compute = @import("compute.zig").Compute;
const Tensor = hip.ops.Tensor;

pub const Layer = struct { k: hip.DeviceBuffer, v: hip.DeviceBuffer };
pub const State = struct {
    layers: [c.layers]Layer,
    hidden: hip.DeviceBuffer,
    total: usize,
    len: usize = 0,

    pub fn init(r: *const hip.Runtime, total: usize) !State {
        if (total == 0 or total > c.context) return error.BadContext;
        var out: State = .{ .layers = undefined, .hidden = undefined, .total = total };
        var loaded: usize = 0;
        errdefer for (out.layers[0..loaded]) |*l| {
            l.k.free();
            l.v.free();
        };
        const bytes = c.kv_heads * c.head_dim * total * 2;
        for (&out.layers) |*l| {
            l.k = try hip.DeviceBuffer.alloc(r, bytes);
            errdefer l.k.free();
            l.v = try hip.DeviceBuffer.alloc(r, bytes);
            loaded += 1;
        }
        out.hidden = try hip.DeviceBuffer.alloc(r, total * c.hidden * 2);
        return out;
    }

    pub fn deinit(s: *State) void {
        for (&s.layers) |*l| {
            l.k.free();
            l.v.free();
        }
        s.hidden.free();
        s.* = undefined;
    }

    /// Truncation leaves accepted KV and final-hidden rows intact; later windows overwrite discarded suffix rows.
    pub fn keep(s: *State, length: usize) !void {
        if (length > s.len) return error.BadKeep;
        s.len = length;
    }

    pub fn hiddenAt(s: *const State, position: usize) !Tensor {
        if (position >= s.len) return error.BadPosition;
        return .{ .ptr = s.hidden.base() + position * c.hidden * 2, .kind = .bf16 };
    }
};

pub fn embed(compute: Compute, ids: u64, rows: usize) !Tensor {
    const out = try compute.alloc(rows * c.hidden, .bf16);
    try compute.kernels.embed(compute.stream, try compute.weights.get("decoder.embed_tokens.weight"), ids, out, rows, c.hidden, 32);
    return out;
}

/// One stream's contiguous span within a packed forward; its KV remains separate from every other stream.
pub const Span = struct { state: *State, rows: usize };

/// Append embedded rows to one stream; the returned final norm lives in owned stream state.
pub fn forward(compute: Compute, state: *State, embeddings: Tensor, rows: usize) !Tensor {
    const at = state.len;
    _ = try forwardMany(compute, &.{.{ .state = state, .rows = rows }}, embeddings);
    return state.hiddenAt(at);
}

/// One shared dense forward over all spans, with causal attention confined to each span's own KV.
pub fn forwardMany(compute: Compute, spans: []const Span, embeddings: Tensor) !Tensor {
    const rows = try rowsOf(spans);
    const x = try compute.norm(embeddings, "decoder.layernorm_embedding", rows, c.hidden, c.norm_eps);
    const mark = compute.ops.arena.mark();
    try compute.record("decoder.embedding_norm", x, rows, c.hidden);
    var buf: [160]u8 = undefined;
    for (0..c.layers) |i| {
        try compute.poll();
        try layer(compute, spans, i, x, rows);
        try compute.record(try std.fmt.bufPrint(&buf, "decoder.layer.{d}", .{i}), x, rows, c.hidden);
        compute.ops.arena.release(mark);
    }
    const out = try compute.norm(x, "decoder.layer_norm", rows, c.hidden, c.norm_eps);
    try compute.record("decoder.final_norm", out, rows, c.hidden);
    var offset: usize = 0;
    for (spans) |span| {
        try hip.raw.copy(span.state.hidden, span.state.len * c.hidden * 2, out.ptr + offset * c.hidden * 2, span.rows * c.hidden * 2, compute.stream.handle);
        offset += span.rows;
    }
    // Publish lengths only after all commands have been queued successfully.
    for (spans) |span| span.state.len += span.rows;
    return out;
}

/// Check every context and duplicate state before a batch can enqueue any GPU work.
pub fn rowsOf(spans: []const Span) !usize {
    var rows: usize = 0;
    for (spans, 0..) |span, i| {
        const state = span.state;
        if (span.rows == 0 or state.len > state.total or span.rows > state.total - state.len) return error.ContextExceeded;
        for (spans[0..i]) |prior| if (prior.state == state) return error.DuplicateState;
        rows = try std.math.add(usize, rows, span.rows);
    }
    if (rows == 0) return error.ContextExceeded;
    return rows;
}

pub fn logits(compute: Compute, hidden: Tensor, rows: usize) !Tensor {
    return compute.linear(hidden, "decoder.embed_tokens", rows, c.vocab, c.hidden, false);
}

fn layer(compute: Compute, spans: []const Span, index: usize, x: Tensor, rows: usize) !void {
    var name: [192]u8 = undefined;
    var root_buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "decoder.layers.{d}", .{index});
    const norm = try compute.norm(x, try sub(&name, root, "self_attn_layer_norm"), rows, c.hidden, c.norm_eps);
    const q = try compute.linear(norm, try sub(&name, root, "self_attn.q_proj"), rows, c.hidden, c.hidden, true);
    const k = try compute.linear(norm, try sub(&name, root, "self_attn.k_proj"), rows, c.kv_heads * c.head_dim, c.hidden, true);
    const v = try compute.linear(norm, try sub(&name, root, "self_attn.v_proj"), rows, c.kv_heads * c.head_dim, c.hidden, true);
    const row_att = try compute.alloc(rows * c.hidden, .bf16);
    const mark = compute.ops.arena.mark();
    var offset: usize = 0;
    for (spans) |span| {
        const state = span.state;
        const qs: Tensor = .{ .ptr = q.ptr + offset * c.hidden * 2, .kind = .bf16 };
        const ks: Tensor = .{ .ptr = k.ptr + offset * c.kv_heads * c.head_dim * 2, .kind = .bf16 };
        const vs: Tensor = .{ .ptr = v.ptr + offset * c.kv_heads * c.head_dim * 2, .kind = .bf16 };
        const wide_q = try compute.heads(qs, span.rows, c.heads, c.head_dim, true);
        const cache = state.layers[index];
        try compute.ops.kvWrite(ks, cache.k.base(), span.rows, c.kv_heads, c.head_dim, state.total, state.len);
        try compute.ops.kvWrite(vs, cache.v.base(), span.rows, c.kv_heads, c.head_dim, state.total, state.len);
        const attended = try compute.alloc(span.rows * c.hidden, .f32);
        try compute.ops.causalPrefill(wide_q.ptr, .{ .k = cache.k.base(), .v = cache.v.base(), .kind = .bf16, .kv_heads = c.kv_heads, .total = state.total, .d = c.head_dim }, attended.ptr, span.rows, state.len + span.rows, c.heads, 1 / @sqrt(@as(f32, c.head_dim)), state.len);
        const dst: Tensor = .{ .ptr = row_att.ptr + offset * c.hidden * 2, .kind = .bf16 };
        try compute.kernels.layout(compute.stream, attended, dst, span.rows, c.heads, c.head_dim, .heads_to_rows);
        offset += span.rows;
        compute.ops.arena.release(mark);
    }
    const projected = try compute.linear(row_att, try sub(&name, root, "self_attn.out_proj"), rows, c.hidden, c.hidden, true);
    try compute.residual(x, projected, rows * c.hidden);
    const ffn_norm = try compute.norm(x, try sub(&name, root, "final_layer_norm"), rows, c.hidden, c.norm_eps);
    const up = try compute.linear(ffn_norm, try sub(&name, root, "fc1"), rows, c.ffn, c.hidden, true);
    try compute.gelu(up, rows * c.ffn);
    const down = try compute.linear(up, try sub(&name, root, "fc2"), rows, c.hidden, c.ffn, true);
    try compute.residual(x, down, rows * c.hidden);
}

fn sub(buf: []u8, root: []const u8, end: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ root, end });
}
