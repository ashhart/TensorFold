//! The five-layer draft dataflow is encoded through typed native callbacks, never through a copied model binding.
const std = @import("std");
const w = @import("weights.zig");
const op = @import("operators.zig");
const Config = @import("config.zig").Config;

fn project(ops: op.Ops, frame: u64, x: op.View, weight: w.Tensor, mode: op.Mode) !op.View {
    if (weight.rank != 2 or weight.dtype != .bf16 or x.dtype != .bf16 or x.width != weight.shape[1]) return error.DraftOperatorShape;
    const y = try ops.vtable.project(ops.ptr, frame, x, weight, mode);
    try y.check(x.rows, @intCast(weight.shape[0]), .bf16);
    return y;
}
fn norm(ops: op.Ops, frame: u64, x: op.View, weight: w.Tensor, eps: f32, heads: u32, dim: u32) !op.View {
    if (weight.rank != 1 or weight.shape[0] != dim or x.width != heads * dim) return error.DraftOperatorShape;
    const y = try ops.vtable.norm(ops.ptr, frame, x, .{ .weight = weight, .eps = eps, .heads = heads, .head_dim = dim });
    try y.check(x.rows, x.width, .bf16);
    return y;
}
fn rotate(ops: op.Ops, frame: u64, x: op.View, positions: []const u64, heads: u32, c: Config) !op.View {
    const y = try ops.vtable.rope(ops.ptr, frame, x, .{ .positions = positions, .heads = heads, .dim = c.head_dim, .theta = c.theta });
    try y.check(x.rows, x.width, .bf16);
    return y;
}
fn convolve(ops: op.Ops, frame: u64, x: op.View, dynamic: op.View, weights: w.Conv, branch: u32, residual: ?op.View, c: Config) !op.View {
    try dynamic.check(x.rows, c.dynamicWidth(), .bf16);
    if (residual) |r| try r.check(x.rows, x.width, .bf16);
    const y = try ops.vtable.conv(ops.ptr, frame, x, .{ .dynamic = dynamic, .base = weights.base, .branch = branch, .group = c.group, .block = x.rows, .residual = residual });
    try y.check(x.rows, c.hidden, .bf16);
    return y;
}

pub fn append(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, taps: op.View, positions: []const u64, next: op.Context) !void {
    try preparation.check();
    const c = g.config;
    try c.check();
    if (context.cache == 0 or context.begin > context.end or next.cache != context.cache or positions.len == 0 or next.end != try std.math.add(u64, context.end, positions.len) or next.begin != @max(context.begin, next.end -| (@as(u64, c.window) - 1))) return error.BadDraftContext;
    for (positions, 0..) |p, i| if (p != context.end + i or p >= next.end) return error.BadCommittedTaps;
    try taps.check(@intCast(positions.len), c.tapWidth(), .bf16);
    const frame = try ops.vtable.begin(ops.ptr);
    defer ops.vtable.release(ops.ptr, frame);
    const transaction = try ops.vtable.context_begin(ops.ptr, context);
    errdefer ops.vtable.context_abort(ops.ptr, transaction);
    try encodeAppend(ops, g, preparation, taps, positions, frame, transaction);
    try ops.vtable.context_commit(ops.ptr, transaction, frame, next);
}

fn checkAppend(g: *const w.Graph, preparation: op.Preparation, context: op.Context, taps: op.View, positions: []const u64, next: op.Context) !void {
    try preparation.check();
    const c = g.config;
    try c.check();
    if (context.cache == 0 or context.begin > context.end or next.cache != context.cache or positions.len == 0 or next.end != try std.math.add(u64, context.end, positions.len) or next.begin != @max(context.begin, next.end -| (@as(u64, c.window) - 1))) return error.BadDraftContext;
    for (positions, 0..) |p, i| if (p != context.end + i or p >= next.end) return error.BadCommittedTaps;
    try taps.check(@intCast(positions.len), c.tapWidth(), .bf16);
}

fn encodeAppend(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, taps: op.View, positions: []const u64, frame: u64, transaction: u64) !void {
    const c = g.config;
    const fusion = try project(ops, frame, taps, g.fusion, preparation.mode);
    const hidden = try norm(ops, frame, fusion, g.hidden_norm, c.eps, 1, c.hidden);
    for (g.layers, 0..) |layer, i| {
        const raw_keys = try project(ops, frame, hidden, layer.attention.k, preparation.mode);
        const keys = try rotate(ops, frame, try norm(ops, frame, raw_keys, layer.attention.k_norm, c.eps, c.kv_heads, c.head_dim), positions, c.kv_heads, c);
        const values = try project(ops, frame, hidden, layer.attention.v, preparation.mode);
        try ops.vtable.context_append(ops.ptr, transaction, frame, @intCast(i), keys, values, positions);
    }
}

/// append then forwardBlock in one frame: one commit and one wait for the committed taps and the block that reads them.
pub fn appendBlock(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, taps: op.View, positions: []const u64, next: op.Context, anchor: u32, block: u32) !op.Lattice {
    if (block < 2 or block > 16) return error.BadRuntimeBlock;
    try checkAppend(g, preparation, context, taps, positions, next);
    if (anchor >= g.config.vocab or next.end - next.begin > g.config.window - 1) return error.BadDraftContext;
    const frame = try ops.vtable.begin(ops.ptr);
    defer ops.vtable.release(ops.ptr, frame);
    const transaction = try ops.vtable.context_begin(ops.ptr, context);
    errdefer ops.vtable.context_abort(ops.ptr, transaction);
    try encodeAppend(ops, g, preparation, taps, positions, frame, transaction);
    try ops.vtable.context_stage(ops.ptr, transaction, frame, next);
    const output = try encode(ops, g, preparation, next, anchor, frame, block, &.{});
    var lattice = try ops.vtable.lattice(ops.ptr, frame, output.head, output.projected, g.config);
    errdefer lattice.deinit();
    try lattice.check(g.config);
    ops.vtable.context_abort(ops.ptr, transaction); // completed: clears the stage mark
    return lattice;
}

pub const Output = struct { head: op.Head, projected: op.View };
pub const Consumer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u64, Output, Config) anyerror!void };
fn start(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32) !u64 {
    try preparation.check();
    try g.config.check();
    if (anchor >= g.config.vocab or context.cache == 0 or context.begin > context.end or context.end - context.begin > g.config.window - 1) return error.BadDraftContext;
    return ops.vtable.begin(ops.ptr);
}
fn encode(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32, frame: u64, block: u32, given: []const u32) !Output {
    const c = g.config;
    var ids: [16]u32 = @splat(c.mask);
    ids[0] = anchor;
    for (given, 1..) |t, i| ids[i] = t;
    var positions: [16]u64 = undefined;
    for (positions[0..block], 0..) |*p, i| p.* = try std.math.add(u64, context.end, i);
    var hidden = try ops.vtable.embed(ops.ptr, frame, g.embed, ids[0..block]);
    try hidden.check(block, c.hidden, .bf16);
    for (g.layers, 0..) |layer, i| {
        var residual = hidden;
        const xn = try norm(ops, frame, hidden, layer.input_norm, c.eps, 1, c.hidden);
        const dynamic = try project(ops, frame, xn, layer.attention_conv.projection, preparation.mode);
        const input = try convolve(ops, frame, xn, dynamic, layer.attention_conv, 0, null, c);
        const query = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.q, preparation.mode), layer.attention.q_norm, c.eps, c.heads, c.head_dim), positions[0..block], c.heads, c);
        const keys = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.k, preparation.mode), layer.attention.k_norm, c.eps, c.kv_heads, c.head_dim), positions[0..block], c.kv_heads, c);
        const values = try project(ops, frame, input, layer.attention.v, preparation.mode);
        const attended = try ops.vtable.attention(ops.ptr, frame, .{ .context = context, .config = c, .layer = @intCast(i), .query = query, .keys = keys, .values = values, .positions = positions[0..block] });
        try attended.check(block, c.qWidth(), .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, attended, layer.attention.o, preparation.mode), dynamic, layer.attention_conv, 1, residual, c);
        residual = hidden;
        const post = try norm(ops, frame, hidden, layer.post_norm, c.eps, 1, c.hidden);
        const dyn_mlp = try project(ops, frame, post, layer.mlp_conv.projection, preparation.mode);
        const mlp_input = try convolve(ops, frame, post, dyn_mlp, layer.mlp_conv, 0, null, c);
        const gate = try project(ops, frame, mlp_input, layer.mlp.gate, preparation.mode);
        const up = try project(ops, frame, mlp_input, layer.mlp.up, preparation.mode);
        const activated = try ops.vtable.swiglu(ops.ptr, frame, gate, up);
        try activated.check(block, c.intermediate, .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, activated, layer.mlp.down, preparation.mode), dyn_mlp, layer.mlp_conv, 1, residual, c);
    }
    const masks = try ops.vtable.mask_rows(ops.ptr, frame, hidden, 1, block - 1);
    try masks.check(block - 1, c.hidden, .bf16);
    const final = try norm(ops, frame, masks, g.norm, c.eps, 1, c.hidden);
    const head = try ops.vtable.head(ops.ptr, frame, g.head, final);
    if (head.logits.handle == 0 or head.logits.rows != block - 1 or head.logits.dtype != .bf16 or head.logits.width == 0 or head.logits.width > c.vocab or head.logits.stride < head.logits.width) return error.DraftOperatorShape;
    if (head.token_ids) |mapping| try mapping.check(1, head.logits.width, .u32) else if (head.logits.width != c.vocab) return error.DraftCandidateId;
    const projected = try project(ops, frame, final, g.hidden_projection, preparation.mode);
    return .{ .head = head, .projected = projected };
}
pub fn forwardToBlock(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32, block: u32, consumer: Consumer) !void {
    if (block < 2 or block > 16) return error.BadRuntimeBlock;
    const frame = try start(ops, g, preparation, context, anchor);
    defer ops.vtable.release(ops.ptr, frame);
    const output = try encode(ops, g, preparation, context, anchor, frame, block, &.{});
    try consumer.call(consumer.ptr, frame, output, g.config);
}
pub fn forwardTo(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32, consumer: Consumer) !void {
    return forwardToBlock(ops, g, preparation, context, anchor, g.config.block, consumer);
}
pub fn forwardBlock(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32, block: u32) !op.Lattice {
    if (block < 2 or block > 16) return error.BadRuntimeBlock;
    const frame = try start(ops, g, preparation, context, anchor);
    defer ops.vtable.release(ops.ptr, frame);
    const output = try encode(ops, g, preparation, context, anchor, frame, block, &.{});
    var lattice = try ops.vtable.lattice(ops.ptr, frame, output.head, output.projected, g.config);
    errdefer lattice.deinit();
    try lattice.check(g.config);
    return lattice;
}
/// A block whose first positions after the anchor hold given tokens, not masks (later positions see them).
pub fn forwardGiven(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32, given: []const u32, block: u32) !op.Lattice {
    if (block < 2 or block > 16 or given.len + 1 >= block) return error.BadRuntimeBlock;
    const frame = try start(ops, g, preparation, context, anchor);
    defer ops.vtable.release(ops.ptr, frame);
    const output = try encode(ops, g, preparation, context, anchor, frame, block, given);
    var lattice = try ops.vtable.lattice(ops.ptr, frame, output.head, output.projected, g.config);
    errdefer lattice.deinit();
    try lattice.check(g.config);
    return lattice;
}
pub fn forward(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, context: op.Context, anchor: u32) !op.Lattice {
    return forwardBlock(ops, g, preparation, context, anchor, g.config.block);
}
