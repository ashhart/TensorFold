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
/// `block`: rows a conv window spans; a batched pass stacks several streams' blocks.
fn convolve(ops: op.Ops, frame: u64, x: op.View, dynamic: op.View, weights: w.Conv, branch: u32, residual: ?op.View, c: Config, block: u32) !op.View {
    try dynamic.check(x.rows, c.dynamicWidth(), .bf16);
    if (residual) |r| try r.check(x.rows, x.width, .bf16);
    const y = try ops.vtable.conv(ops.ptr, frame, x, .{ .dynamic = dynamic, .base = weights.base, .branch = branch, .group = c.group, .block = block, .residual = residual });
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

/// Where appended keys and values go: a transaction's context, or a batched pass's stream.
const Sink = union(enum) { transaction: u64, stream: u32 };
fn encodeAppend(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, taps: op.View, positions: []const u64, frame: u64, transaction: u64) !void {
    return encodeAppendTo(ops, g, preparation, taps, positions, frame, .{ .transaction = transaction });
}
fn encodeAppendTo(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, taps: op.View, positions: []const u64, frame: u64, sink: Sink) !void {
    const c = g.config;
    const fusion = try project(ops, frame, taps, g.fusion, preparation.mode);
    const hidden = try norm(ops, frame, fusion, g.hidden_norm, c.eps, 1, c.hidden);
    for (g.layers, 0..) |layer, i| {
        const raw_keys = try project(ops, frame, hidden, layer.attention.k, preparation.mode);
        const keys = try rotate(ops, frame, try norm(ops, frame, raw_keys, layer.attention.k_norm, c.eps, c.kv_heads, c.head_dim), positions, c.kv_heads, c);
        const values = try project(ops, frame, hidden, layer.attention.v, preparation.mode);
        switch (sink) {
            .transaction => |t| try ops.vtable.context_append(ops.ptr, t, frame, @intCast(i), keys, values, positions),
            .stream => |k| try ops.vtable.stream_append(ops.ptr, frame, k, @intCast(i), keys, values, positions),
        }
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
    return encodeStreams(ops, g, preparation, &.{.{ .context = context, .anchor = anchor }}, frame, block, given);
}

/// A stream of a batched block pass: its context and anchor (its caches are the backend's batch, in stream order).
pub const Stream = struct { context: op.Context, anchor: u32 };

/// Every stream's block stacked in one pass, weights read once; each stream attends to its own context.
fn encodeStreams(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, streams: []const Stream, frame: u64, block: u32, given: []const u32) !Output {
    const c = g.config;
    const n: u32 = @intCast(streams.len);
    const rows = n * block;
    if (n == 0 or rows > 128 or (n > 1 and given.len > 0)) return error.BadRuntimeBlock;
    var ids: [128]u32 = @splat(c.mask);
    var positions: [128]u64 = undefined;
    var contexts: [8]op.Context = undefined;
    if (n > contexts.len) return error.BadRuntimeBlock;
    for (streams, 0..) |s, k| {
        contexts[k] = s.context;
        ids[k * block] = s.anchor;
        for (positions[k * block ..][0..block], 0..) |*p, i| p.* = try std.math.add(u64, s.context.end, i);
    }
    for (given, 1..) |t, i| ids[i] = t;
    var hidden = try ops.vtable.embed(ops.ptr, frame, g.embed, ids[0..rows]);
    try hidden.check(rows, c.hidden, .bf16);
    for (g.layers, 0..) |layer, i| {
        var residual = hidden;
        const xn = try norm(ops, frame, hidden, layer.input_norm, c.eps, 1, c.hidden);
        const dynamic = try project(ops, frame, xn, layer.attention_conv.projection, preparation.mode);
        const input = try convolve(ops, frame, xn, dynamic, layer.attention_conv, 0, null, c, block);
        const query = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.q, preparation.mode), layer.attention.q_norm, c.eps, c.heads, c.head_dim), positions[0..rows], c.heads, c);
        const keys = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.k, preparation.mode), layer.attention.k_norm, c.eps, c.kv_heads, c.head_dim), positions[0..rows], c.kv_heads, c);
        const values = try project(ops, frame, input, layer.attention.v, preparation.mode);
        const attended = try ops.vtable.attention(ops.ptr, frame, .{ .context = streams[0].context, .config = c, .layer = @intCast(i), .query = query, .keys = keys, .values = values, .positions = positions[0..rows], .streams = if (n > 1) contexts[0..n] else &.{}, .block = block });
        try attended.check(rows, c.qWidth(), .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, attended, layer.attention.o, preparation.mode), dynamic, layer.attention_conv, 1, residual, c, block);
        residual = hidden;
        const post = try norm(ops, frame, hidden, layer.post_norm, c.eps, 1, c.hidden);
        const dyn_mlp = try project(ops, frame, post, layer.mlp_conv.projection, preparation.mode);
        const mlp_input = try convolve(ops, frame, post, dyn_mlp, layer.mlp_conv, 0, null, c, block);
        const gate = try project(ops, frame, mlp_input, layer.mlp.gate, preparation.mode);
        const up = try project(ops, frame, mlp_input, layer.mlp.up, preparation.mode);
        const activated = try ops.vtable.swiglu(ops.ptr, frame, gate, up);
        try activated.check(rows, c.intermediate, .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, activated, layer.mlp.down, preparation.mode), dyn_mlp, layer.mlp_conv, 1, residual, c, block);
    }
    // one stream heads only its mask rows; a batch heads every row and each stream's lattice skips its anchor
    const masks = if (n == 1) try ops.vtable.mask_rows(ops.ptr, frame, hidden, 1, block - 1) else hidden;
    const headed = if (n == 1) block - 1 else rows;
    try masks.check(headed, c.hidden, .bf16);
    const final = try norm(ops, frame, masks, g.norm, c.eps, 1, c.hidden);
    const head = try ops.vtable.head(ops.ptr, frame, final);
    if (head.logits.handle == 0 or head.logits.rows != headed or head.logits.dtype != .bf16 or head.logits.width == 0 or head.logits.width > c.vocab or head.logits.stride < head.logits.width) return error.DraftOperatorShape;
    if (head.token_ids) |mapping| try mapping.check(1, head.logits.width, .u32) else if (head.logits.width != c.vocab) return error.DraftCandidateId;
    const projected = try project(ops, frame, final, g.hidden_projection, preparation.mode);
    return .{ .head = head, .projected = projected };
}
/// A stream's new taps for a batched pass: absorbed into its caches before its block reads them.
pub const Absorb = struct { taps: op.View, positions: []const u64 };

/// In one frame, absorb taps per stream, draft 2-8 blocks and build each lattice without its anchor row.
pub fn absorbBlocks(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, frame: u64, absorbs: []const ?Absorb, streams: []const Stream, block: u32, out: []op.Lattice) !void {
    try preparation.check();
    try g.config.check();
    if (block < 2 or block > 16 or streams.len < 2 or streams.len * block > 128 or out.len != streams.len or absorbs.len != streams.len) return error.BadRuntimeBlock;
    for (streams) |s| if (s.anchor >= g.config.vocab or s.context.cache == 0 or s.context.begin > s.context.end or s.context.end - s.context.begin > g.config.window - 1) return error.BadDraftContext;
    for (absorbs, 0..) |a, k| if (a) |x| {
        try x.taps.check(@intCast(x.positions.len), g.config.tapWidth(), .bf16);
        try encodeAppendTo(ops, g, preparation, x.taps, x.positions, frame, .{ .stream = @intCast(k) });
    };
    const output = try encodeStreams(ops, g, preparation, streams, frame, block, &.{});
    try ops.vtable.lattices(ops.ptr, frame, output.head, output.projected, g.config, block, out);
    for (out) |l| try l.check(g.config);
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
