//! GLM-local DFlash2 encoding: 5 bidirectional layers over full blocks, active draft rows only at norm+head+topk.
const std = @import("std");
const op = @import("../qwen27/dflash/operators.zig");
const w = @import("../qwen27/dflash/weights.zig");
const Config = @import("../qwen27/dflash/config.zig").Config;
const execution = @import("../qwen27/dflash/execution.zig");
const runtime_lattice = @import("../qwen27/dflash/runtime_lattice.zig");
const backend_mod = @import("metal_dflash_backend.zig");

fn project(ops: op.Ops, frame: u64, x: op.View, weight: anytype, mode: op.Mode) !op.View {
    return ops.vtable.project(ops.ptr, frame, x, weight, mode);
}

fn norm(ops: op.Ops, frame: u64, x: op.View, weight: anytype, eps: f32, heads: u32, dim: u32) !op.View {
    const y = try ops.vtable.norm(ops.ptr, frame, x, .{ .weight = weight, .eps = eps, .heads = heads, .head_dim = dim });
    try y.check(x.rows, x.width, .bf16);
    return y;
}

fn rotate(ops: op.Ops, frame: u64, x: op.View, positions: []const u64, heads: u32, c: Config) !op.View {
    const y = try ops.vtable.rope(ops.ptr, frame, x, .{ .positions = positions, .heads = heads, .dim = c.head_dim, .theta = c.theta });
    try y.check(x.rows, x.width, .bf16);
    return y;
}

fn convolve(ops: op.Ops, frame: u64, x: op.View, dynamic: op.View, weights: w.Conv, branch: u32, residual: ?op.View, c: Config, block: u32) !op.View {
    try dynamic.check(x.rows, c.dynamicWidth(), .bf16);
    if (residual) |r| try r.check(x.rows, x.width, .bf16);
    const y = try ops.vtable.conv(ops.ptr, frame, x, .{ .dynamic = dynamic, .base = weights.base, .branch = branch, .group = c.group, .block = block, .residual = residual });
    try y.check(x.rows, c.hidden, .bf16);
    return y;
}

fn encodeLayers(ops: op.Ops, g: *const w.Graph, preparation: op.Preparation, streams: []const execution.Stream, frame: u64, block: u32) !op.View {
    const c = g.config;
    const n: u32 = @intCast(streams.len);
    const rows = n * block;
    if (n == 0 or rows > 128 or n > 8) return error.BadRuntimeBlock;
    var ids: [128]u32 = @splat(c.mask);
    var positions: [128]u64 = undefined;
    var contexts: [8]op.Context = undefined;
    for (streams, 0..) |s, k| {
        contexts[k] = s.context;
        ids[k * block] = s.anchor;
        for (positions[k * block ..][0..block], 0..) |*p, i| p.* = try std.math.add(u64, s.context.end, i);
    }
    var hidden = try ops.vtable.embed(ops.ptr, frame, g.embed, ids[0..rows]);
    try hidden.check(rows, c.hidden, .bf16);
    for (g.layers, 0..) |layer, i| {
        const residual = hidden;
        const xn = try norm(ops, frame, hidden, layer.input_norm, c.eps, 1, c.hidden);
        const dynamic = try project(ops, frame, xn, layer.attention_conv.projection, preparation.mode);
        const input = try convolve(ops, frame, xn, dynamic, layer.attention_conv, 0, null, c, block);
        const query = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.q, preparation.mode), layer.attention.q_norm, c.eps, c.heads, c.head_dim), positions[0..rows], c.heads, c);
        const keys = try rotate(ops, frame, try norm(ops, frame, try project(ops, frame, input, layer.attention.k, preparation.mode), layer.attention.k_norm, c.eps, c.kv_heads, c.head_dim), positions[0..rows], c.kv_heads, c);
        const values = try project(ops, frame, input, layer.attention.v, preparation.mode);
        const attended = try ops.vtable.attention(ops.ptr, frame, .{ .context = streams[0].context, .config = c, .layer = @intCast(i), .query = query, .keys = keys, .values = values, .positions = positions[0..rows], .streams = if (n > 1) contexts[0..n] else &.{}, .block = block });
        try attended.check(rows, c.qWidth(), .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, attended, layer.attention.o, preparation.mode), dynamic, layer.attention_conv, 1, residual, c, block);
        const res2 = hidden;
        const post = try norm(ops, frame, hidden, layer.post_norm, c.eps, 1, c.hidden);
        const dyn_mlp = try project(ops, frame, post, layer.mlp_conv.projection, preparation.mode);
        const mlp_input = try convolve(ops, frame, post, dyn_mlp, layer.mlp_conv, 0, null, c, block);
        const gate = try project(ops, frame, mlp_input, layer.mlp.gate, preparation.mode);
        const up = try project(ops, frame, mlp_input, layer.mlp.up, preparation.mode);
        const activated = try ops.vtable.swiglu(ops.ptr, frame, gate, up);
        try activated.check(rows, c.intermediate, .bf16);
        hidden = try convolve(ops, frame, try project(ops, frame, activated, layer.mlp.down, preparation.mode), dyn_mlp, layer.mlp_conv, 1, res2, c, block);
    }
    return hidden;
}

/// Encode 1 stream over `block` bidirectional rows and slice only `1..1+depth` for norm, head and top-k.
pub fn single(ops: op.Ops, b: *backend_mod.Backend, g: *const w.Graph, preparation: op.Preparation, stream: execution.Stream, frame: u64, block: u32, depth: u32) !op.Lattice {
    try preparation.check();
    const c = g.config;
    try c.check();
    if (block < 2 or block > 16 or depth == 0 or depth >= block) return error.BadRuntimeBlock;
    const hidden = try encodeLayers(ops, g, preparation, &.{stream}, frame, block);
    const prev_block = b.batch_block;
    b.batch_block = 0;
    defer b.batch_block = prev_block;
    const masks = try ops.vtable.mask_rows(ops.ptr, frame, hidden, 1, depth);
    try masks.check(depth, c.hidden, .bf16);
    const final = try norm(ops, frame, masks, g.norm, c.eps, 1, c.hidden);
    const head_out = try ops.vtable.head(ops.ptr, frame, final);
    if (head_out.logits.handle == 0 or head_out.logits.rows != depth or head_out.logits.dtype != .bf16 or head_out.logits.width == 0 or head_out.logits.width > c.vocab or head_out.logits.stride < head_out.logits.width) return error.DraftOperatorShape;
    const projected = try project(ops, frame, final, g.hidden_projection, preparation.mode);
    var l = try ops.vtable.lattice(ops.ptr, frame, head_out, projected, c);
    errdefer l.deinit();
    try l.check(c);
    return l;
}

/// Encode 2..8 streams over `block` bidirectional rows and gather active draft rows before norm, head and top-k.
pub fn batch(ops: op.Ops, b: *backend_mod.Backend, g: *const w.Graph, preparation: op.Preparation, streams: []const execution.Stream, frame: u64, block: u32, depths: []const u32, out: []op.Lattice) !void {
    try preparation.check();
    const c = g.config;
    try c.check();
    const n: u32 = @intCast(streams.len);
    if (block < 2 or block > 16 or n < 2 or n * block > 128 or out.len != n or depths.len != n) return error.BadRuntimeBlock;
    var active: u32 = 0;
    for (depths) |d| {
        if (d == 0 or d >= block) return error.BadRuntimeBlock;
        active += d;
    }
    const hidden = try encodeLayers(ops, g, preparation, streams, frame, block);
    const masks = try b.frame.alloc(active, c.hidden, .bf16);
    const src_ref = try b.frame.ref(hidden);
    const dst_ref = try b.frame.ref(masks);
    const e = try b.encoder(frame);
    var off_r: usize = 0;
    for (depths, 0..) |d, k| {
        try b.copy(e, .{ .buf = src_ref.buf, .off = src_ref.off + (@as(usize, k) * block + 1) * hidden.stride * 2 }, .{ .buf = dst_ref.buf, .off = dst_ref.off + off_r * masks.stride * 2 }, d, c.hidden);
        off_r += d;
    }
    const prev_block = b.batch_block;
    b.batch_block = 0;
    defer b.batch_block = prev_block;
    const final = try norm(ops, frame, masks, g.norm, c.eps, 1, c.hidden);
    const head_out = try ops.vtable.head(ops.ptr, frame, final);
    if (head_out.logits.handle == 0 or head_out.logits.rows != active or head_out.logits.dtype != .bf16 or head_out.logits.width == 0 or head_out.logits.width > c.vocab or head_out.logits.stride < head_out.logits.width) return error.DraftOperatorShape;
    const projected = try project(ops, frame, final, g.hidden_projection, preparation.mode);
    const topk_view = try backend_mod.encodeTopk(b, frame, head_out, c.topk);
    try backend_mod.finishStamped(b);
    defer backend_mod.stampRead(b);
    var made: usize = 0;
    errdefer for (out[0..made]) |*l| l.deinit();
    var row0: u32 = 0;
    for (depths, out) |d, *l| {
        const hs: op.Head = .{ .logits = try backend_mod.rowsOf(b, head_out.logits, row0, d), .token_ids = head_out.token_ids };
        l.* = try runtime_lattice.read(b.allocator, &b.frame, try backend_mod.rowsOf(b, topk_view, row0, d), try backend_mod.rowsOf(b, projected, row0, d), c, hs);
        try l.check(c);
        made += 1;
        row0 += d;
    }
}
