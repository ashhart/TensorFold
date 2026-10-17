//! Only completed committed-tap transactions advance the drafter's absolute rolling context.
const std = @import("std");
const op = @import("operators.zig");
const Backend = @import("runtime_backend.zig").Backend;
const Ref = @import("runtime_backend.zig").Ref;
fn cast(ptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(ptr));
}
pub fn begin(ptr: *anyopaque, current: op.Context) !u64 {
    const b = cast(ptr);
    if (b.failed or b.transaction != null or !std.meta.eql(current, b.context)) return error.BadDraftContext;
    try b.frame.require(b.frame.epoch);
    b.transaction = b.frame.epoch;
    b.appended = 0;
    b.appended_rows = 0;
    return b.frame.epoch;
}
pub fn append(ptr: *anyopaque, transaction: u64, epoch: u64, layer: u32, keys: op.View, values: op.View, positions: []const u64) !void {
    const b = cast(ptr);
    const e = try b.encoder(epoch);
    if (b.transaction != transaction or transaction != epoch or layer >= b.config.layers or b.appended & (@as(u32, 1) << @intCast(layer)) != 0 or positions.len == 0) return error.BadDraftContext;
    try keys.check(@intCast(positions.len), b.config.kvWidth(), .bf16);
    try values.check(keys.rows, keys.width, .bf16);
    if (b.appended != 0 and b.appended_rows != positions.len) return error.BadCommittedTaps;
    for (positions, 0..) |position, i| if (position != b.context.end + i) return error.BadCommittedTaps;
    try appendRows(b, e, b.caches[layer], keys, values, positions);
    b.appended |= @as(u32, 1) << @intCast(layer);
    b.appended_rows = @intCast(positions.len);
}
pub fn streamAppend(ptr: *anyopaque, epoch: u64, stream: u32, layer: u32, keys: op.View, values: op.View, positions: []const u64) !void {
    const b = cast(ptr);
    const e = try b.encoder(epoch);
    if (b.transaction != null or stream >= b.batch.len or layer >= b.config.layers or positions.len == 0) return error.BadDraftContext;
    try keys.check(@intCast(positions.len), b.config.kvWidth(), .bf16);
    try values.check(keys.rows, keys.width, .bf16);
    try appendRows(b, e, b.batch[stream][layer], keys, values, positions);
}
/// Rows' keys and values into `cache` at their positions' ring slots, 16 a dispatch.
fn appendRows(b: *Backend, e: @import("metal").ComputeEncoder, cache: @import("runtime_backend.zig").Cache, keys: op.View, values: op.View, positions: []const u64) !void {
    const k = try b.frame.ref(keys);
    const v = try b.frame.ref(values);
    var first: usize = 0;
    while (first < positions.len) : (first += 16) {
        const count = @min(positions.len - first, 16);
        var slots: [16]u32 = undefined;
        for (slots[0..count], 0..) |*slot, i| slot.* = @intCast(positions[first + i] % b.config.window);
        const off = first * b.config.kvWidth() * 2;
        try b.attention.appendKv(e, slots[0..count], .{ .buf = k.buf, .off = k.off + off }, .{ .buf = v.buf, .off = v.off + off }, .{ .buf = cache.keys }, .{ .buf = cache.values });
        e.barrier();
    }
}
fn ready(b: *Backend, transaction: u64, epoch: u64, next: op.Context) !void {
    if (b.transaction != transaction or transaction != epoch or b.appended != (@as(u32, 1) << @intCast(b.config.layers)) - 1 or next.cache != b.context.cache or next.end != b.context.end + b.appended_rows or next.begin != @max(b.context.begin, next.end -| (@as(u64, b.config.window) - 1))) return error.BadDraftContext;
}
pub fn commit(ptr: *anyopaque, transaction: u64, epoch: u64, next: op.Context) !void {
    const b = cast(ptr);
    try ready(b, transaction, epoch, next);
    b.frame.finish() catch |err| {
        b.failed = true;
        return err;
    };
    b.context = next;
    b.transaction = null;
}
/// The context advances now and the frame stays open; if the frame then never completes, abort fails the backend.
pub fn stage(ptr: *anyopaque, transaction: u64, epoch: u64, next: op.Context) !void {
    const b = cast(ptr);
    try ready(b, transaction, epoch, next);
    b.context = next;
    b.transaction = null;
    b.staged = transaction;
}
pub fn abort(ptr: *anyopaque, transaction: u64) void {
    const b = cast(ptr);
    if (b.transaction == transaction) {
        if (b.frame.completed) b.failed = true;
        b.transaction = null;
    }
    if (b.staged == transaction) {
        if (!b.frame.completed) b.failed = true;
        b.staged = null;
    }
}
fn gather(b: *Backend, e: @import("metal").ComputeEncoder, context: op.Context, cache: @import("metal").Buffer, out: Ref) !void {
    const rows: u32 = @intCast(context.end - context.begin);
    if (rows == 0) return;
    const at: u32 = @intCast(context.begin % b.config.window);
    const first = @min(rows, b.config.window - at);
    const width = b.config.kvWidth();
    try b.copy(e, .{ .buf = cache, .off = @as(usize, at) * width * 2 }, out, first, width);
    if (first < rows) try b.copy(e, .{ .buf = cache }, .{ .buf = out.buf, .off = out.off + @as(usize, first) * width * 2 }, rows - first, width);
}
pub fn attention(ptr: *anyopaque, epoch: u64, p: op.Attention) !op.View {
    const b = cast(ptr);
    const e = try b.encoder(epoch);
    if (b.transaction != null or p.layer >= b.config.layers or p.mask != .all_slots) return error.BadDraftContext;
    const result = try b.frame.alloc(p.query.rows, b.config.qWidth(), .bf16);
    if (p.streams.len == 0) {
        if (!std.meta.eql(p.context, b.context)) return error.BadDraftContext;
        try attendRows(b, e, p, p.context, b.caches, 0, p.query.rows, result);
    } else {
        if (p.streams.len > b.batch.len or p.block == 0 or p.query.rows != p.streams.len * p.block) return error.BadDraftContext;
        for (p.streams, 0..) |context, k| try attendRows(b, e, p, context, b.batch[k], @intCast(k * p.block), p.block, result);
    }
    e.barrier();
    return result;
}

/// Query rows [first, first + rows) against `context`'s cached keys and their own, into the same rows of `result`.
fn attendRows(b: *Backend, e: @import("metal").ComputeEncoder, p: op.Attention, context: op.Context, caches: []const @import("runtime_backend.zig").Cache, first: u32, rows: u32, result: op.View) !void {
    const context_rows: u32 = @intCast(context.end - context.begin);
    const keys = context_rows + rows;
    const merged_k = try b.frame.alloc(keys, b.config.kvWidth(), .bf16);
    const merged_v = try b.frame.alloc(keys, b.config.kvWidth(), .bf16);
    const k = try b.frame.ref(merged_k);
    const v = try b.frame.ref(merged_v);
    try gather(b, e, context, caches[p.layer].keys, k);
    try gather(b, e, context, caches[p.layer].values, v);
    const off = @as(usize, context_rows) * b.config.kvWidth() * 2;
    const own_k = try b.frame.ref(p.keys);
    const own_v = try b.frame.ref(p.values);
    try b.copy(e, .{ .buf = own_k.buf, .off = own_k.off + @as(usize, first) * p.keys.stride * 2 }, .{ .buf = k.buf, .off = k.off + off }, rows, p.keys.width);
    try b.copy(e, .{ .buf = own_v.buf, .off = own_v.off + @as(usize, first) * p.values.stride * 2 }, .{ .buf = v.buf, .off = v.off + off }, rows, p.values.width);
    const mask = try b.frame.alloc(rows, keys, .u8);
    const mask_ref = try b.frame.ref(mask);
    try fillMask(mask_ref.buf.contents()[mask_ref.off..][0 .. @as(usize, rows) * keys], context_rows, rows, b.config.window, b.config.block);
    const q = try b.frame.ref(p.query);
    const y = try b.frame.ref(result);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(b.config.head_dim)));
    const R = @import("core").shared_attention.Ref;
    const views = [_]R{ .{ .buf = q.buf, .off = q.off + @as(usize, first) * p.query.stride * 2 }, .{ .buf = k.buf, .off = k.off }, .{ .buf = v.buf, .off = v.off }, .{ .buf = mask_ref.buf, .off = mask_ref.off }, .{ .buf = y.buf, .off = y.off + @as(usize, first) * result.stride * 2 } };
    // Long contexts share each 256-key chunk across the query heads of one KV head.
    if (keys > 1024) try b.attention.encodeSplit(e, rows, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]) else try b.attention.encode(e, rows, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]);
}

/// Slots before `trained` cannot see later slots, retaining their trained-block mask.
pub fn fillMask(out: []u8, context_rows: u32, queries: u32, window: u32, trained: u32) !void {
    if (queries == 0 or window == 0 or context_rows >= window or trained == 0) return error.BadDraftContext;
    const keys = try std.math.add(u32, context_rows, queries);
    if (out.len != try std.math.mul(usize, queries, keys)) return error.BadDraftContext;
    @memset(out, 1);
    for (0..queries) |query| {
        const hidden = @min(context_rows, (@as(usize, context_rows) + query + 1) -| window);
        @memset(out[query * keys ..][0..hidden], 0);
        if (query < trained and queries > trained) @memset(out[query * keys + context_rows + trained ..][0 .. queries - trained], 0);
    }
}
test "sliding context advances per query; trained slots see no later slot, later slots see every slot" {
    const a = std.testing.allocator;
    for ([_]u32{ 0, 1, 63, 2047 }) |context| {
        const window: u32 = if (context == 2047) 2048 else 64;
        for ([_]u32{ 8, 16 }) |queries| {
            const keys = context + queries;
            const mask = try a.alloc(u8, keys * queries);
            defer a.free(mask);
            try fillMask(mask, context, queries, window, 8);
            for (0..queries) |query| for (0..keys) |key| {
                const late = key >= context + 8 and query < 8;
                const visible = (key >= context or context + query < key + window) and !late;
                try std.testing.expectEqual(@as(u8, @intFromBool(visible)), mask[query * keys + key]);
            };
        }
    }
}

test "sliding context advances per query while every proposal slot remains visible" {
    const a = std.testing.allocator;
    for ([_]u32{ 0, 1, 63, 2047 }) |context| {
        const window: u32 = if (context == 2047) 2048 else 64;
        for ([_]u32{ 8, 16 }) |queries| {
            const keys = context + queries;
            const mask = try a.alloc(u8, keys * queries);
            defer a.free(mask);
            try fillMask(mask, context, queries, window, queries);
            for (0..queries) |query| for (0..keys) |key| {
                const visible = key >= context or context + query < key + window;
                try std.testing.expectEqual(@as(u8, @intFromBool(visible)), mask[query * keys + key]);
            };
        }
    }
}
