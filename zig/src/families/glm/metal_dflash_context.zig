//! GLM's draft context: only completed committed-tap transactions advance the drafter's rolling window.
const std = @import("std");
const mtl = @import("metal");
const op = @import("../qwen27/dflash/operators.zig");
const Backend = @import("metal_dflash_backend.zig").Backend;
const Cache = @import("metal_dflash_backend.zig").Cache;
const dattn = @import("metal_dflash_attention.zig");

/// Rows a stream's K/V ring cycles through: all its caches hold, so rows past the context's end miss its keys.
pub fn ring(c: @import("../qwen27/dflash/config.zig").Config) u32 {
    return c.window + 16;
}

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

/// A batched pass's stream `stream`: its new taps' keys and values into its own caches (the backend's batch).
pub fn streamAppend(ptr: *anyopaque, epoch: u64, stream: u32, layer: u32, keys: op.View, values: op.View, positions: []const u64) !void {
    const b = cast(ptr);
    const e = try b.encoder(epoch);
    if (b.transaction != null or stream >= b.batch.len or layer >= b.config.layers or positions.len == 0) return error.BadDraftContext;
    try keys.check(@intCast(positions.len), b.config.kvWidth(), .bf16);
    try values.check(keys.rows, keys.width, .bf16);
    try appendRows(b, e, b.batch[stream][layer], keys, values, positions);
}

/// Rows' keys and values into `cache` at their positions' ring slots, 16 a dispatch.
fn appendRows(b: *Backend, e: mtl.ComputeEncoder, cache: Cache, keys: op.View, values: op.View, positions: []const u64) !void {
    const k = try b.frame.ref(keys);
    const v = try b.frame.ref(values);
    var first: usize = 0;
    while (first < positions.len) : (first += 16) {
        const count = @min(positions.len - first, 16);
        var slots: [16]u32 = undefined;
        for (slots[0..count], 0..) |*slot, i| slot.* = @intCast(positions[first + i] % ring(b.config));
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

pub fn attention(ptr: *anyopaque, epoch: u64, p: op.Attention) !op.View {
    const b = cast(ptr);
    const e = try b.encoder(epoch);
    if (b.transaction != null or p.layer >= b.config.layers or p.mask != .all_slots) return error.BadDraftContext;
    if (b.follow) |f| return attendFollow(b, e, p, f);
    var rings: [dattn.max_streams]dattn.Ring = undefined;
    var contexts: [dattn.max_streams]op.Context = undefined;
    var n: usize = 1;
    var block = p.query.rows;
    if (p.streams.len == 0) {
        if (!std.meta.eql(p.context, b.context)) return error.BadDraftContext;
        contexts[0] = p.context;
        rings[0] = .{ .k = .{ .buf = b.caches[p.layer].keys }, .v = .{ .buf = b.caches[p.layer].values } };
    } else {
        n = p.streams.len;
        if (n > dattn.max_streams or n > b.batch.len or p.block == 0 or p.query.rows != n * p.block) return error.BadDraftContext;
        block = p.block;
        for (p.streams, 0..) |context, k| {
            contexts[k] = context;
            rings[k] = .{ .k = .{ .buf = b.batch[k][p.layer].keys }, .v = .{ .buf = b.batch[k][p.layer].values } };
        }
    }
    const result = try b.frame.alloc(p.query.rows, b.config.qWidth(), .bf16);
    try attendStreams(b, e, p, contexts[0..n], rings[0..n], block, result);
    e.barrier();
    return result;
}

/// Each stream's block rows over its context's cached keys (read where its ring holds them) and its own, one launch.
fn attendStreams(b: *Backend, e: mtl.ComputeEncoder, p: op.Attention, contexts: []const op.Context, rings: []const dattn.Ring, block: u32, result: op.View) !void {
    const width = b.config.kvWidth();
    if (p.keys.stride != width or p.values.stride != width or p.query.stride != b.config.qWidth() or result.stride != b.config.qWidth()) return error.DraftOperatorShape;
    var table: [dattn.max_streams]dattn.Stream = undefined;
    for (contexts, table[0..contexts.len]) |context, *t| {
        const context_rows: u32 = @intCast(context.end - context.begin);
        t.* = .{ .keys = context_rows + block, .context = context_rows, .at = @intCast(context.begin % ring(b.config)) };
    }
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(b.config.head_dim)));
    const args = try dattn.plan(table[0..contexts.len], block, ring(b.config), scale);
    const mask = try b.frame.ref(try b.frame.alloc(args.rows, args.mask_stride, .u8));
    const bytes = mask.buf.contents()[mask.off..][0 .. @as(usize, args.rows) * args.mask_stride];
    var one: [16 * (2048 + 16)]u8 = undefined;
    for (table[0..contexts.len], 0..) |t, k| {
        const m = one[0 .. @as(usize, block) * t.keys];
        try fillMask(m, t.context, block, b.config.window, b.config.block);
        for (0..block) |r| @memcpy(bytes[(k * block + r) * args.mask_stride ..][0..t.keys], m[r * t.keys ..][0..t.keys]);
    }
    const part = try b.frame.ref(try b.frame.alloc(1, @intCast(dattn.partBytes(b.draft_attention.shape, args)), .u8));
    const q = try b.frame.ref(p.query);
    const own_k = try b.frame.ref(p.keys);
    const own_v = try b.frame.ref(p.values);
    const y = try b.frame.ref(result);
    try b.draft_attention.encode(e, args, table[0..contexts.len], rings, .{ .buf = q.buf, .off = q.off }, .{ .buf = own_k.buf, .off = own_k.off }, .{ .buf = own_v.buf, .off = own_v.off }, .{ .buf = mask.buf, .off = mask.off }, .{ .buf = part.buf, .off = part.off }, .{ .buf = y.buf, .off = y.off });
}

/// A block planned on the GPU: each stream's rows over its longest context, the mask the follow kernel wrote.
fn attendFollow(b: *Backend, e: mtl.ComputeEncoder, p: op.Attention, f: @import("metal_dflash_backend.zig").Device) !op.View {
    const n = f.streams;
    if (n == 0 or n > dattn.max_streams or n > b.batch.len or p.query.rows % n != 0) return error.BadDraftContext;
    const block = p.query.rows / n;
    const width = b.config.kvWidth();
    if (p.keys.stride != width or p.values.stride != width or p.query.stride != b.config.qWidth()) return error.DraftOperatorShape;
    var rings: [dattn.max_streams]dattn.Ring = undefined;
    for (rings[0..n], 0..) |*r, k| r.* = .{ .k = .{ .buf = b.batch[k][p.layer].keys }, .v = .{ .buf = b.batch[k][p.layer].values } };
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(b.config.head_dim)));
    const args = try dattn.plan(f.keys[0..n], block, ring(b.config), scale);
    if (args.mask_stride != f.mask_stride) return error.BadDraftContext;
    const result = try b.frame.alloc(p.query.rows, b.config.qWidth(), .bf16);
    const part = try b.frame.ref(try b.frame.alloc(1, @intCast(dattn.partBytes(b.draft_attention.shape, args)), .u8));
    const q = try b.frame.ref(p.query);
    const own_k = try b.frame.ref(p.keys);
    const own_v = try b.frame.ref(p.values);
    const y = try b.frame.ref(result);
    try b.draft_attention.encode(e, args, f.keys[0..n], rings[0..n], .{ .buf = q.buf, .off = q.off }, .{ .buf = own_k.buf, .off = own_k.off }, .{ .buf = own_v.buf, .off = own_v.off }, .{ .buf = f.mask.buf, .off = f.mask.off }, .{ .buf = part.buf, .off = part.off }, .{ .buf = y.buf, .off = y.off });
    e.barrier();
    return result;
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

test "a stream's K/V ring cycles through every row its caches hold" {
    const c: @import("../qwen27/dflash/config.zig").Config = .{};
    const caches = @as(usize, c.window + 16) * c.kvWidth() * 2; // caches.zig's rows a layer
    try std.testing.expectEqual(caches, @as(usize, ring(c)) * c.kvWidth() * 2);
    try std.testing.expect(ring(c) >= c.window - 1 + 16); // a full context plus a window's rows past its end
}

test "GLM's sliding context advances per query and hides later slots from trained ones" {
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
