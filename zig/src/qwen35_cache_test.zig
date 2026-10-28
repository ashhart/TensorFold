//! The actual Qwen snapshot and prompt lifecycle on host-backed buffers; token arithmetic is a sentinel.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const q = @import("families/qwen3_5/backend.zig");
const c = @import("families/qwen3_5/config.zig");
const st = @import("families/qwen3_5/state.zig");
const snap = @import("families/qwen3_5/snapshot.zig");
const prefill = @import("families/qwen3_5/prefill.zig");
const equal = @import("families/qwen3_5/qualify.zig").equal;
const pc = @import("core/prompt_cache.zig");
const Pages = @import("core/metal_kv_pages.zig").Pages;
const t = std.testing;
const a = t.allocator;

test {
    _ = c;
    _ = st;
    _ = @import("families/qwen3_5/draw.zig");
}

fn seed(cache: *st.Cache) void {
    @memset(cache.logits.contents()[0 .. c.vocab * 2], 0);
    for (cache.blocks) |block| if (block == .delta) {
        @memset(block.delta.recurrence.contents()[0..st.delta_bytes], 0);
        @memset(block.delta.conv.contents()[0..st.conv_bytes], 0);
    };
}

fn step(_: void, cache: *st.Cache, ids: []const u32) !void {
    for (ids) |id| {
        var sum: u32 = id;
        for (cache.blocks, 0..) |block, layer| switch (block) {
            .delta => |d| {
                const state = d.recurrence.slice(u32, st.delta_bytes / 4);
                const conv = d.conv.slice(u16, st.conv_bytes / 2);
                state[0] = state[0] *% 31 +% id +% @as(u32, @intCast(layer));
                state[state.len - 1] +%= state[0];
                conv[0] +%= @truncate(id);
                conv[conv.len - 1] +%= conv[0];
                sum +%= state[0] +% state[state.len - 1] +% conv[conv.len - 1];
            },
            .attention => |kv| for ([_]mtl.Buffer{ kv.keys, kv.values }, 0..) |buf, value| for (0..c.kv_heads) |h| {
                const row = (buf.contents() + (h * cache.capacity + cache.len) * c.head_dim * 2)[0 .. c.head_dim * 2];
                @memset(row, @truncate(id +% @as(u32, @intCast(layer * 7 + h * 3 + value))));
                for (0..cache.len + 1) |r| sum +%= buf.contents()[(h * cache.capacity + r) * c.head_dim * 2];
            },
        };
        const logits = cache.logits.slice(u32, c.vocab / 2);
        logits[0] = sum;
        logits[logits.len - 1] +%= sum;
        cache.len += 1;
    }
}

fn reply(cache: *st.Cache) ![8]u32 {
    var out: [8]u32 = undefined;
    for (&out) |*id| {
        id.* = cache.logits.slice(u32, 1)[0] % 97;
        try step({}, cache, &.{id.*});
    }
    return out;
}

fn tokens() [350]u32 {
    var ids: [350]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i * 17 % 101);
    return ids;
}

test "Qwen retained: snapshot copies all committed bytes across unequal capacities" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    var source = try st.Cache.init(a, device, 320);
    defer source.deinit();
    for (source.memory.buffers.items, 0..) |buf, k| {
        for (buf.data, 0..) |*b, i| b.* = @truncate(i * 7 + k * 19);
    }
    source.len = 257;
    source.last = .{ .start = 254, .base = 0, .rows = 3 };
    const saved = try snap.save(a, &source, 257);
    defer snap.drop(a, saved);
    try t.expectEqual(st.cacheBytes(257), saved.bytes.len);
    var restored = try st.Cache.init(a, device, 384);
    defer restored.deinit();
    for (restored.memory.buffers.items) |buf| @memset(buf.data, 0xe7);
    restored.last = source.last;
    try snap.restore(&restored, saved);
    try equal(&source, &restored);
    try t.expect(restored.last == null);
    restored.blocks[0].delta.conv.contents()[st.conv_bytes - 1] ^= 1;
    try t.expectError(error.QwenRecurrenceMismatch, equal(&source, &restored));
    restored.blocks[0].delta.conv.contents()[st.conv_bytes - 1] ^= 1;
    restored.logits.contents()[c.vocab * 2 - 1] ^= 1;
    try t.expectError(error.QwenCommittedLogitsMismatch, equal(&source, &restored));
}

test "Qwen retained: paged attention restores head strides, page boundaries and untouched tails" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    var source = try st.Cache.init(a, device, 300);
    defer source.deinit();
    for (source.memory.buffers.items, 0..) |buf, k| {
        for (buf.data, 0..) |*b, i| b.* = @truncate(i * 11 + k * 23);
    }
    source.len = 257;
    const saved = try snap.save(a, &source, 257);
    defer snap.drop(a, saved);
    const layout = st.pagedLayout();
    const sized = Pages.bytes(layout, 2, 4, 512);
    const arena = try device.buffer(@intCast(sized.arena), 0);
    defer arena.deinit();
    const tables = try device.buffer(@intCast(sized.tables), 0);
    defer tables.deinit();
    const pages = try Pages.initIn(a, layout, 2, 4, 512, arena, tables);
    defer pages.deinit();
    try pages.admit(0, 2);
    try pages.cover(0, 256);
    try pages.admit(1, 2);
    try pages.cover(1, 512);
    const ids = pages.leases[1].ids.items;
    std.mem.swap(u32, &ids[0], &ids[1]);
    for (ids, 0..) |id, i| pages.tables.set(1, i, id);
    var paged = try st.Cache.initPaged(a, device, 512, .{ .pages = pages, .lane = 1 });
    defer paged.deinit();
    @memset(pages.arena.buf.?.data, 0xd7);
    try snap.restore(&paged, saved);
    const other = (arena.contents() + @as(usize, pages.ids(0)[0]) * @as(usize, @intCast(layout.pageBytes())))[0..@intCast(layout.pageBytes())];
    for (other) |b| try t.expectEqual(@as(u8, 0xd7), b);
    var layer: usize = 0;
    for (source.blocks) |block| if (block == .attention) {
        for ([_]mtl.Buffer{ block.attention.keys, block.attention.values }, 0..) |buf, v| {
            for (0..c.kv_heads) |h| for (0..512) |r| {
                const page = pages.ids(1)[r / 256];
                const offset = @as(usize, page) * @as(usize, @intCast(layout.pageBytes())) +
                    st.pagePart(layer, v == 1) + (h * 256 + r % 256) * c.head_dim * 2;
                const got = (pages.arena.buf.?.contents() + offset)[0 .. c.head_dim * 2];
                if (r < source.len) {
                    const expected = (buf.contents() + (h * source.capacity + r) * c.head_dim * 2)[0 .. c.head_dim * 2];
                    try t.expectEqualSlices(u8, expected, got);
                } else for (got) |b| try t.expectEqual(@as(u8, 0xd7), b);
            };
        }
        layer += 1;
    };
    const copied = try snap.save(a, &paged, 257);
    defer snap.drop(a, copied);
    try t.expectEqualSlices(u8, saved.bytes, copied.bytes);
    var contiguous = try st.Cache.init(a, device, 400);
    defer contiguous.deinit();
    try snap.restore(&contiguous, copied);
    try equal(&source, &contiguous);
}

fn allocate(gpa: std.mem.Allocator, cache: *st.Cache) !void {
    const saved = try snap.save(gpa, cache, @intCast(cache.len));
    snap.drop(gpa, saved);
}

test "Qwen retained: malformed positions and lengths refuse before changing a cache; save unwinds allocation failures" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    var source = try st.Cache.init(a, device, 5);
    defer source.deinit();
    seed(&source);
    try step({}, &source, &.{ 1, 2, 3 });
    try t.expectError(error.QwenSnapshotPosition, snap.save(a, &source, 2));
    try t.checkAllAllocationFailures(a, allocate, .{&source});
    const saved = try snap.save(a, &source, 3);
    defer snap.drop(a, saved);
    var short = try st.Cache.init(a, device, 2);
    defer short.deinit();
    try t.expectError(error.QwenSnapshotPosition, snap.restore(&short, saved));
    var target = try st.Cache.init(a, device, 5);
    defer target.deinit();
    const invalid: snap.Snap = .{ .at = 3, .bytes = saved.bytes[0 .. saved.bytes.len - 1] };
    try t.expectError(error.QwenSnapshotPosition, snap.restore(&target, &invalid));
    try t.expectEqual(@as(usize, 0), target.len);
    try snap.restore(&target, saved);
    try t.expectError(error.QwenSnapshotPosition, snap.restore(&target, saved));
}

const Harness = struct {
    metal: q.Metal = undefined,
    store: pc.Store = undefined,
    active: ?*st.Cache = null,
    cancel_at: ?usize = null,
    fail_at: ?usize = null,

    fn init(h: *Harness) void {
        h.metal.gpa = a;
        h.metal.caches = .empty;
        const S = q.Snaps;
        h.store = pc.Store.init(a, .{ .ptr = &h.metal, .vtable = &.{ .bytes = S.bytes, .save = S.save, .restore = S.restore, .drop = S.drop } }, .{ .planned = true, .grid = 128, .chunk = 128 }, 128 << 20);
    }
    fn deinit(h: *Harness) void {
        h.store.deinit();
        h.metal.caches.deinit(a);
    }
    fn kept(ptr: *anyopaque, stream: *lanes.Stream, at: u32) void {
        const h: *Harness = @ptrCast(@alignCast(ptr));
        _ = h.store.keep(stream.prompt(), at, stream, stream.chunks, stream.decode_spans);
    }
    fn cancelled(ptr: *anyopaque) bool {
        const h: *Harness = @ptrCast(@alignCast(ptr));
        return if (h.cancel_at) |at| h.active.?.len >= at else false;
    }
    fn advance(h: *Harness, cache: *st.Cache, ids: []const u32) !void {
        if (h.fail_at) |at| if (cache.len >= at) return error.FakeForward;
        try step({}, cache, ids);
    }
    fn pass(h: *Harness, cache: *st.Cache, ids: []const u32, cuts: []const u32) !u32 {
        const lookup = try h.store.lookup(a, ids, 128, &.{}, cuts, &.{});
        defer a.free(lookup.marks);
        var stream = try lanes.Stream.init(a, .{ .id = "fixture", .prompt = ids, .max_new = 8, .chunks = cuts, .cancel_check = .{ .ptr = h, .check = cancelled }, .reuse = .{
            .saved = if (lookup.entry) |entry| entry.saved else null,
            .at = if (lookup.entry) |entry| entry.at else 0,
            .marks = lookup.marks,
            .hook = .{ .ptr = h, .at = kept },
        } });
        defer stream.deinit(a);
        defer if (lookup.entry) |entry| h.store.resumedWithReason(entry, ids, if (stream.reuse_failed) stream.reuse_error orelse error.FakeRestore else null);
        try h.metal.caches.put(a, &stream, cache);
        defer _ = h.metal.caches.remove(&stream);
        h.active = cache;
        defer h.active = null;
        try prefill.run(cache, &stream, 128, h, advance);
        return stream.cached;
    }
};

test "Qwen retained: store resume equals fresh sentinel tokens, and incompatible cuts run fresh" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const ids = tokens();
    var h: Harness = .{};
    h.init();
    defer h.deinit();
    var first = try st.Cache.init(a, device, 400);
    defer first.deinit();
    seed(&first);
    try t.expectEqual(@as(u32, 0), try h.pass(&first, ids[0..300], &.{}));
    try t.expect(h.store.entries.items.len > 0);
    var resumed = try st.Cache.init(a, device, 430);
    defer resumed.deinit();
    seed(&resumed);
    try t.expectEqual(@as(u32, 128), try h.pass(&resumed, &ids, &.{}));
    var fresh = try st.Cache.init(a, device, 450);
    defer fresh.deinit();
    seed(&fresh);
    try step({}, &fresh, &ids);
    try equal(&resumed, &fresh);
    try t.expectEqualSlices(u32, &(try reply(&fresh)), &(try reply(&resumed)));
    var changed = try st.Cache.init(a, device, 420);
    defer changed.deinit();
    seed(&changed);
    try t.expectEqual(@as(u32, 0), try h.pass(&changed, &ids, &.{ 0, 64, 192, 320 }));
}

test "Qwen retained: cancellation after restore preserves the valid prefix and a failed restore falls back" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const ids = tokens();
    var h: Harness = .{};
    h.init();
    defer h.deinit();
    var first = try st.Cache.init(a, device, 400);
    defer first.deinit();
    seed(&first);
    _ = try h.pass(&first, ids[0..300], &.{});
    var cancelled = try st.Cache.init(a, device, 410);
    defer cancelled.deinit();
    h.cancel_at = 128;
    try t.expectError(error.Cancelled, h.pass(&cancelled, &ids, &.{}));
    try t.expectEqual(@as(usize, 1), h.store.entries.items.len);
    h.cancel_at = null;
    const saved: *snap.Snap = @ptrCast(@alignCast(h.store.entries.items[0].saved));
    saved.at = 401;
    var fallback = try st.Cache.init(a, device, 420);
    defer fallback.deinit();
    seed(&fallback);
    try t.expectEqual(@as(u32, 0), try h.pass(&fallback, &ids, &.{}));
    var fresh = try st.Cache.init(a, device, 440);
    defer fresh.deinit();
    seed(&fresh);
    try step({}, &fresh, &ids);
    try equal(&fallback, &fresh);
    try t.expect(h.store.counts.failed > 0);
}

test "Qwen retained: a failed chunk never captures its incomplete state" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const ids = tokens();
    var h: Harness = .{};
    h.init();
    defer h.deinit();
    h.fail_at = 0;
    var failed = try st.Cache.init(a, device, 400);
    defer failed.deinit();
    seed(&failed);
    try t.expectError(error.FakeForward, h.pass(&failed, ids[0..300], &.{}));
    try t.expectEqual(@as(usize, 0), h.store.entries.items.len);
}
