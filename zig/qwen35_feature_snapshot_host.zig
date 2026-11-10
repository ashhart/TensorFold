//! Target state, KV and cached prompt features traverse their real snapshot adapters on a byte-addressable driver.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const D = @import("qwen35_cache_bytes_driver.zig").Device;
const t = std.testing;
const cap = 8192;
const at = 4096;
const World = struct {
    d: cuda.Driver,
    e: q.engine.Engine,
    c: q.cache.Cache,
    fn init(w: *World) !void {
        return w.initFormat(false);
    }
    fn initFormat(w: *World, kv8: bool) !void {
        w.d = D.driver();
        w.e = undefined;
        w.e.d = &w.d;
        w.e.gpa = t.allocator;
        w.e.kit = null;
        w.e.stream = .{ .d = &w.d, .handle = null };
        w.e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = cap, .linear = @splat(false) };
        w.e.g.linear[0] = true;
        w.e.kv8 = kv8;
        w.e.n_taps = 1;
        w.e.taps[0] = 0;
        w.e.tap_capacity = 65;
        w.e.prompt_taps = .{ .buffer = try cuda.DeviceBuffer.alloc(&w.d, 2112 * 512), .rows = 2112, .row_bytes = 512, .cache = true };
        w.c = .{ .state = try cuda.DeviceBuffer.alloc(&w.d, w.e.g.stateBytes()), .kv = try cuda.DeviceBuffer.alloc(&w.d, w.e.g.kvBytes(cap, kv8)), .tap = try cuda.DeviceBuffer.alloc(&w.d, 65 * 512), .log = .{ .d = &w.d, .ptr = 0, .len = 0, .borrowed = true }, .cap = cap, .len = at, .written = at, .logits = &.{} };
        fill(w.c.state, 17);
        fill(w.c.kv, 29);
        fill(w.e.prompt_taps.buffer.?, 43);
        try q.prompt_taps.restored(&w.e, &w.c, at);
    }
    fn deinit(w: *World) void {
        w.e.stream.synchronize() catch unreachable;
        w.c.state.free();
        w.c.kv.free();
        w.c.tap.?.free();
        w.e.prompt_taps.deinit();
        std.debug.assert(D.live() == 0 and D.guards == 0 and D.early_frees == 0);
    }
};
fn bytes(b: cuda.DeviceBuffer) []u8 {
    return @as([*]u8, @ptrFromInt(b.ptr))[0..b.len];
}
fn fill(b: cuda.DeviceBuffer, seed: u8) void {
    for (bytes(b), 0..) |*v, i| v.* = @truncate(i *% 17 +% seed);
}
fn hash(b: []const u8) u64 {
    return std.hash.Wyhash.hash(51, b);
}

test "cached feature snapshots: target state KV and full draft feature history round-trip on host and device" {
    for ([_]bool{ false, true }) |kv8| {
        for ([_]bool{ false, true }) |device| {
            var w: World = undefined;
            try w.initFormat(kv8);
            defer w.deinit();
            const state_hash = hash(bytes(w.c.state));
            const tap_hash = hash(bytes(w.e.prompt_taps.buffer.?));
            const snapshot = try q.reuse.saveCache(&w.e, &w.c, at, &.{}, device);
            defer q.reuse.dropCopy(&w.e, snapshot);
            var layout = q.reuse.layoutOf(&w.e, at);
            layout.cap = cap;
            try t.expectEqual(layout.total() + 2112 * 512, snapshot.len);
            @memset(bytes(w.c.state), 0xcc);
            @memset(bytes(w.c.kv), 0xdd);
            @memset(bytes(w.e.prompt_taps.buffer.?), 0xee);
            try q.reuse.restoreCache(&w.e, &w.c, snapshot, &.{});
            try t.expectEqual(state_hash, hash(bytes(w.c.state)));
            try t.expectEqual(tap_hash, hash(bytes(w.e.prompt_taps.buffer.?)));
            const held = w.e.prompt_taps.held.?;
            try t.expectEqual(@intFromPtr(&w.c), held.owner);
            try t.expectEqual(w.c.tap_epoch, held.epoch);
            try t.expectEqual(@as(u64, at), held.range.end);
            var off: usize = 0;
            for (0..layout.count()) |i| {
                const p = layout.piece(i);
                if (p.kv) for (bytes(w.c.kv)[p.off..][0..p.len], 0..) |value, j|
                    try t.expectEqual(@as(u8, @truncate((p.off + j) *% 17 +% 29)), value);
                off += p.len;
            }
            try t.expectEqual(layout.total(), off);
        }
    }
}

test "cached feature snapshots: incompatible feature layouts refuse before copying any state" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const snapshot = try q.reuse.saveCache(&w.e, &w.c, at, &.{}, false);
    defer q.reuse.dropCopy(&w.e, snapshot);
    const before = D.copies;
    snapshot.features.rows -= 1;
    try t.expectError(error.BadReuse, q.reuse.restoreCache(&w.e, &w.c, snapshot, &.{}));
    try t.expectEqual(before, D.copies);
}

test "cached feature snapshots: every failed copy rolls back its host allocation and protects queued writes" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const n = q.reuse.layoutOf(&w.e, at).count() + 1;
    for (1..n + 1) |fail| {
        const before = D.live();
        D.fail_copy_in = fail;
        try t.expectError(error.NotReady, q.reuse.saveCache(&w.e, &w.c, at, &.{}, false));
        try t.expectEqual(before, D.live());
        try t.expectEqual(@as(usize, 0), D.early_frees);
    }
}

fn seedPrefix(w: *World, c: *q.cache.Cache, seed: u8) !void {
    c.len = 1024;
    c.written = 1024;
    c.last = null;
    fill(c.state, seed);
    fill(w.e.prompt_taps.buffer.?, seed + 8);
    try q.prompt_taps.restored(&w.e, c, 1024);
}
fn ids(seed: u32) ![]u32 {
    const out = try t.allocator.alloc(u32, 1600);
    for (out, 0..) |*v, i| v.* = seed + @as(u32, @intCast(i));
    return out;
}

test "cached feature snapshots: radix hits restore full features and target state after cancellation and edits miss" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const l = q.pages.layout(&w.e, cap, 256, 0, 0);
    const p = try q.pages.Pages.init(t.allocator, &w.e, l, cap, 0, 2, 3 * l.need(cap, 0, q.pages.window_rows));
    defer p.deinit();
    const x = try q.prefix.Prefix.initWithGrid(t.allocator, &w.e, p, 4, 1024);
    defer x.deinit();
    const prompt = try ids(500);
    defer t.allocator.free(prompt);
    var active: ?*q.cache.Cache = null;
    defer if (active) |c| {
        _ = p.close(c);
    };
    const cold = try x.open(prompt, 16, 1600, &.{});
    defer t.allocator.free(cold.plan.marks);
    active = cold.cache;
    try seedPrefix(&w, cold.cache, 61);
    const expected_state = hash(bytes(cold.cache.state));
    const expected_taps = hash(bytes(w.e.prompt_taps.buffer.?));
    try x.keep(cold.cache, prompt, 1024);
    try t.expect(p.close(cold.cache));
    active = null;
    prompt[1500] += 1;
    for (0..2) |_| {
        @memset(bytes(w.e.prompt_taps.buffer.?), 0xcc);
        const hit = try x.open(prompt, 16, 1600, &.{});
        defer t.allocator.free(hit.plan.marks);
        active = hit.cache;
        try t.expectEqual(@as(u32, 1024), hit.plan.from);
        try t.expectEqual(@as(usize, 1024), hit.cache.len);
        try t.expectEqual(expected_state, hash(bytes(hit.cache.state)));
        try t.expectEqual(expected_taps, hash(bytes(w.e.prompt_taps.buffer.?)));
        try t.expectEqual(@intFromPtr(hit.cache), w.e.prompt_taps.held.?.owner);
        try t.expect(p.close(hit.cache));
        active = null;
    }
    prompt[10] += 10000;
    const edited = try x.open(prompt, 16, 1600, &.{});
    defer t.allocator.free(edited.plan.marks);
    active = edited.cache;
    try t.expectEqual(@as(u32, 0), edited.plan.from);
    try t.expectEqual(@as(usize, 0), edited.cache.len);
}

test "cached feature snapshots: eviction cannot restore a different prefix's feature history" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const l = q.pages.layout(&w.e, cap, 256, 0, 0);
    const p = try q.pages.Pages.init(t.allocator, &w.e, l, cap, 0, 2, 3 * l.need(cap, 0, q.pages.window_rows));
    defer p.deinit();
    const x = try q.prefix.Prefix.initWithGrid(t.allocator, &w.e, p, 1, 1024);
    defer x.deinit();
    const a = try ids(500);
    defer t.allocator.free(a);
    const b = try ids(7000);
    defer t.allocator.free(b);
    var active: ?*q.cache.Cache = null;
    defer if (active) |c| {
        _ = p.close(c);
    };
    for ([_][]const u32{ a, b }, 0..) |prompt, i| {
        const cold = try x.open(prompt, 16, 1600, &.{});
        defer t.allocator.free(cold.plan.marks);
        active = cold.cache;
        try seedPrefix(&w, cold.cache, @intCast(61 + i));
        try x.keep(cold.cache, prompt, 1024);
        try t.expect(p.close(cold.cache));
        active = null;
    }
    const evicted = try x.open(a, 16, 1600, &.{});
    defer t.allocator.free(evicted.plan.marks);
    active = evicted.cache;
    try t.expectEqual(@as(u32, 0), evicted.plan.from);
}

test "cached feature snapshots: failed restore sources remain pinned until their completion proof" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const snapshot = try q.reuse.saveCache(&w.e, &w.c, at, &.{}, false);
    const retained = cuda.copy_source.retained();
    const live = D.live();
    const previous_level = t.log_level;
    t.log_level = .err;
    defer t.log_level = previous_level;
    D.fail_copy_in = 2;
    D.fail_wait = true;
    try t.expectError(error.NotReady, q.reuse.restoreCache(&w.e, &w.c, snapshot, &.{}));
    const level = t.log_level;
    t.log_level = .err;
    q.reuse.dropCopy(&w.e, snapshot);
    t.log_level = level;
    try t.expectEqual(live, D.live());
    try t.expectEqual(retained + 1, cuda.copy_source.retained());
    try t.expectEqual(@as(usize, 0), D.early_frees);
    D.fail_wait = false;
    try w.e.stream.synchronize();
    try t.expectEqual(retained, cuda.copy_source.retained());
    try t.expectEqual(live - 1, D.live());
}

test "cached feature snapshots: layer identity and shared ownership are checked before any copy" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const snapshot = try q.reuse.saveCache(&w.e, &w.c, at, &.{}, false);
    defer q.reuse.dropCopy(&w.e, snapshot);
    const before = D.copies;
    w.e.taps[0] = 1;
    try t.expectError(error.BadReuse, q.reuse.restoreCache(&w.e, &w.c, snapshot, &.{}));
    try t.expectEqual(before, D.copies);
    w.e.taps[0] = 0;
    w.e.prompt_taps.held.?.owner += 16;
    try t.expectError(error.RowsNotHeld, q.reuse.saveCache(&w.e, &w.c, at, &.{}, false));
    try t.expectEqual(before, D.copies);
}

test "cached feature snapshots: cache-off snapshots keep their original allocation and copy count" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    w.e.prompt_taps.cache = false;
    const before = D.copies;
    const snapshot = try q.reuse.saveCache(&w.e, &w.c, at, &.{}, false);
    defer q.reuse.dropCopy(&w.e, snapshot);
    const layout = q.reuse.layoutOf(&w.e, at);
    try t.expectEqual(layout.total(), snapshot.len);
    try t.expectEqual(layout.count(), D.copies - before);
    try t.expectEqual(@as(usize, 0), snapshot.features.bytes());
}

test "cached feature snapshots: enabling snapshots leaves all sixteen persistent stream runs unchanged" {
    var e: q.engine.Engine = undefined;
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 131072, .linear = @splat(true) };
    for (0..64) |i| e.g.linear[i] = i % 4 != 3;
    e.n_taps = 5;
    for (e.taps[0..5], 0..) |*tap, i| tap.* = @intCast(i);
    e.tap_capacity = 65;
    e.kv8 = true;
    e.rec_budget = .{ .streams = 16, .limit = 1057640448 };
    e.prompt_taps = .{ .cache = false, .rows = 2112, .row_bytes = 51200, .buffer = .{ .d = undefined, .ptr = 0x1000, .len = 2112 * 51200, .borrowed = true } };
    const before = q.pages.layout(&e, 131072, 512, 42270720, 0);
    e.prompt_taps.cache = true;
    const after = q.pages.layout(&e, 131072, 512, 42270720, 0);
    try t.expectEqual(before.run_bytes, after.run_bytes);
    try t.expectEqual(@as(u64, 199542784), after.run_bytes);
    try t.expectEqual(@as(usize, 12), after.runPages());
    try t.expectEqual(@as(usize, 40), after.need(7680, 6144, 128));
    try t.expectEqual(@as(u64, 262078464), after.snap_bytes);
    try t.expectEqual(before.snap_bytes + 2112 * 51200, after.snap_bytes);
}

test "cached feature snapshots: reclaim during a warm open preserves the selected snapshot source" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const l = q.pages.layout(&w.e, cap, 256, 0, 0);
    const p = try q.pages.Pages.init(t.allocator, &w.e, l, cap, 0, 2, 18);
    defer p.deinit();
    const x = try q.prefix.Prefix.initWithGrid(t.allocator, &w.e, p, 4, 512);
    defer x.deinit();
    const a = try ids(500);
    defer t.allocator.free(a);
    const b = try t.allocator.alloc(u32, 2561);
    defer t.allocator.free(b);
    for (b, 0..) |*v, i| v.* = @intCast(7000 + i);
    var active: ?*q.cache.Cache = null;
    defer if (active) |c| {
        _ = p.close(c);
    };
    const first = try x.open(a[0..1025], 0, 1025, &.{});
    defer t.allocator.free(first.plan.marks);
    active = first.cache;
    try seedPrefix(&w, first.cache, 61);
    const expected_state = hash(bytes(first.cache.state));
    const expected_taps = hash(bytes(w.e.prompt_taps.buffer.?));
    try x.keep(first.cache, a[0..1025], 1024);
    try t.expect(p.close(first.cache));
    active = null;
    const second = try x.open(b, 0, 2561, &.{});
    defer t.allocator.free(second.plan.marks);
    active = second.cache;
    second.cache.len = 2560;
    second.cache.written = 2560;
    fill(second.cache.state, 83);
    fill(w.e.prompt_taps.buffer.?, 91);
    try q.prompt_taps.restored(&w.e, second.cache, 2560);
    try x.keep(second.cache, b, 2560);
    try t.expect(p.close(second.cache));
    active = null;
    try t.expectEqual(@as(usize, 4), p.pool.census().free);
    const hit = try x.open(a, 16, 1600, &.{});
    defer t.allocator.free(hit.plan.marks);
    active = hit.cache;
    try t.expectEqual(@as(u32, 1024), hit.plan.from);
    try t.expectEqual(expected_state, hash(bytes(hit.cache.state)));
    try t.expectEqual(expected_taps, hash(bytes(w.e.prompt_taps.buffer.?)));
}

test "cached feature snapshots: failed capture destinations remain pinned until their queued writes drain" {
    const level = t.log_level;
    t.log_level = .err;
    defer t.log_level = level;
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const retained = cuda.copy_source.retained();
    const live = D.live();
    D.fail_copy_in = q.reuse.layoutOf(&w.e, at).count() + 1;
    D.fail_wait_on_copy_error = true;
    try t.expectError(error.NotReady, q.reuse.saveCache(&w.e, &w.c, at, &.{}, false));
    try t.expectEqual(retained + 1, cuda.copy_source.retained());
    try t.expectEqual(live + 1, D.live());
    try t.expectEqual(@as(usize, 0), D.early_frees);
    D.fail_wait = false;
    D.fail_wait_on_copy_error = false;
    try w.e.stream.synchronize();
    try t.expectEqual(retained, cuda.copy_source.retained());
    try t.expectEqual(live, D.live());
}
