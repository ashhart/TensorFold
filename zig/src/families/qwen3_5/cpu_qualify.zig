//! Admit wider CPU lanes only after forward rows and committed caches match one-row execution (qualify.zig's twin).
const std = @import("std");
const Model = @import("cpu_model.zig").Model;
const st = @import("cpu_state.zig");
const fwd = @import("cpu_forward.zig");
const c = @import("config.zig");

pub fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    if (a.len != b.len) return error.QwenCacheLengthMismatch;
    if (!std.mem.eql(u16, a.logits, b.logits)) return error.QwenCommittedLogitsMismatch;
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .delta => |d| {
            if (!std.mem.eql(f32, d.recurrence, y.delta.recurrence) or !std.mem.eql(u16, d.conv, y.delta.conv)) return error.QwenRecurrenceMismatch;
        },
        .attention => |kv| {
            for ([_][]const u16{ kv.keys, kv.values }, [_][]const u16{ y.attention.keys, y.attention.values }) |ab, bb| {
                for (0..c.kv_heads) |h| {
                    const n = a.len * c.head_dim;
                    if (!std.mem.eql(u16, ab[h * a.capacity * c.head_dim ..][0..n], bb[h * b.capacity * c.head_dim ..][0..n])) return error.QwenAttentionCacheMismatch;
                }
            }
        },
    };
}

/// Two 16-row windows over two streams against one-row steps, then kept prefixes against fresh replays.
pub fn check(m: *const Model, wide: *st.Scratch) !void {
    var solo = try st.Scratch.init(m.gpa, 1, 64, m.pool.workers());
    defer solo.deinit();
    var caches: [4]st.Cache = undefined;
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*cache| cache.deinit();
    for (&caches) |*cache| {
        cache.* = try st.Cache.init(m.gpa, 64 + initialized);
        initialized += 1;
    }
    const ids = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373, 825, 264, 15352, 4829, 9944, 13, 42, 264, 271, 279, 15352, 854, 13, 4829, 825, 22373, 9944, 264, 271, 42, 13, 279 };
    for (&caches, 0..) |*cache, i| {
        const rows: usize = if (i % 2 == 0) 1 else 3;
        try fwd.run(m, wide, &.{.{ .cache = cache, .rows = rows }}, ids[0..rows], false, .last);
    }
    try fwd.run(m, wide, &.{ .{ .cache = &caches[0], .rows = 16 }, .{ .cache = &caches[1], .rows = 16 } }, &ids, true, .all);
    for (0..2) |stream| {
        for (0..16) |r| {
            try fwd.run(m, &solo, &.{.{ .cache = &caches[stream + 2], .rows = 1 }}, ids[stream * 16 + r ..][0..1], false, .all);
            const actual = wide.logits[(stream * 16 + r) * c.vocab ..][0..c.vocab];
            if (!std.mem.eql(u16, actual, solo.logits[0..c.vocab])) return error.QwenWindowLogitsMismatch;
        }
        try equal(&caches[stream], &caches[stream + 2]);
    }
    try caches[0].keep(wide, &.{ 0, 1, 2 });
    try caches[1].keep(wide, &.{0});
    for (0..2) |stream| {
        var fresh = try st.Cache.init(m.gpa, 64);
        defer fresh.deinit();
        const prefix: usize = if (stream == 0) 1 else 3;
        const kept: usize = if (stream == 0) 3 else 1;
        for (0..prefix) |r| try fwd.run(m, &solo, &.{.{ .cache = &fresh, .rows = 1 }}, ids[r..][0..1], false, .last);
        for (0..kept) |r| try fwd.run(m, &solo, &.{.{ .cache = &fresh, .rows = 1 }}, ids[stream * 16 + r ..][0..1], false, .all);
        try equal(&caches[stream], &fresh);
    }
}

/// A prompt in chunks of 1, 7 and 128 commits the same state and logits as one pass.
pub fn chunkCuts(m: *const Model, wide: *st.Scratch, ids: []const u32) !void {
    var whole = try st.Cache.init(m.gpa, ids.len + 1);
    defer whole.deinit();
    try fwd.run(m, wide, &.{.{ .cache = &whole, .rows = ids.len }}, ids, false, .last);
    for ([_]usize{ 1, 7 }) |chunk| {
        var cut = try st.Cache.init(m.gpa, ids.len + 1);
        defer cut.deinit();
        var at: usize = 0;
        while (at < ids.len) : (at += chunk) {
            const end = @min(ids.len, at + chunk);
            try fwd.run(m, wide, &.{.{ .cache = &cut, .rows = end - at }}, ids[at..end], false, .last);
        }
        try equal(&whole, &cut);
    }
}

test "synthetic weights: windows, kept prefixes and chunk cuts reproduce one-row steps bit for bit" {
    const gpa = std.testing.allocator;
    const m = try Model.synthetic(gpa, std.testing.io, 11, null);
    defer m.deinit();
    var wide = try st.Scratch.init(gpa, 128, 64, m.pool.workers());
    defer wide.deinit();
    try check(m, &wide);
    try chunkCuts(m, &wide, &.{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373, 825, 264, 15352, 4829, 9944, 13, 42, 264, 271, 279, 15352, 854, 13 });
}

test "synthetic weights: the committed states and logits are the same bytes at 1 to 8 threads" {
    const gpa = std.testing.allocator;
    const ids = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373, 825, 264, 15352, 4829, 9944, 13, 42 };
    var one: ?st.Cache = null;
    defer if (one) |*cache| cache.deinit();
    for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 }) |workers| {
        const m = try Model.synthetic(gpa, std.testing.io, 11, workers);
        defer m.deinit();
        var scratch = try st.Scratch.init(gpa, 32, 64, m.pool.workers());
        defer scratch.deinit();
        var cache = try st.Cache.init(gpa, 64);
        errdefer cache.deinit();
        try fwd.run(m, &scratch, &.{.{ .cache = &cache, .rows = ids.len - 1 }}, ids[0 .. ids.len - 1], false, .last);
        try fwd.run(m, &scratch, &.{.{ .cache = &cache, .rows = 1 }}, ids[ids.len - 1 ..], false, .all);
        if (one) |*first| {
            try equal(first, &cache);
            cache.deinit();
        } else one = cache;
    }
}
