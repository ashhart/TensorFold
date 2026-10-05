//! Admit wider native lanes only after real-weight forward rows and committed caches match one-row execution on this GPU.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const fwd = @import("forward.zig");
const c = @import("config.zig");

pub fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    if (a.len != b.len) return error.QwenCacheLengthMismatch;
    if (!std.mem.eql(u8, a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2])) return error.QwenCommittedLogitsMismatch;
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .delta => |d| {
            if (!std.mem.eql(u8, d.recurrence.contents()[0..st.delta_bytes], y.delta.recurrence.contents()[0..st.delta_bytes]) or
                !std.mem.eql(u8, d.conv.contents()[0..st.conv_bytes], y.delta.conv.contents()[0..st.conv_bytes])) return error.QwenRecurrenceMismatch;
        },
        .attention => |kv| {
            for ([_]mtl.Buffer{ kv.keys, kv.values }, [_]mtl.Buffer{ y.attention.keys, y.attention.values }) |ab, bb| {
                for (0..c.kv_heads) |h| {
                    const ao = h * a.capacity * c.head_dim * 2;
                    const bo = h * b.capacity * c.head_dim * 2;
                    const bytes = a.len * c.head_dim * 2;
                    if (!std.mem.eql(u8, (ab.contents() + ao)[0..bytes], (bb.contents() + bo)[0..bytes])) return error.QwenAttentionCacheMismatch;
                }
            }
        },
    };
}

pub fn check(m: *Model, wide: *st.Scratch) !void {
    var solo = try st.Scratch.init(m.gpa, m.device, 1, 64);
    defer solo.deinit();
    var caches: [4]st.Cache = undefined;
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*cache| cache.deinit();
    for (&caches) |*cache| {
        cache.* = try st.Cache.init(m.gpa, m.device, 64 + initialized);
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
            const actual = (wide.logits.contents() + (stream * 16 + r) * c.vocab * 2)[0 .. c.vocab * 2];
            if (!std.mem.eql(u8, actual, solo.logits.contents()[0 .. c.vocab * 2])) return error.QwenWindowLogitsMismatch;
        }
        try equal(&caches[stream], &caches[stream + 2]);
    }
    try caches[0].keep(wide, &.{ 0, 1, 2 });
    try caches[1].keep(wide, &.{0});
    for (0..2) |stream| {
        var fresh = try st.Cache.init(m.gpa, m.device, 64);
        defer fresh.deinit();
        const prefix: usize = if (stream == 0) 1 else 3;
        const kept: usize = if (stream == 0) 3 else 1;
        for (0..prefix) |r| try fwd.run(m, &solo, &.{.{ .cache = &fresh, .rows = 1 }}, ids[r..][0..1], false, .last);
        for (0..kept) |r| try fwd.run(m, &solo, &.{.{ .cache = &fresh, .rows = 1 }}, ids[stream * 16 + r ..][0..1], false, .all);
        try equal(&caches[stream], &fresh);
    }
}
