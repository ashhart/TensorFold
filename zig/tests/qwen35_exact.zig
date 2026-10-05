//! Real-dimension Qwen forward, cache, partial-keep and unequal-stream contracts against native one-row execution.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen35;
const st = q.state;
const fwd = q.forward;
const c = q.config;

fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    try std.testing.expectEqual(a.len, b.len);
    try std.testing.expectEqualSlices(u8, a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2]);
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .delta => |d| {
            try std.testing.expectEqualSlices(u8, d.recurrence.contents()[0..st.delta_bytes], y.delta.recurrence.contents()[0..st.delta_bytes]);
            try std.testing.expectEqualSlices(u8, d.conv.contents()[0..st.conv_bytes], y.delta.conv.contents()[0..st.conv_bytes]);
        },
        .attention => |k| {
            for ([_]mtl.Buffer{ k.keys, k.values }, [_]mtl.Buffer{ y.attention.keys, y.attention.values }) |a_buf, b_buf| {
                for (0..c.kv_heads) |h| {
                    const ao = h * a.capacity * c.head_dim * 2;
                    const bo = h * b.capacity * c.head_dim * 2;
                    const bytes = a.len * c.head_dim * 2;
                    try std.testing.expectEqualSlices(u8, (a_buf.contents() + ao)[0..bytes], (b_buf.contents() + bo)[0..bytes]);
                }
            }
        },
    };
}

fn prompt(m: *q.Model, scratch: *st.Scratch, cache: *st.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const rows: usize = @min(32, ids.len - at);
        try fwd.run(m, scratch, &.{.{ .cache = cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
}

fn window(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32, prefix: usize, rows: usize, keep: usize) !void {
    var a = try st.Cache.init(m.gpa, m.device, 512);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, 529);
    defer b.deinit();
    try prompt(m, wide, &a, ids[0..prefix]);
    try prompt(m, solo, &b, ids[0..prefix]);
    try equal(&a, &b);
    try fwd.run(m, wide, &.{.{ .cache = &a, .rows = rows }}, ids[prefix..][0..rows], true, .all);
    for (0..rows) |r| {
        try fwd.run(m, solo, &.{.{ .cache = &b, .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
        const want = solo.logits.contents()[0 .. c.vocab * 2];
        const actual = (wide.logits.contents() + r * c.vocab * 2)[0 .. c.vocab * 2];
        try std.testing.expectEqualSlices(u8, want, actual);
    }
    try equal(&a, &b);
    var path: [32]u32 = undefined;
    for (&path, 0..) |*r, i| r.* = @intCast(i);
    try a.keep(wide, path[0..keep]);
    var accepted = try st.Cache.init(m.gpa, m.device, 512);
    defer accepted.deinit();
    try prompt(m, solo, &accepted, ids[0..prefix]);
    for (0..keep) |r| try fwd.run(m, solo, &.{.{ .cache = &accepted, .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
    try equal(&a, &accepted);
    try fwd.run(m, wide, &.{.{ .cache = &a, .rows = 1 }}, ids[prefix + keep ..][0..1], false, .all);
    try fwd.run(m, solo, &.{.{ .cache = &accepted, .rows = 1 }}, ids[prefix + keep ..][0..1], false, .all);
    try equal(&a, &accepted);
    try std.testing.expectEqualSlices(u8, a.logits.contents()[0 .. c.vocab * 2], accepted.logits.contents()[0 .. c.vocab * 2]);
}

fn shared(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var caches: [4]st.Cache = undefined;
    var n: usize = 0;
    defer for (caches[0..n]) |*cache| cache.deinit();
    for (&caches) |*cache| {
        cache.* = try st.Cache.init(m.gpa, m.device, 512);
        n += 1;
    }
    for (&caches, 0..) |*cache, i| try prompt(m, solo, cache, ids[0..if (i % 2 == 0) @as(usize, 127) else 256]);
    var tokens: [20]u32 = undefined;
    @memcpy(tokens[0..7], ids[127..][0..7]);
    @memcpy(tokens[7..20], ids[256..][0..13]);
    try fwd.run(m, wide, &.{ .{ .cache = &caches[0], .rows = 7 }, .{ .cache = &caches[1], .rows = 13 } }, &tokens, true, .all);
    for ([_]usize{ 7, 13 }, 0..) |rows, stream| {
        const prefix: usize = if (stream == 0) 127 else 256;
        const base: usize = if (stream == 0) 0 else 7;
        for (0..rows) |r| {
            try fwd.run(m, solo, &.{.{ .cache = &caches[stream + 2], .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
            try std.testing.expectEqualSlices(u8, solo.logits.contents()[0 .. c.vocab * 2], (wide.logits.contents() + (base + r) * c.vocab * 2)[0 .. c.vocab * 2]);
        }
        try equal(&caches[stream], &caches[stream + 2]);
    }
    try caches[0].keep(wide, &.{ 0, 1, 2 });
    try caches[1].keep(wide, &.{0});
    for ([_]usize{ 130, 257 }, 0..) |length, stream| {
        var fresh = try st.Cache.init(m.gpa, m.device, 512);
        defer fresh.deinit();
        try prompt(m, solo, &fresh, ids[0..length]);
        try equal(&caches[stream], &fresh);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModel;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    var wide = try st.Scratch.init(init.gpa, m.device, 32, 529);
    defer wide.deinit();
    var solo = try st.Scratch.init(init.gpa, m.device, 32, 529);
    defer solo.deinit();
    var ids: [300]u32 = undefined;
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (&ids, 0..) |*id, i| id.* = public[i % public.len];
    var passed: usize = 0;
    for ([_]usize{ 0, 127, 128, 129, 255, 256 }) |prefix| {
        for ([_]usize{ 1, 2, 7, 8, 15, 16, 17, 32 }) |rows| {
            try window(m, &wide, &solo, &ids, prefix, rows, @max(1, rows / 2));
            passed += 1;
            std.debug.print("ok prefix {d}, window {d}, partial keep {d}\n", .{ prefix, rows, @max(1, rows / 2) });
        }
    }
    try shared(m, &wide, &solo, &ids);
    passed += 1;
    std.debug.print("{d} forward/cache checks passed, 0 failed, 0 skipped\n", .{passed});
}
