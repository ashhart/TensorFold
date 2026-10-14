//! Real-weight LFM2 prompt chunks, windows, partial keeps and unequal streams against native one-row execution.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm2;
const st = q.state;
const fwd = q.forward;
const c = q.config;

fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    try std.testing.expectEqual(a.len, b.len);
    try std.testing.expectEqualSlices(u8, a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2]);
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .conv => |s| try std.testing.expectEqualSlices(u8, s.contents()[0..st.conv_bytes], y.conv.contents()[0..st.conv_bytes]),
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
        const rows: usize = @min(scratch.rows, ids.len - at);
        try fwd.run(m, scratch, &.{.{ .cache = cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
}

fn window(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32, prefix: usize, rows: usize, keep: usize) !void {
    var a = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.attention, 529);
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
    var accepted = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
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
        cache.* = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
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
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer fresh.deinit();
        try prompt(m, solo, &fresh, ids[0..length]);
        try equal(&caches[stream], &fresh);
    }
}

/// One window shared by streams at `lens` (0 rows skip one): its logits and caches equal one-row decode.
fn round(
    m: *q.Model,
    wide: *st.Scratch,
    solo: *st.Scratch,
    ids: []const u32,
    caches: []st.Cache,
    lens: []const usize,
    rows: []const usize,
) !void {
    var segs: [4]fwd.Segment = undefined;
    var tokens: [32]u32 = undefined;
    var n: usize = 0;
    var total: usize = 0;
    for (caches, lens, rows) |*cache, len, r| {
        if (r == 0) continue;
        segs[n] = .{ .cache = cache, .rows = r };
        @memcpy(tokens[total..][0..r], ids[len..][0..r]);
        n += 1;
        total += r;
    }
    try fwd.run(m, wide, segs[0..n], tokens[0..total], true, .all);
    var base: usize = 0;
    for (caches, lens, rows) |*cache, len, r| {
        if (r == 0) continue;
        var ref = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer ref.deinit();
        try prompt(m, solo, &ref, ids[0..len]);
        for (0..r) |i| {
            try fwd.run(m, solo, &.{.{ .cache = &ref, .rows = 1 }}, ids[len + i ..][0..1], false, .all);
            const actual = (wide.logits.contents() + (base + i) * c.vocab * 2)[0 .. c.vocab * 2];
            try std.testing.expectEqualSlices(u8, solo.logits.contents()[0 .. c.vocab * 2], actual);
        }
        try equal(cache, &ref);
        base += r;
    }
}

/// Four streams at unequal prefixes share 32 rows, keep unequal paths, then share 16 rows (the by-expert width).
fn shared4(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var caches: [4]st.Cache = undefined;
    var n: usize = 0;
    defer for (caches[0..n]) |*cache| cache.deinit();
    var lens = [_]usize{ 1, 127, 128, 256 };
    for (&caches, lens) |*cache, len| {
        cache.* = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        n += 1;
        try prompt(m, solo, cache, ids[0..len]);
    }
    try round(m, wide, solo, ids, &caches, &lens, &.{ 1, 5, 10, 16 });
    const keep = [_]usize{ 1, 3, 10, 7 };
    var path: [16]u32 = undefined;
    for (&path, 0..) |*r, i| r.* = @intCast(i);
    for (&caches, &lens, keep) |*cache, *len, kept| {
        try cache.keep(wide, path[0..kept]);
        len.* += kept;
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer fresh.deinit();
        try prompt(m, solo, &fresh, ids[0..len.*]);
        try equal(cache, &fresh);
    }
    try round(m, wide, solo, ids, &caches, &lens, &.{ 3, 4, 9, 0 });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModel;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    var wide = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 32);
    defer wide.deinit();
    var solo = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 32);
    defer solo.deinit();
    var chunk = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 128);
    defer chunk.deinit();
    var one = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 1);
    defer one.deinit();
    var ids: [300]u32 = undefined;
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (&ids, 0..) |*id, i| id.* = public[i % public.len];
    var passed: usize = 0;
    for ([_]usize{ 1, 127, 128, 129, 300 }) |length| {
        var a = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer a.deinit();
        var b = try st.Cache.init(m.gpa, m.device, &m.config.attention, 300);
        defer b.deinit();
        try prompt(m, &chunk, &a, ids[0..length]);
        try prompt(m, &one, &b, ids[0..length]);
        try equal(&a, &b);
        passed += 1;
        std.debug.print("ok prompt {d} in 128-row chunks equals one row at a time\n", .{length});
    }
    for ([_]usize{ 0, 127, 128, 129, 255, 256 }) |prefix| {
        for ([_]usize{ 1, 2, 7, 8, 9, 15, 16, 17, 24, 32 }) |rows| {
            try window(m, &wide, &solo, &ids, prefix, rows, @max(1, rows / 2));
            passed += 1;
            std.debug.print("ok prefix {d}, window {d}, partial keep {d}\n", .{ prefix, rows, @max(1, rows / 2) });
        }
    }
    try shared(m, &wide, &solo, &ids);
    passed += 1;
    try shared4(m, &wide, &solo, &ids);
    passed += 1;
    std.debug.print("ok four streams at prefixes 1, 127, 128, 256: windows of 32 and 16 rows, unequal keeps\n", .{});
    std.debug.print("{d} forward/cache checks passed, 0 failed, 0 skipped\n", .{passed});
}
