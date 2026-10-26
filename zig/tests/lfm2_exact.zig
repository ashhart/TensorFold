//! Real-weight LFM2 prompt chunks, windows, partial keeps and unequal streams against native one-row execution.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm2_moe;
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

/// Each of a forward's `rows` head rows: the GPU's greedy pick equals the host's draw on its logits.
fn picks(m: *q.Model, s: *st.Scratch, rows: usize) !void {
    const best = s.best.slice(u32, 2 * rows);
    for (0..rows) |r| {
        const want = try q.backend.draw(m.gpa, s.logits.slice(u16, rows * c.vocab)[r * c.vocab ..][0..c.vocab], null, 0, &.{});
        try std.testing.expectEqual(want, best[2 * r]);
        try std.testing.expectEqual(@as(u32, 0), best[2 * r + 1]);
    }
}

/// A prompt of `prefix` rows, then plain decode to `len`: what a window's kept rows equal.
fn decoded(m: *q.Model, solo: *st.Scratch, cache: *st.Cache, ids: []const u32, prefix: usize, len: usize) !void {
    try prompt(m, solo, cache, ids[0..prefix]);
    for (prefix..len) |r| try fwd.run(m, solo, &.{.{ .cache = cache, .rows = 1 }}, ids[r..][0..1], false, .all);
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
    try picks(m, wide, rows);
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
    for ([_]usize{ 127, 256 }, [_]usize{ 130, 257 }, 0..) |prefix, length, stream| {
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer fresh.deinit();
        try decoded(m, solo, &fresh, ids, prefix, length);
        try equal(&caches[stream], &fresh);
    }
}

/// One window shared by streams at `lens`, prompted to `prompts` (0 rows skip one): it equals one-row decode.
fn round(
    m: *q.Model,
    wide: *st.Scratch,
    solo: *st.Scratch,
    ids: []const u32,
    caches: []st.Cache,
    prompts: []const usize,
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
    try picks(m, wide, total);
    var base: usize = 0;
    for (caches, prompts, lens, rows) |*cache, start, len, r| {
        if (r == 0) continue;
        var ref = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer ref.deinit();
        try decoded(m, solo, &ref, ids, start, len);
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
    const prompts = [_]usize{ 1, 127, 128, 256 };
    var lens = prompts;
    for (&caches, lens) |*cache, len| {
        cache.* = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        n += 1;
        try prompt(m, solo, cache, ids[0..len]);
    }
    try round(m, wide, solo, ids, &caches, &prompts, &lens, &.{ 1, 5, 10, 16 });
    const keep = [_]usize{ 1, 3, 10, 7 };
    var path: [16]u32 = undefined;
    for (&path, 0..) |*r, i| r.* = @intCast(i);
    for (&caches, prompts, &lens, keep) |*cache, start, *len, kept| {
        try cache.keep(wide, path[0..kept]);
        len.* += kept;
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer fresh.deinit();
        try decoded(m, solo, &fresh, ids, start, len.*);
        try equal(cache, &fresh);
    }
    try round(m, wide, solo, ids, &caches, &prompts, &lens, &.{ 3, 4, 9, 0 });
}

/// A prompt read in 4,096-row chunks (attention in two slices) equals the same prompt in 2,048-row chunks.
fn chunks(m: *q.Model, gpa: std.mem.Allocator) !void {
    const length = 4500;
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    var a = try st.Cache.init(m.gpa, m.device, &m.config.attention, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.attention, length + 64);
    defer b.deinit();
    for ([_]usize{ 4096, 2048 }, 0..) |rows, i| {
        var scratch = try st.Scratch.init(gpa, m.device, &m.config.attention, rows);
        defer scratch.deinit();
        try prompt(m, &scratch, if (i == 0) &a else &b, ids);
    }
    try equal(&a, &b);
}

/// A prompt in the served 2,048-row chunks equals it one row at a time, across attention slices and 4,096-key cuts.
fn seam(m: *q.Model, gpa: std.mem.Allocator, length: usize) !void {
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    var a = try st.Cache.init(m.gpa, m.device, &m.config.attention, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.attention, length + 64);
    defer b.deinit();
    var served = try st.Scratch.init(gpa, m.device, &m.config.attention, 2048);
    defer served.deinit();
    var one = try st.Scratch.init(gpa, m.device, &m.config.attention, 1);
    defer one.deinit();
    try prompt(m, &served, &a, ids);
    try prompt(m, &one, &b, ids);
    try equal(&a, &b);
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
    // A prompt cut anywhere equals it whole: one-row prompt calls are the reference (the tensor units' on an M5).
    var half = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 1);
    defer half.deinit();
    const ref_name = "one row at a time";
    var ids: [300]u32 = undefined;
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (&ids, 0..) |*id, i| id.* = public[i % public.len];
    var passed: usize = 0;
    var failed: usize = 0;
    for ([_]usize{ 1, 127, 128, 129, 300 }) |length| {
        var a = try st.Cache.init(m.gpa, m.device, &m.config.attention, 512);
        defer a.deinit();
        var b = try st.Cache.init(m.gpa, m.device, &m.config.attention, 300);
        defer b.deinit();
        try prompt(m, &chunk, &a, ids[0..length]);
        try prompt(m, &half, &b, ids[0..length]);
        if (equal(&a, &b)) |_| {
            passed += 1;
            std.debug.print("ok prompt {d} in 128-row chunks equals {s}\n", .{ length, ref_name });
        } else |_| {
            failed += 1;
            std.debug.print("FAIL prompt {d} in 128-row chunks differs from {s}\n", .{ length, ref_name });
        }
    }
    for ([_]usize{ 0, 127, 128, 129, 255, 256 }) |prefix| {
        for ([_]usize{ 1, 2, 7, 8, 9, 15, 16, 17, 24, 32 }) |rows| {
            if (window(m, &wide, &solo, &ids, prefix, rows, @max(1, rows / 2))) |_| {
                passed += 1;
                std.debug.print("ok prefix {d}, window {d}, partial keep {d}\n", .{ prefix, rows, @max(1, rows / 2) });
            } else |_| {
                failed += 1;
                const keep = @max(1, rows / 2);
                std.debug.print("FAIL prefix {d}, window {d}, partial keep {d}\n", .{ prefix, rows, keep });
            }
        }
    }
    if (chunks(m, init.gpa)) |_| {
        passed += 1;
        std.debug.print("ok prompt 4500 in 4096-row chunks equals 2048-row chunks\n", .{});
    } else |_| {
        failed += 1;
        std.debug.print("FAIL prompt 4500 in 4096-row chunks differs from 2048-row chunks\n", .{});
    }
    for ([_]usize{ 2047, 2049, 4097, 8193 }) |length| {
        if (seam(m, init.gpa, length)) |_| {
            passed += 1;
            std.debug.print("ok prompt {d} in 2048-row chunks equals one row at a time\n", .{length});
        } else |_| {
            failed += 1;
            std.debug.print("FAIL prompt {d} in 2048-row chunks differs from one row at a time\n", .{length});
        }
    }
    if (shared(m, &wide, &solo, &ids)) |_| {
        passed += 1;
    } else |_| {
        failed += 1;
        std.debug.print("FAIL two streams of 7 and 13 rows against each alone\n", .{});
    }
    if (shared4(m, &wide, &solo, &ids)) |_| {
        passed += 1;
        const line = "ok four streams at prefixes 1, 127, 128, 256: windows of 32 and 16 rows, unequal keeps\n";
        std.debug.print(line, .{});
    } else |_| {
        failed += 1;
        std.debug.print("FAIL four streams at prefixes 1, 127, 128, 256\n", .{});
    }
    std.debug.print("{d} forward/cache checks passed, {d} failed, 0 skipped\n", .{ passed, failed });
    if (failed != 0) return error.ExactnessFailed;
}
