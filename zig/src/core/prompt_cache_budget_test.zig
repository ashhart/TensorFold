//! Affordable cold checkpoints preserve the baseline chunk arithmetic and every snapshot identity.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const t = std.testing;
const rules: pc.Rules = .{ .planned = true, .grid = 2048, .chunk = 2048, .chunk_starts = false, .lookahead = 1, .min_prompt = 0 };

fn bytes(_: *anyopaque, at: u32) u64 {
    return at;
}

fn words(a: std.mem.Allocator, n: usize) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*x, i| x.* = @intCast(i % 101);
    return out;
}

// The backend owns 2048-row cold calls; ending one at another position changes all subsequent state bytes.
fn pass(f: *Fake, s: ?*pc.Store, ids: []const u32, p: pc.Plan) ![2]u64 {
    if (p.from == 0) f.at, f.sum = .{ 0, 0 };
    while (f.at < ids.len) {
        const start = f.at;
        var end = @min(@as(u32, @intCast(ids.len)), start + 2048);
        for (p.marks) |at| if (at > start and at < end) {
            end = at;
        };
        for (ids[start..end]) |id| f.sum = f.sum *% 31 +% id;
        f.sum ^= @as(u64, end - start) *% 0x9e3779b97f4a7c15;
        f.at = end;
        if (s) |store| if (std.mem.indexOfScalar(u32, p.marks, end) != null) try t.expect(store.keep(ids, end, null, &.{}, &.{}));
    }
    return .{ f.at, f.sum };
}

test "cache budget fallback: undersized symbolic budget retains the greatest earlier cut exactly" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    var f: Fake = .{ .gpa = t.allocator };
    var vtable = f.snapshots().vtable.*;
    vtable.bytes = bytes;
    var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 8192);
    defer s.deinit();
    const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
    try t.expectEqualSlices(u32, &.{8192}, p.marks);
    var cold: Fake = .{ .gpa = t.allocator };
    const expected = try pass(&cold, null, ids, .{});
    const saved = try pass(&f, &s, ids, p);
    try t.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&saved));
    const q = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
    try t.expectEqual(@as(u32, 8192), q.from);
    const resumed = try pass(&f, &s, ids, q);
    try t.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&resumed));
    // Planted corrupt restore: the full-state comparison must detect it.
    _ = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
    f.sum ^= 1;
    const bad = try pass(&f, null, ids, q);
    try t.expect(!std.mem.eql(u8, std.mem.asBytes(&expected), std.mem.asBytes(&bad)));
    // Planted arithmetic change: an extra 8191-row cut must also fail the state oracle.
    const split = try pass(&cold, null, ids, .{ .marks = &.{8191} });
    try t.expect(!std.mem.eql(u8, std.mem.asBytes(&expected), std.mem.asBytes(&split)));
    try t.expect(s.held + s.spare() <= s.budget);
}

test "cache budget fallback: nothing affordable stays empty and an affordable endpoint is unchanged" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    for ([_]u64{ 0, 2047, 2048, 12287, 12288, 1 << 20 }) |budget| {
        var f: Fake = .{ .gpa = t.allocator };
        var vtable = f.snapshots().vtable.*;
        vtable.bytes = bytes;
        var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, budget);
        defer s.deinit();
        const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
        const at: u32 = @intCast(@min(budget - budget % 2048, 12288));
        if (at == 0) try t.expectEqual(@as(usize, 0), p.marks.len) else try t.expectEqualSlices(u32, &.{at}, p.marks);
    }
}

test "cache budget fallback: non-GLM chunked family uses an existing cold cut" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    var f: Fake = .{ .gpa = t.allocator };
    var vtable = f.snapshots().vtable.*;
    vtable.bytes = bytes;
    var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, .{ .chunk = 2048, .chunk_starts = false, .min_prompt = 0 }, 8192);
    defer s.deinit();
    const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
    try t.expectEqualSlices(u32, &.{8192}, p.marks);
    var cold: Fake = .{ .gpa = t.allocator };
    const expected = try pass(&cold, null, ids, .{});
    const got = try pass(&f, &s, ids, p);
    try t.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&got));
}

fn none(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}
fn all(_: *anyopaque, _: *const pc.Entry) bool {
    return true;
}
fn spare(ptr: *anyopaque) u64 {
    return @as(*Fake, @ptrCast(@alignCast(ptr))).spare_bytes;
}

test "cache budget fallback: borrowers, protected states, promises, slots and eviction fund only real room" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    const foreign = try a.dupe(u32, ids);
    foreign[0] += 1;
    for (0..7) |mode| {
        var f: Fake = .{ .gpa = t.allocator };
        var vtable = f.snapshots().vtable.*;
        vtable.bytes = bytes;
        var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 8192);
        defer s.deinit();
        f.at = 2048;
        try t.expect(s.keep(foreign, 2048, null, &.{}, &.{}));
        const held = s.entries.items[0];
        if (mode == 0 or mode == 4) held.borrowers = 1;
        if (mode != 0) s.reservation_guard = .{ .ctx = &f, .needed = if (mode == 1 or mode == 2) all else none };
        if (mode == 2) s.reservation_guard.?.promised = .{ .bytes = 4096 };
        if (mode == 3) s.reservation_guard.?.promised = .{ .bytes = 10000 };
        if (mode == 4) s.most = 1;
        if (mode == 5) {
            s.most = 2;
            s.reservation_guard.?.promised = .{ .slots = 2 };
        }
        var promised: @import("prompt_cache_reserve.zig").Room = .{};
        if (mode == 6) s.reservation_guard.?.promise = &promised;
        const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
        const at: u32 = switch (mode) {
            0, 1 => 6144,
            2 => 2048,
            6 => 8192,
            else => 0,
        };
        if (at == 0) try t.expectEqual(@as(usize, 0), p.marks.len) else try t.expectEqualSlices(u32, &.{at}, p.marks);
        try t.expectEqual(@as(u64, if (mode == 6) 1 else 0), s.counts.evicted);
        if (mode == 6) {
            try t.expectEqual(@as(u64, 8192), promised.bytes);
            try t.expectEqual(@as(usize, 1), promised.slots);
        }
        _ = try pass(&f, &s, ids, p);
        try t.expect(s.held + s.spare() <= s.budget);
        if (mode == 0 or mode == 4) s.release(held);
    }
}

test "cache budget fallback: existing admitted marks and fixed spare storage keep their room" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    for ([_]bool{ false, true }) |fixed_spare| {
        var f: Fake = .{ .gpa = t.allocator, .spare_bytes = if (fixed_spare) 4096 else 0 };
        var vtable = f.snapshots().vtable.*;
        vtable.bytes = bytes;
        if (fixed_spare) vtable.spare = spare;
        var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 8192);
        defer s.deinit();
        const p = try s.beginUser(a, ids, 12705, if (fixed_spare) 0 else 2049, 0, &.{}, &.{}, null, &.{});
        try t.expectEqualSlices(u32, if (fixed_spare) &.{4096} else &.{ 2048, 6144 }, p.marks);
        _ = try pass(&f, &s, ids, p);
        try t.expect(s.held + s.spare() <= s.budget);
    }
}

test "cache budget fallback: supplied cuts, decoded edges, tap banks and unknown geometry" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    const cases = [_]struct { r: pc.Rules, starts: []const u32 = &.{}, spans: []const @import("cache_modes.zig").Span = &.{}, at: u32 }{
        .{ .r = .{ .chunk = 1024, .min_prompt = 0 }, .starts = &.{ 1500, 3500 }, .spans = &.{.{ 4000, 4700 }}, .at = 7772 },
        .{ .r = .{ .chunk = 1024, .chunk_starts = false, .min_prompt = 0 }, .starts = &.{ 1500, 3500 }, .spans = &.{.{ 4000, 4700 }}, .at = 8192 },
        .{ .r = rules, .starts = &.{ 1500, 3500, 4700, 9000, 12000 }, .at = 4700 },
        .{ .r = .{ .chunk = 1024, .chunk_marks = false, .min_prompt = 0 }, .starts = &.{ 1500, 3500, 4700, 9000 }, .at = 4700 },
        .{ .r = .{ .chunk = 1024, .chunk_marks = false, .min_prompt = 0 }, .at = 0 },
        .{ .r = .{ .min_prompt = 0 }, .at = 0 },
    };
    for (cases) |case| {
        var f: Fake = .{ .gpa = t.allocator };
        var vtable = f.snapshots().vtable.*;
        vtable.bytes = bytes;
        var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, case.r, 8192);
        defer s.deinit();
        const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, case.starts, null, case.spans);
        if (case.at == 0) try t.expectEqual(@as(usize, 0), p.marks.len) else try t.expectEqualSlices(u32, &.{case.at}, p.marks);
    }
}

fn uneven(_: *anyopaque, at: u32) u64 {
    return if (at == 6144) 8000 else 20000;
}

test "cache budget fallback: state sizes need not be monotone and holds or absent slots keep nothing" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    for (0..3) |mode| {
        var f: Fake = .{ .gpa = t.allocator };
        var vtable = f.snapshots().vtable.*;
        vtable.bytes = uneven;
        var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 8192);
        defer s.deinit();
        if (mode == 1) s.hold = true;
        if (mode == 2) s.most = 0;
        const p = try s.beginUser(a, ids, 12705, 0, 0, &.{}, &.{}, null, &.{});
        if (mode == 0) try t.expectEqualSlices(u32, &.{6144}, p.marks) else try t.expectEqual(@as(usize, 0), p.marks.len);
    }
}

test "cache budget fallback: lookahead, decoded spans, image identities, frames and salt still key the state" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try words(a, 12719);
    var f: Fake = .{ .gpa = t.allocator };
    var vtable = f.snapshots().vtable.*;
    vtable.bytes = bytes;
    var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 8192);
    defer s.deinit();
    const Image = @import("lanes").stream.Image;
    const image: Image = .{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 512, .tokens = 1, .hash = 7 };
    const spans: []const @import("cache_modes.zig").Span = &.{.{ 100, 200 }};
    const p = try s.lookupUserFrame(a, ids, @intCast(ids.len), 12705, 0, 0, &.{}, &.{}, spans, &.{image}, 42);
    try t.expectEqualSlices(u32, &.{8192}, p.marks);
    f.at = 8192;
    try t.expect(s.keepBornFrame(ids, @intCast(ids.len), 8192, null, &.{}, spans, &.{image}, 42));
    const entry = s.entries.items[0];
    entry.borrowers = 1; // miss probes cannot evict the state whose different identities they test
    defer s.release(entry);
    try t.expectEqual(@as(usize, 8193), entry.tokens.len);
    const good = try s.lookupUserFrame(a, ids, @intCast(ids.len), 12705, 0, 0, &.{}, &.{}, spans, &.{image}, 42);
    try t.expectEqual(entry, good.entry.?);
    s.release(good.entry.?);
    const edited = try a.dupe(u32, ids);
    edited[8192] += 1;
    for (0..5) |mode| {
        var changed_image = image;
        if (mode == 2) changed_image.hash += 1;
        if (mode == 4) s.salt = 1;
        const miss = try s.lookupUserFrame(a, if (mode == 0) edited else ids, @intCast(ids.len), 12705, 0, 0, &.{}, &.{}, if (mode == 1) &.{.{ 100, 201 }} else spans, &.{changed_image}, if (mode == 3) 43 else 42);
        try t.expect(miss.entry == null);
    }
}

test "cache budget fallback: evicting a longer old image cannot spend its shorter restore credit twice" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const old = try words(a, 30000);
    const ids = try a.dupe(u32, old[0..20000]);
    ids[3000] += 1;
    var f: Fake = .{ .gpa = t.allocator };
    var vtable = f.snapshots().vtable.*;
    vtable.bytes = bytes;
    var s = pc.Store.init(t.allocator, .{ .ptr = &f, .vtable = &vtable }, rules, 12288);
    defer s.deinit();
    for ([_]u32{ 2048, 4096 }) |at| {
        f.at = at;
        try t.expect(s.keepBorn(old, 30000, at, null, &.{}, &.{}));
    }
    var promise: @import("prompt_cache_reserve.zig").Room = .{};
    s.reservation_guard = .{ .ctx = &f, .needed = none, .promise = &promise };
    const p = try s.lookupUser(a, ids, @intCast(ids.len), 19986, 4097, 0, &.{}, &.{}, &.{});
    defer s.release(p.entry.?);
    try t.expectEqual(@as(u32, 2048), p.entry.?.at);
    try t.expectEqualSlices(u32, &.{ 4096, 6144 }, p.marks);
    try t.expectEqual(@as(u64, 10240), promise.bytes);
    try t.expectEqual(@as(u64, 1), s.counts.evicted);
}
