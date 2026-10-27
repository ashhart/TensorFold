//! The real learned-state and row-file readers must publish into both snapshot containers or neither.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rows = @import("rows_store.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const RowFake = @import("rows_fake.zig").Fake;
const oracle = @import("prompt_cache_index_oracle.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const a = std.testing.allocator;
const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const harness = [_]u32{ 7, 7, 7, 7, 7, 7 };
const second = harness ++ [_]u32{ 3, 4, 9 };

fn root(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-index-{s}-{d}", .{ tag, platform.c.getpid() });
}

fn seedLearned(dir: []const u8, scratch: std.mem.Allocator) !void {
    var im = try imprint.Imprint.open(a, dir, 13, 1 << 30);
    defer im.deinit();
    im.admission.floor = 1 << 20;
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const first = harness ++ [_]u32{ 1, 2, 9 };
    const plan = try s.begin(scratch, &first, 8, &.{5}, &.{}, null, &.{});
    try std.testing.expectEqual(@import("prompt_cache_test.zig").fresh(&first), f.pass(&s, &first, plan));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&harness)));
    try oracle.check(&s);
}

test "snapshot index: learned restart publishes an indexed state and preserves the old winner" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = try root(&buf, "learned");
    defer rmTree(dir);
    try seedLearned(dir, arena.allocator());
    var im = try imprint.Imprint.open(a, dir, 13, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.learned(), rules, 1 << 20);
    defer s.deinit();
    const other = [_]u32{ 1, 2, 3, 4 };
    try @import("prompt_cache_index_test.zig").keep(&s, &f, &other, 2, &.{});
    s.imprint = &im;
    const plan = try s.begin(arena.allocator(), &second, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 5), plan.from);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 2), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u32, 5), s.entries.items[1].at);
    try std.testing.expect(oracle.find(&s, &second, &.{}, &.{}) == s.find(&second, &.{}, &.{}));
    try std.testing.expectEqual(@import("prompt_cache_test.zig").fresh(&second), f.pass(&s, &second, plan));
    try oracle.check(&s);
}

test "snapshot index: learned recall unwinds metadata and either publication allocation" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = try root(&buf, "learned-fault");
    defer rmTree(dir);
    try seedLearned(dir, arena.allocator());
    var im = try imprint.Imprint.open(a, dir, 13, 1 << 30);
    defer im.deinit();
    var failures: usize = 0;
    var succeeded = false;
    for (0..8) |fail_at| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at, .resize_fail_index = 0 });
        s.gpa = failing.allocator();
        const looked = s.lookup(arena.allocator(), &second, 0, &.{}, &.{}, &.{});
        s.gpa = a;
        const result = try looked;
        try oracle.check(&s);
        try std.testing.expectEqual(s.entries.items.len, f.live);
        try std.testing.expect(im.has(imprint.Imprint.keyOf(&harness)));
        if (failing.has_induced_failure) {
            failures += 1;
            try std.testing.expect(result.entry == null);
            try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
            const retry = try s.lookup(arena.allocator(), &second, 0, &.{}, &.{}, &.{});
            try std.testing.expectEqual(@as(u32, 5), retry.entry.?.at);
            s.release(retry.entry.?);
            try oracle.check(&s);
        } else {
            succeeded = true;
            try std.testing.expectEqual(@as(u32, 5), result.entry.?.at);
            s.release(result.entry.?);
            break;
        }
    }
    try std.testing.expect(succeeded);
    try std.testing.expectEqual(@as(usize, 5), failures);
}

fn options(dir: []const u8) rows.Options {
    return .{ .root = dir, .identity = 17, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
}

fn rowPrompt(out: []u32) void {
    for (out, 0..) |*token, i| token.* = @intCast(1000 + i * 7 % 997);
}

fn seedRows(dir: []const u8, scratch: std.mem.Allocator, prompt: []const u32) !void {
    var f: RowFake = .{ .gpa = a, .prompt = prompt, .spans = &.{.{ 500, 600 }} };
    const r = try rows.Rows.open(a, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const plan = try s.begin(scratch, prompt, 1400, &.{}, &.{}, null, f.spans);
    try std.testing.expectEqual(@import("rows_fake.zig").fresh(prompt), f.pass(&s, plan));
    while (rows.write.step(r)) {}
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try oracle.check(&s);
}

test "snapshot index: row-file restart publishes exact spans and a byte-equal resumed image" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = try root(&buf, "rows");
    defer rmTree(dir);
    var prompt: [1500]u32 = undefined;
    rowPrompt(&prompt);
    try seedRows(dir, arena.allocator(), &prompt);
    prompt[1499] = 987;
    var f: RowFake = .{ .gpa = a, .prompt = &prompt, .spans = &.{.{ 500, 600 }} };
    const r = try rows.Rows.open(a, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    f.at = 200;
    try std.testing.expect(s.keep(&prompt, 200, null, &.{}, f.spans));
    s.disk = r;
    const plan = try s.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, f.spans);
    try std.testing.expectEqual(@as(u32, 1400), plan.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(@import("rows_fake.zig").fresh(&prompt), f.pass(&s, plan));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 200), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u32, 1400), s.entries.items[1].at);
    try std.testing.expectEqualSlices([2]u32, f.spans, s.entries.items[1].decode_spans);
    try std.testing.expect(oracle.find(&s, &prompt, &.{}, f.spans) == s.find(&prompt, &.{}, f.spans));
    try std.testing.expectEqual(@as(u64, 1), r.counts.reads);
    try oracle.check(&s);
}

test "snapshot index: row recall unwinds metadata spans and either publication allocation" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var buf: [128]u8 = undefined;
    const dir = try root(&buf, "rows-fault");
    defer rmTree(dir);
    var prompt: [1500]u32 = undefined;
    rowPrompt(&prompt);
    try seedRows(dir, arena.allocator(), &prompt);
    var failures: usize = 0;
    var succeeded = false;
    for (0..9) |fail_at| {
        var f: RowFake = .{ .gpa = a, .prompt = &prompt, .spans = &.{.{ 500, 600 }} };
        const r = try rows.Rows.open(a, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at, .resize_fail_index = 0 });
        s.gpa = failing.allocator();
        const looked = s.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, f.spans);
        s.gpa = a;
        const result = try looked;
        try oracle.check(&s);
        try std.testing.expectEqual(s.entries.items.len, f.live);
        try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
        if (failing.has_induced_failure) {
            failures += 1;
            try std.testing.expect(result.entry == null);
            try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
            const retry = try s.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, f.spans);
            try std.testing.expectEqual(@as(u32, 1400), retry.entry.?.at);
            s.release(retry.entry.?);
            try oracle.check(&s);
        } else {
            succeeded = true;
            try std.testing.expectEqual(@as(u32, 1400), result.entry.?.at);
            s.release(result.entry.?);
            break;
        }
    }
    try std.testing.expect(succeeded);
    try std.testing.expectEqual(@as(usize, 6), failures);
}
