//! The production Metal capability must route Flash Next prompt work through shared fill choices.
const std = @import("std");
const Backend = @import("backend.zig").Backend;

test "flashnext prefill: actual backend advertises completed prompt steps" {
    if (comptime @import("builtin").os.tag == .macos) {
        var b: Backend = undefined;
        const callbacks = @import("prefill.zig").Callbacks(Backend);
        try std.testing.expect(b.backend().vtable.prefill_step == callbacks.one);
        try std.testing.expect(b.backend().vtable.prefill == callbacks.whole);
    } else return error.SkipZigTest;
}

const lanes = @import("lanes");
const fill = @import("prefill.zig");
const recording = @import("prefill_test_engine.zig");
const Fake = recording.Fake;
const a = std.testing.allocator;

fn stream(tokens: []const u32, id: []const u8) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = id, .prompt = tokens, .max_new = 4 });
}

test "flashnext prefill: independent cursors retain restored prefixes, marks and prompt head state" {
    var f: Fake = .{ .restored_at = 3 };
    defer f.deinit();
    const tokens = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2 };
    var one = try stream(&tokens, "a");
    defer one.deinit(a);
    var two = try stream(&tokens, "b");
    defer two.deinit(a);
    one.chunks = &.{ 5, 9, 16 };
    two.chunks = &.{ 2, 7, 10 };
    const Marks = struct {
        values: [3]u32 = undefined,
        n: usize = 0,
        fn hit(ptr: *anyopaque, _: *lanes.Stream, at: u32) void {
            const m: *@This() = @ptrCast(@alignCast(ptr));
            m.values[m.n] = at;
            m.n += 1;
        }
    };
    var marks: Marks = .{};
    one.reuse.marks = &.{ 7, 16 };
    one.reuse.hook = .{ .ptr = &marks, .at = Marks.hit };
    try std.testing.expect(!try fill.step(&f, &one));
    try std.testing.expectEqual(@as(u32, 3), one.cached);
    try std.testing.expectEqual(@as(u32, 5), one.fill_at.load(.monotonic));
    f.restored_at = 0;
    try std.testing.expect(!try fill.step(&f, &two));
    try std.testing.expectEqual(@as(u32, 2), two.fill_at.load(.monotonic));
    var done_one = false;
    var done_two = false;
    while (!done_one or !done_two) {
        if (!done_one) done_one = try fill.step(&f, &one);
        if (!done_two) done_two = try fill.step(&f, &two);
    }
    try std.testing.expectEqualSlices(u32, &.{ 7, 16 }, marks.values[0..marks.n]);
    try std.testing.expectEqualDeep(recording.reference(&tokens), f.complete.get(&one).?);
    try std.testing.expectEqualDeep(f.complete.get(&one).?, f.complete.get(&two).?);
    try std.testing.expectEqual(@as(u32, 3), one.cached);
    try std.testing.expectEqual(@as(usize, 2), f.begins);
    try std.testing.expectEqual(f.owners, f.freed);
}

test "flashnext prefill: whole and stepped drivers retain P minus one heads at every odd seam" {
    var tokens: [257]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i % 97);
    for ([_]usize{ 1, 2, 3, 17, 31, 32, 33, 65, 257 }) |n| {
        var f: Fake = .{};
        defer f.deinit();
        var one = try stream(tokens[0..n], "whole");
        defer one.deinit(a);
        var two = try stream(tokens[0..n], "pieces");
        defer two.deinit(a);
        two.chunks = &.{ 1, 7, 17, 32, 64, 256 };
        try fill.Callbacks(Fake).whole(&f.target, &one);
        while (!try fill.Callbacks(Fake).one(&f.target, &two)) {}
        try std.testing.expectEqualDeep(f.complete.get(&one).?, f.complete.get(&two).?);
        try std.testing.expectEqual(n - 1, f.complete.get(&two).?.pairs);
        try std.testing.expectEqual(@as(u32, 0), two.cached);
        try std.testing.expectEqual(@as(u32, @intCast(n)), two.fill_at.load(.monotonic));
        try std.testing.expectEqual(f.owners, f.freed);
    }
}

test "flashnext prefill: failed chunks, completion and cache marks never publish later progress" {
    const tokens: [65]u32 = @splat(7);
    for (0..4) |fault| {
        var f: Fake = .{};
        defer f.deinit();
        var s = try stream(&tokens, "fault");
        defer s.deinit(a);
        s.chunks = &.{ 17, 33 };
        s.reuse.marks = &.{33};
        try std.testing.expect(!try fill.step(&f, &s));
        try std.testing.expectEqual(@as(u32, 17), s.fill_at.load(.monotonic));
        switch (fault) {
            0 => f.fail_chunk = true,
            1 => f.fail_completion = true,
            2 => f.fail_mark = true,
            else => f.drop_lookahead = true,
        }
        if (fault == 2) {
            try std.testing.expectError(error.MarkFailed, fill.step(&f, &s));
            try std.testing.expectEqual(@as(u32, 33), s.fill_at.load(.monotonic));
        } else if (fault == 3) {
            try std.testing.expectError(error.LostPromptLookahead, fill.step(&f, &s));
            try std.testing.expectEqual(@as(u32, 17), s.fill_at.load(.monotonic));
        } else {
            try std.testing.expectError(error.GpuFailed, fill.step(&f, &s));
            try std.testing.expectEqual(@as(u32, 17), s.fill_at.load(.monotonic));
        }
        try std.testing.expectEqual(@as(usize, 0), f.filling.count());
        try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
        try std.testing.expectEqual(f.owners, f.freed);
    }
}

test "flashnext prefill: a lane-memory retry and stale prefix do not reset another filling stream" {
    var f: Fake = .{ .max_slots = 1 };
    defer f.deinit();
    const tokens: [33]u32 = @splat(3);
    var first = try stream(&tokens, "first");
    defer first.deinit(a);
    first.chunks = &.{17};
    var second = try stream(&tokens, "second");
    defer second.deinit(a);
    try std.testing.expect(!try fill.step(&f, &first));
    try std.testing.expectError(error.LaneMemory, fill.step(&f, &second));
    try std.testing.expectEqual(@as(u32, 17), first.fill_at.load(.monotonic));
    while (!try fill.step(&f, &first)) {}
    f.abortPrompt(&first);
    f.restored_at = 99;
    while (!try fill.step(&f, &second)) {}
    try std.testing.expect(second.reuse_failed);
    try std.testing.expectEqual(@as(u32, 0), second.cached);
    try std.testing.expectEqualDeep(recording.reference(&tokens), f.complete.get(&second).?);
    try std.testing.expectEqual(f.owners, f.freed);
}

test "flashnext prefill: cancellation at every seam frees its owner without affecting another fill" {
    const Cancel = struct {
        stopped: bool = false,
        fn check(ptr: *anyopaque) bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            return c.stopped;
        }
    };
    const tokens: [33]u32 = @splat(5);
    for (0..4) |seam| {
        var f: Fake = .{};
        defer f.deinit();
        var c: Cancel = .{};
        var s = try stream(&tokens, "cancelled");
        defer s.deinit(a);
        s.chunks = &.{ 1, 7, 17, 32 };
        s.cancel_check = .{ .ptr = &c, .check = Cancel.check };
        var other = try stream(&tokens, "survivor");
        defer other.deinit(a);
        other.chunks = &.{17};
        try std.testing.expect(!try fill.step(&f, &other));
        for (0..seam) |_| try std.testing.expect(!try fill.step(&f, &s));
        const before = s.fill_at.load(.monotonic);
        c.stopped = true;
        try std.testing.expectError(error.Cancelled, fill.step(&f, &s));
        try std.testing.expectEqual(before, s.fill_at.load(.monotonic));
        try std.testing.expectEqual(@as(u32, 17), other.fill_at.load(.monotonic));
        try std.testing.expect(!f.filling.contains(&s));
        while (!try fill.step(&f, &other)) {}
        try std.testing.expectEqualDeep(recording.reference(&tokens), f.complete.get(&other).?);
        try std.testing.expectEqual(f.owners, f.freed);
    }
}

fn allocationCase(gpa: std.mem.Allocator) !void {
    var f = Fake.init(gpa);
    defer f.deinit();
    const tokens: [65]u32 = @splat(5);
    var s = try lanes.Stream.init(gpa, .{ .id = "allocation", .prompt = &tokens, .max_new = 4, .chunks = &.{ 1, 17, 33 } });
    defer {
        f.abortPrompt(&s);
        s.deinit(gpa);
    }
    while (!try fill.step(&f, &s)) {}
    try std.testing.expectEqual(f.owners, f.freed);
}

test "flashnext prefill: every cursor and owned-head allocation failure is leak free" {
    try std.testing.checkAllAllocationFailures(a, allocationCase, .{});
}

test "flashnext prefill: a failed image owner leaves the other prompt rows alive across yields" {
    var f: Fake = .{};
    defer f.deinit();
    const tokens: [33]u32 = @splat(5);
    const pixels = [_]f32{1};
    const images_a = [_]lanes.stream.Image{.{ .pixels = &pixels, .gh = 1, .gw = 1, .at = 0, .tokens = 1, .hash = 111 }};
    const images_b = [_]lanes.stream.Image{.{ .pixels = &pixels, .gh = 1, .gw = 1, .at = 0, .tokens = 1, .hash = 222 }};
    var first = try stream(&tokens, "image-a");
    defer first.deinit(a);
    var second = try stream(&tokens, "image-b");
    defer second.deinit(a);
    first.images = &images_a;
    second.images = &images_b;
    first.chunks = &.{ 1, 17 };
    second.chunks = &.{17};
    try std.testing.expect(!try fill.step(&f, &first));
    const owned = f.filling.get(&first).?.owner;
    f.fail_image = true;
    try std.testing.expectError(error.ImageFailed, fill.step(&f, &second));
    try std.testing.expectEqual(owned, f.filling.get(&first).?.owner);
    try std.testing.expectEqual(@as(u32, 1), first.fill_at.load(.monotonic));
    f.fail_image = false;
    while (!try fill.step(&f, &second)) {}
    while (!try fill.step(&f, &first)) {}
    try std.testing.expectEqualDeep(recording.referenceImages(&tokens, &images_a), f.complete.get(&first).?);
    try std.testing.expectEqualDeep(recording.referenceImages(&tokens, &images_b), f.complete.get(&second).?);
    try std.testing.expect(f.complete.get(&first).?.hash != f.complete.get(&second).?.hash);
    try std.testing.expectEqual(f.owners, f.freed);
}

test "flashnext prefill: callback insertions can rehash the fill map without losing the active cursor" {
    var f: Fake = .{ .max_slots = 64 };
    defer f.deinit();
    const tokens: [33]u32 = @splat(9);
    var first = try stream(&tokens, "active");
    defer first.deinit(a);
    first.chunks = &.{17};
    var others: [32]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (others[0..made]) |*s| {
        f.abortPrompt(s);
        s.deinit(a);
    };
    for (&others) |*s| {
        s.* = try stream(&tokens, "inserted");
        made += 1;
    }
    const Insert = struct {
        fake: *Fake,
        streams: []lanes.Stream,
        failed: ?anyerror = null,
        fn chunk(ptr: *anyopaque, _: *lanes.Stream) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            c.fake.chunk_hook = null;
            for (c.streams) |*s| fill.start(c.fake, s, null) catch |err| {
                c.failed = err;
                return;
            };
        }
    };
    var insert: Insert = .{ .fake = &f, .streams = &others };
    f.chunk_hook = Insert.chunk;
    f.chunk_ctx = &insert;
    try std.testing.expect(!try fill.step(&f, &first));
    try std.testing.expect(insert.failed == null);
    try std.testing.expectEqual(@as(u32, 17), f.filling.get(&first).?.at);
    while (!try fill.step(&f, &first)) {}
    try std.testing.expectEqualDeep(recording.reference(&tokens), f.complete.get(&first).?);
    for (&others) |*s| f.abortPrompt(s);
    try std.testing.expectEqual(f.owners, f.freed);
}

test "flashnext prefill: partial calls cannot draw or draft and max-one EOS opens only after completion" {
    const tokens = [_]u32{ 1, 4, 1, 5, 9, 2, 6 };
    for ([_]bool{ false, true }) |eos| {
        var f: Fake = .{};
        defer f.deinit();
        var cfg = try lanes.Config.init(a, .{ .exact_width = 2, .mtp = true, .speculate = true, .drafts = 1, .hidden_rows = true, .streams_exact = true, .batch_rows = 4, .max_streams = 2 }, 2, 1);
        defer cfg.deinit(a);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
        defer core.deinit();
        const first = lanes.fake.next(&tokens, null, tokens.len);
        var s = try lanes.Stream.init(a, .{ .id = "first-only", .prompt = &tokens, .max_new = if (eos) 8 else 1, .eos = if (eos) &.{first} else &.{}, .chunks = &.{ 1, 3, 6 } });
        defer s.deinit(a);
        try std.testing.expect(!try core.fillStream(&s, true));
        try std.testing.expectEqual(@as(usize, 0), core.activeCount());
        try std.testing.expectEqual(@as(usize, 0), f.target.drafts_calls);
        try std.testing.expectEqual(tokens.len, s.context.items.len);
        try std.testing.expect(s.pending == null);
        while (!try core.fillStream(&s, false)) {
            try std.testing.expectEqual(@as(usize, 0), core.activeCount());
            try std.testing.expectEqual(@as(usize, 0), f.target.drafts_calls);
        }
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(tokens.len + 1, s.context.items.len);
        try std.testing.expectEqual(first, s.context.items[tokens.len]);
        try std.testing.expectEqual(@as(usize, 0), core.activeCount());
        try std.testing.expectEqual(@as(usize, 1), f.begins);
        try std.testing.expectEqual(@as(usize, 1), f.ends);
        try std.testing.expectEqual(f.owners, f.freed);
    }
}
