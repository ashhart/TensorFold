//! Actual snapshot driver mocks discriminate reservation-only handoff, valid save and fenced failure ownership.
const std = @import("std");
const lanes = @import("lanes");
const cache_mod = @import("exact_prompt_cache.zig");
const control = @import("exact_post_first.zig");
const mocks = @import("exact_cache_test.zig");
const Mock = mocks.Mock;
const Fixture = mocks.Fixture;
const g = @import("shape.zig").the_27b;
const a = std.testing.allocator;

fn stream() !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = "post-first", .prompt = &.{ 1, 2, 3 }, .max_new = 256, .drafts = false });
}

fn committed(s: *lanes.Stream) !void {
    _ = try s.commit(a, &.{7}, &.{});
    s.pending = 7;
}

fn cache() cache_mod.Cache {
    var out = cache_mod.Cache.init(a, true);
    out.logits_enabled = true;
    out.post_first.enabled = true;
    return out;
}

test "post-first flag is strict default-off and requires exact cache" {
    try std.testing.expect(!try control.enabled(null, false));
    try std.testing.expect(!try control.enabled("0", true));
    try std.testing.expect(try control.enabled("1", true));
    try std.testing.expectError(error.PostFirstCaptureNeedsExactCache, control.enabled("1", false));
    for ([_][]const u8{ "", "2", "true", " 1" }) |raw|
        try std.testing.expectError(error.InvalidPostFirstCaptureFlag, control.enabled(raw, true));
}

test "one reservation performs no CUDA work and saves original state and raw logits only after commit" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var c = cache();
    defer c.deinit(ops) catch unreachable;
    var s = try stream();
    defer s.deinit(a);
    var other = try stream();
    defer other.deinit(a);
    var seq = f.seq(3);
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try c.captureBeforeFirst(&other, &seq, null, ops, g);
    try std.testing.expectEqual(@as(usize, 0), Mock.live + Mock.copies + Mock.fences);
    try std.testing.expectEqual(@as(u64, 1), c.post_first.reservations);
    try std.testing.expectEqual(@as(u64, 1), c.stats.refusals);
    try c.finishAfterFirst(&other, &seq, null, ops, g, 0x7770);
    try std.testing.expect(c.post_first.reserved.?.stream == &s);
    try committed(&s);
    try c.finishAfterFirst(&s, &seq, null, ops, g, 0x7770);
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    try std.testing.expectEqual(@as(usize, 131), Mock.copies);
    try std.testing.expectEqual(@as(usize, 0), Mock.fences);
    try std.testing.expectEqual(@as(u64, 0x7770), Mock.copy_rows[130].src);
    try std.testing.expectEqual(@as(usize, 496640), Mock.copy_rows[130].bytes);
    try std.testing.expect(c.pending.?.logits_valid and c.post_first.reserved == null);
    try std.testing.expectEqual(@as(u64, 1), c.post_first.saves);
    s.finished = true;
    s.reason = .length;
    try c.release(&s, true, ops);
    try std.testing.expectEqual(@as(u64, 1), c.stats.promotions);
}

test "disabled path keeps capture before first and terminal reservation creates no snapshot" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var c = cache();
    defer c.deinit(ops) catch unreachable;
    var s = try stream();
    defer s.deinit(a);
    var seq = f.seq(3);
    c.post_first.enabled = false;
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    try std.testing.expectEqual(@as(usize, 130), Mock.copies);
    try c.abandon(&s, ops);
    Mock.reset();
    c.post_first.enabled = true;
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try committed(&s);
    s.finished = true;
    s.reason = .length;
    try c.release(&s, true, ops);
    try std.testing.expectEqual(@as(usize, 0), Mock.live + Mock.copies + Mock.fences);
    try std.testing.expect(c.post_first.reserved == null);
}

test "allocation refusal after handoff remains uncached generation with no device copies" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var c = cache();
    defer c.deinit(ops) catch unreachable;
    var s = try stream();
    defer s.deinit(a);
    var seq = f.seq(3);
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try committed(&s);
    Mock.fail_alloc = true;
    try c.finishAfterFirst(&s, &seq, null, ops, g, 0x7770);
    Mock.fail_alloc = false;
    try std.testing.expectEqual(@as(usize, 0), Mock.live + Mock.copies + Mock.fences);
    try std.testing.expect(!c.failed and c.pending == null);
    try std.testing.expectEqual(@as(u64, 1), c.stats.refusals);
    try std.testing.expectEqualSlices(u32, &.{7}, s.emitted());
}

test "post-send copy failure preserves pending ownership until a successful original fence" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var c = cache();
    defer c.deinit(ops) catch unreachable;
    var s = try stream();
    defer s.deinit(a);
    var seq = f.seq(3);
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try committed(&s);
    Mock.fail_copy = true;
    try std.testing.expectError(error.NotReady, c.finishAfterFirst(&s, &seq, null, ops, g, 0x7770));
    Mock.fail_copy = false;
    try std.testing.expect(c.failed and c.pending != null and c.post_first.reserved == null);
    try std.testing.expectEqual(@as(u64, 1), c.post_first.failures);
    Mock.fail_fence = true;
    try std.testing.expectError(error.NotReady, c.abandon(&s, ops));
    try std.testing.expectEqual(@as(usize, 1), Mock.live);
    Mock.fail_fence = false;
    try c.abandon(&s, ops);
    try std.testing.expectEqual(@as(usize, 0), Mock.live);
    try std.testing.expectEqualSlices(u32, &.{7}, s.emitted());
}

const Cancel = struct {
    var value = false;
    fn check(_: *anyopaque) bool {
        return value;
    }
    fn copied() void {
        value = true;
    }
};

test "cancel before or during save clears reservation and fenced abandon purges copied state" {
    for ([_]bool{ false, true }) |during| {
        Mock.reset();
        Cancel.value = false;
        var marker: u8 = 0;
        var f: Fixture = .{};
        const ops = f.ops();
        var c = cache();
        defer c.deinit(ops) catch unreachable;
        var s = try stream();
        defer s.deinit(a);
        s.cancel_check = .{ .ptr = &marker, .check = Cancel.check };
        var seq = f.seq(3);
        try c.captureBeforeFirst(&s, &seq, null, ops, g);
        try committed(&s);
        if (during) Mock.copied = Cancel.copied else Cancel.value = true;
        try std.testing.expectError(error.Cancelled, c.finishAfterFirst(&s, &seq, null, ops, g, 0x7770));
        Mock.copied = null;
        try std.testing.expect(c.post_first.reserved == null);
        try c.abandon(&s, ops);
        try std.testing.expectEqual(@as(usize, 0), Mock.live);
        try std.testing.expectEqual(@as(u64, 0), c.stats.promotions);
        Cancel.value = false;
    }
}

test "state boundary changed after reservation rejects before allocation or copy" {
    Mock.reset();
    var f: Fixture = .{};
    const ops = f.ops();
    var c = cache();
    defer c.deinit(ops) catch unreachable;
    var s = try stream();
    defer s.deinit(a);
    var seq = f.seq(3);
    try c.captureBeforeFirst(&s, &seq, null, ops, g);
    try committed(&s);
    seq.pos += 1;
    try std.testing.expectError(error.InvalidPostFirstCapture, c.finishAfterFirst(&s, &seq, null, ops, g, 0x7770));
    try std.testing.expectEqual(@as(usize, 0), Mock.live + Mock.copies + Mock.fences);
}
