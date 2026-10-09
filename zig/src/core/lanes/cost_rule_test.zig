//! Scripted GPU-free measurements exercise the same pass used by Metal timing.
const std = @import("std");
const rule = @import("cost_rule.zig");

const Script = struct {
    at: usize = 0,
    window_calls: usize = 0,
    head_calls: usize = 0,
    events: [128]u8 = undefined,
    count: usize = 0,
    retry: bool = false,
    reject_head: bool = false,

    fn record(s: *Script, event: u8) void {
        s.events[s.count] = event;
        s.count += 1;
    }

    const Windows = struct {
        script: *Script,
        pub fn time(w: Windows, i: usize) !f64 {
            const s = w.script;
            s.at = i;
            s.window_calls += 1;
            s.record(@intCast(i));
            return if (s.retry) (if (i == 0) 12 else 18) else (if (i == 0) 10 else 20);
        }
    };

    const Heads = struct {
        script: *Script,
        pub fn time(h: Heads, _: usize) !f64 {
            const s = h.script;
            s.head_calls += 1;
            s.record(255);
            if (s.reject_head) return error.UnexpectedHead;
            return if (s.at == 0) (if (s.retry) 0.35 else 0.34) else 0.61;
        }
    };
};

test "the head is sampled once between widths and keeps the pass minimum" {
    var s: Script = .{};
    var ms: [3]f64 = undefined;
    var head: f64 = 0;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, &head, false);
    try std.testing.expectEqual(@as(f64, 0.34), head);
    try std.testing.expectEqual(@as(usize, 3), s.head_calls);
    try std.testing.expectEqual(@as(usize, 24), s.window_calls);
    for (0..3) |width| {
        const start = width * (rule.reps + 2);
        for (s.events[start .. start + rule.reps + 1]) |event| try std.testing.expectEqual(@as(u8, @intCast(width)), event);
        try std.testing.expectEqual(@as(u8, 255), s.events[start + rule.reps + 1]);
    }
}

test "an end-only slow stretch cannot set the head level" {
    var s: Script = .{};
    var ms: [2]f64 = undefined;
    var head: f64 = 0;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, &head, false);
    try std.testing.expectEqualSlices(f64, &.{ 10, 20 }, &ms);
    try std.testing.expectEqual(@as(f64, 0.34), head);
}

test "head-only drift retries the head and windows while retaining faster values" {
    var s: Script = .{};
    var ms: [2]f64 = undefined;
    var head: f64 = 0.5;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, null, false);
    try std.testing.expect(!rule.drifted(&ms, &.{ 10, 20 }));
    try std.testing.expect(rule.drifted(&.{head}, &.{0.34}));
    s.retry = true;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, &head, true);
    try std.testing.expectEqualSlices(f64, &.{ 10, 18 }, &ms);
    try std.testing.expectEqual(@as(f64, 0.35), head);
    try std.testing.expectEqual(@as(usize, 2), s.head_calls);
    s.retry = false;
    s.at = 1;
    head = 0.2;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, &head, true);
    try std.testing.expectEqual(@as(f64, 0.2), head);
}

test "headless passes never invoke the head timer and head errors propagate" {
    var s: Script = .{ .reject_head = true };
    var ms: [2]f64 = undefined;
    try rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, null, false);
    try std.testing.expectEqual(@as(usize, 0), s.head_calls);
    var head: f64 = 0;
    try std.testing.expectError(error.UnexpectedHead, rule.measure(Script.Windows{ .script = &s }, Script.Heads{ .script = &s }, &ms, &head, false));
}
