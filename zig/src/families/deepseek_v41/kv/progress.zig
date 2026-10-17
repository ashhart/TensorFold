//! When the round planner's log lines are due: admission waits and long prefills, first and then every `every_s`.
const std = @import("std");

/// Seconds between a request's repeated lines.
pub const every_s: f64 = 30;
/// Prompts with fewer tokens to prefill than this get no progress lines.
pub const long_tokens: u64 = 16384;

/// Requests the planner keeps queued: since when, and when their wait was last logged.
pub const Waits = struct {
    map: std.AutoArrayHashMapUnmanaged(usize, [2]f64) = .empty,

    pub fn deinit(w: *Waits, gpa: std.mem.Allocator) void {
        w.map.deinit(gpa);
    }

    /// `key` waits at `now`: the seconds it has waited when a line is due (its first wait, then every `every_s`).
    pub fn waiting(w: *Waits, gpa: std.mem.Allocator, key: usize, now: f64) !?f64 {
        const g = try w.map.getOrPut(gpa, key);
        if (!g.found_existing) {
            g.value_ptr.* = .{ now, now };
            return 0;
        }
        if (now - g.value_ptr[1] < every_s) return null;
        g.value_ptr[1] = now;
        return now - g.value_ptr[0];
    }

    /// `key` was admitted or left: the seconds it waited, if it waited at all.
    pub fn done(w: *Waits, key: usize, now: f64) ?f64 {
        const kv = w.map.fetchSwapRemove(key) orelse return null;
        return now - kv.value[0];
    }
};

/// A long prompt's prefill: its length, where it started, and its last logged point.
const Run = struct { n: u64, from: u64, start: f64, at: f64, done: u64 };

pub const Line = struct { done: u64, n: u64, seconds: f64, rate: f64 };

pub const Prefills = struct {
    map: std.AutoArrayHashMapUnmanaged(usize, Run) = .empty,

    pub fn deinit(p: *Prefills, gpa: std.mem.Allocator) void {
        p.map.deinit(gpa);
    }

    /// `key`'s prompt holds `done` of `n` tokens at `now`: a line when one is due (tokens a second since the last).
    pub fn at(p: *Prefills, gpa: std.mem.Allocator, key: usize, n: u64, done: u64, now: f64) !?Line {
        const g = try p.map.getOrPut(gpa, key);
        // a damaged saved entry restarts its prompt from 0: a new run
        if (!g.found_existing or done < g.value_ptr.done) g.value_ptr.* = .{ .n = n, .from = done, .start = now, .at = now, .done = done };
        const r = g.value_ptr;
        if (r.n -| r.from < long_tokens or now - r.at < every_s) return null;
        const line: Line = .{ .done = done, .n = n, .seconds = now - r.start, .rate = @as(f64, @floatFromInt(done - r.done)) / (now - r.at) };
        r.at = now;
        r.done = done;
        return line;
    }

    /// `key`'s prompt is in (or it left): the whole prefill's line for a long prompt.
    pub fn end(p: *Prefills, key: usize, done: u64, now: f64) ?Line {
        const kv = p.map.fetchSwapRemove(key) orelse return null;
        const r = kv.value;
        if (r.n -| r.from < long_tokens or now <= r.start) return null;
        return .{ .done = done, .n = r.n, .seconds = now - r.start, .rate = @as(f64, @floatFromInt(done -| r.from)) / (now - r.start) };
    }
};

test "progress: a wait's lines come first and then every 30 s; a long prefill's lines carry its rate" {
    const gpa = std.testing.allocator;
    var w: Waits = .{};
    defer w.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 0), try w.waiting(gpa, 7, 100));
    try std.testing.expectEqual(@as(?f64, null), try w.waiting(gpa, 7, 110));
    try std.testing.expectEqual(@as(?f64, 30), try w.waiting(gpa, 7, 130));
    try std.testing.expectEqual(@as(?f64, 45), w.done(7, 145));
    try std.testing.expectEqual(@as(?f64, null), w.done(7, 146)); // logged once
    var p: Prefills = .{};
    defer p.deinit(gpa);
    // a 183K prompt from 0: a line every 30 s with the rows since the last
    try std.testing.expectEqual(@as(?Line, null), try p.at(gpa, 1, 183154, 0, 0));
    try std.testing.expectEqual(@as(?Line, null), try p.at(gpa, 1, 183154, 8192, 10));
    const l = (try p.at(gpa, 1, 183154, 73728, 30)).?;
    try std.testing.expectEqual(@as(u64, 73728), l.done);
    try std.testing.expectApproxEqAbs(@as(f64, 2457.6), l.rate, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2442.04), p.end(1, 183153, 75).?.rate, 0.01);
    // a short prompt (or a resumed one with little left) gets none
    try std.testing.expectEqual(@as(?Line, null), try p.at(gpa, 2, 9000, 0, 0));
    try std.testing.expectEqual(@as(?Line, null), try p.at(gpa, 2, 9000, 8999, 40));
    try std.testing.expectEqual(@as(?Line, null), p.end(2, 8999, 41));
    // a restart from 0 (a damaged entry) is a new run, not a negative rate
    _ = try p.at(gpa, 3, 100000, 40000, 0);
    try std.testing.expectEqual(@as(?Line, null), try p.at(gpa, 3, 100000, 0, 31));
    try std.testing.expectEqual(@as(u64, 61440), (try p.at(gpa, 3, 100000, 61440, 61)).?.done);
}
