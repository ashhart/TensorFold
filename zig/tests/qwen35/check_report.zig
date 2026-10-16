//! The lines of a `check` run: one a check, PASS, FAIL or SKIP with its numbers, and the summary.

const std = @import("std");

pub const Report = struct {
    passed: usize = 0,
    failed: usize = 0,
    skipped: usize = 0,

    pub fn pass(r: *Report, name: []const u8, comptime fmt: []const u8, args: anytype) void {
        r.passed += 1;
        std.debug.print("PASS {s}: " ++ fmt ++ "\n", .{name} ++ args);
    }

    pub fn fail(r: *Report, name: []const u8, comptime fmt: []const u8, args: anytype) void {
        r.failed += 1;
        std.debug.print("FAIL {s}: " ++ fmt ++ "\n", .{name} ++ args);
    }

    pub fn skip(r: *Report, name: []const u8, why: []const u8) void {
        r.skipped += 1;
        std.debug.print("SKIP {s}: {s}\n", .{ name, why });
    }

    /// A check that stopped on an error instead of a verdict.
    pub fn broke(r: *Report, name: []const u8, err: anyerror) void {
        r.fail(name, "error.{s}", .{@errorName(err)});
    }

    pub fn summary(r: Report, seconds: f64) void {
        std.debug.print("check {s}: {d} passed, {d} failed, {d} skipped in {d:.1} s\n", .{ if (r.failed == 0) "PASS" else "FAIL", r.passed, r.failed, r.skipped, seconds });
    }
};
