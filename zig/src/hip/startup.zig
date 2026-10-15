//! A group's startup in steps every rank takes over the link: one rank's refusal reaches all before any moves on.

const std = @import("std");
const collective = @import("collective.zig");

pub const Error = error{ PeerRefused, LinkFailed };

/// A family's refusal: `problem` holds why (the server prints it).
pub const Refused = error{Refused};

/// `problem` set to the reason, formatted in `a`.
pub fn refusal(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// One rank's steps; a lone rank (no link) takes them alone. Between two steps a rank runs no collective.
pub const Startup = struct {
    link: ?collective.Link = null,
    /// The verdict that stopped this rank: the first rank that refused and why (null while every rank goes on).
    stopped: ?collective.Verdict = null,

    /// This rank went through the step: the step's values on every rank, or error.PeerRefused when a rank did not.
    pub fn done(s: *Startup, values: [2]u64, how: collective.How) Error![2]u64 {
        const l = s.link orelse return values;
        const v = l.agree(.ok(values), how) catch return error.LinkFailed;
        if (v.refused != 0) {
            s.stopped = v;
            return error.PeerRefused;
        }
        return v.values;
    }

    /// This rank cannot go on: every rank hears `why` at the step it is on, and this rank stops there too.
    pub fn refuse(s: *Startup, why: []const u8) void {
        const l = s.link orelse return;
        if (s.stopped != null) return;
        s.stopped = l.agree(.refuse(why), .first) catch return;
    }

    /// This rank failed with `err` between steps: the others hear its name (a step's own failure was heard already).
    pub fn failed(s: *Startup, err: anyerror) void {
        if (err != error.PeerRefused and err != error.LinkFailed) s.refuse(@errorName(err));
    }

    /// This rank's refusal: every rank hears it at the step it is on.
    pub fn refuseAll(
        s: *Startup,
        a: std.mem.Allocator,
        problem: *[]const u8,
        comptime fmt: []const u8,
        args: anytype,
    ) Refused {
        const r = refusal(a, problem, fmt, args);
        s.refuse(problem.*);
        return r;
    }

    /// A step every rank takes: its values on every rank, or the refusal of the first rank that could not go on.
    pub fn together(
        s: *Startup,
        a: std.mem.Allocator,
        values: [2]u64,
        how: collective.How,
        problem: *[]const u8,
    ) Refused![2]u64 {
        return s.done(values, how) catch return s.stoppedBy(a, problem);
    }

    /// Why this rank stopped: another rank refused (named, with its reason), or the ranks' link failed.
    pub fn stoppedBy(s: *const Startup, a: std.mem.Allocator, problem: *[]const u8) Refused {
        const v = s.stopped orelse return refusal(a, problem, "the ranks' link failed during startup", .{});
        return refusal(a, problem, "rank {d} of the group refused: {s}", .{ v.rank, v.reason() });
    }
};

test "a lone rank's steps pass its own values; a refusal without a link is its alone" {
    var s: Startup = .{};
    try std.testing.expectEqual([2]u64{ 3, 4 }, try s.done(.{ 3, 4 }, .least));
    s.refuse("no memory");
    try std.testing.expect(s.stopped == null);
}

/// A link whose other rank refuses every step with "rank 1 has no memory"; `steps` counts the steps taken.
const Refusing = struct {
    steps: usize = 0,

    fn agree(p: *anyopaque, mine: collective.Step, how: collective.How) collective.LinkError!collective.Verdict {
        const f: *Refusing = @ptrCast(@alignCast(p));
        f.steps += 1;
        const steps = [_]collective.Step{ mine, .refuse("rank 1 has no memory") };
        return @import("agree.zig").combine(&steps, how);
    }
    fn send(_: *anyopaque, _: []const u32) collective.LinkError!void {}
    fn recv(_: *anyopaque, _: std.mem.Allocator, _: *std.ArrayList(u32)) collective.LinkError!void {}

    fn link(f: *Refusing) collective.Link {
        return .{ .ptr = f, .vtable = &.{ .send = send, .recv = recv, .agree = agree } };
    }
};

test "a step another rank refused stops this rank with that rank's reason" {
    var f: Refusing = .{};
    var s: Startup = .{ .link = f.link() };
    try std.testing.expectError(error.PeerRefused, s.done(.{ 1, 1 }, .least));
    try std.testing.expectEqual(@as(u32, 1), s.stopped.?.rank);
    try std.testing.expectEqualStrings("rank 1 has no memory", s.stopped.?.reason());
}

test "a family's refusals: another rank's named with its reason, this rank's heard once, a lone rank's link" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var f: Refusing = .{};
    var s: Startup = .{ .link = f.link() };
    try std.testing.expectError(error.Refused, s.together(a, .{ 2, 0 }, .least, &problem));
    try std.testing.expectEqualStrings("rank 1 of the group refused: rank 1 has no memory", problem);
    // a rank already stopped sends nothing more: its refusal and a step's own failure take no step
    s.failed(error.OutOfMemory);
    try std.testing.expectEqual(error.Refused, s.refuseAll(a, &problem, "--parallel {d} does not fit", .{4}));
    try std.testing.expectEqualStrings("--parallel 4 does not fit", problem);
    try std.testing.expectEqual(@as(usize, 1), f.steps);
    var g: Refusing = .{};
    var t: Startup = .{ .link = g.link() };
    t.failed(error.PeerRefused);
    t.failed(error.LinkFailed);
    try std.testing.expectEqual(@as(usize, 0), g.steps);
    t.failed(error.OutOfDeviceMemory);
    try std.testing.expectEqual(@as(usize, 1), g.steps);
    var lone: Startup = .{};
    try std.testing.expectEqual(error.Refused, lone.stoppedBy(a, &problem));
    try std.testing.expectEqualStrings("the ranks' link failed during startup", problem);
}
