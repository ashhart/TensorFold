//! A group's startup in steps every rank takes over the link: one rank's refusal reaches all before any moves on.

const std = @import("std");
const collective = @import("collective.zig");

pub const Error = error{ PeerRefused, LinkFailed };

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
};

test "a lone rank's steps pass its own values; a refusal without a link is its alone" {
    var s: Startup = .{};
    try std.testing.expectEqual([2]u64{ 3, 4 }, try s.done(.{ 3, 4 }, .least));
    s.refuse("no memory");
    try std.testing.expect(s.stopped == null);
}

test "a step another rank refused stops this rank with that rank's reason" {
    const Fake = struct {
        fn agree(_: *anyopaque, mine: collective.Step, how: collective.How) collective.LinkError!collective.Verdict {
            const steps = [_]collective.Step{ mine, .refuse("rank 1 has no memory") };
            return @import("agree.zig").combine(&steps, how);
        }
        fn send(_: *anyopaque, _: []const u32) collective.LinkError!void {}
        fn recv(_: *anyopaque, _: std.mem.Allocator, _: *std.ArrayList(u32)) collective.LinkError!void {}
    };
    var dummy: u8 = 0;
    var s: Startup = .{ .link = .{
        .ptr = &dummy,
        .vtable = &.{ .send = Fake.send, .recv = Fake.recv, .agree = Fake.agree },
    } };
    try std.testing.expectError(error.PeerRefused, s.done(.{ 1, 1 }, .least));
    try std.testing.expectEqual(@as(u32, 1), s.stopped.?.rank);
    try std.testing.expectEqualStrings("rank 1 has no memory", s.stopped.?.reason());
}
