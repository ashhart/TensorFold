//! kernel-contract's shard processes without a GPU: stopping them, forwarding signals and following the parent.
const std = @import("std");
const builtin = @import("builtin");
const t = std.testing;
const children = @import("contract_children.zig");

fn sleeper() !std.process.Child {
    return std.process.spawn(t.io, .{ .argv = &.{ "/bin/sleep", "60" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
}

fn killedBy(child: *std.process.Child, sig: std.posix.SIG) !void {
    const term = try child.wait(t.io);
    try t.expect(term == .signal);
    try t.expectEqual(sig, term.signal);
}

test "shard children: a stop reaches every registered shard but the exception, and only the first says why" {
    var kept = try sleeper();
    defer kept.kill(t.io);
    var gone = try sleeper();
    defer gone.kill(t.io);
    children.register(2, kept.id.?);
    children.register(3, gone.id.?);
    defer children.forget(2);
    defer children.forget(3);
    const already = children.isStopping();
    try t.expectEqual(!already, children.stopAll(2, .TERM));
    try killedBy(&gone, .TERM);
    children.forget(3);
    try t.expect(children.wasStopped(3));
    try t.expect(!children.wasStopped(2));
    try t.expect(!children.stopAll(2, .TERM));
    try t.expect(children.isStopping());
}

test "shard children: SIGTERM sent to the parent is forwarded to its shards" {
    var child = try sleeper();
    defer child.kill(t.io);
    children.register(0, child.id.?);
    defer children.forget(0);
    children.forwardSignals();
    defer {
        const default: std.posix.Sigaction = .{ .handler = .{ .handler = std.posix.SIG.DFL }, .mask = std.posix.sigemptyset(), .flags = 0 };
        std.posix.sigaction(.TERM, &default, null);
        std.posix.sigaction(.INT, &default, null);
    }
    try std.posix.raise(.TERM);
    try killedBy(&child, .TERM);
}

test "shard children: a shard follows the parent that started it, and refuses another" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const parent = std.os.linux.getppid();
    try children.followParent(parent);
    try t.expectError(error.ParentGone, children.followParent(parent + 1));
    var armed: c_int = 0;
    try t.expectEqual(@as(usize, 0), std.os.linux.prctl(@backingInt(std.os.linux.PR.GET_PDEATHSIG), @intFromPtr(&armed), 0, 0, 0));
    try t.expectEqual(@as(c_int, @backingInt(std.os.linux.SIG.KILL)), armed);
}
