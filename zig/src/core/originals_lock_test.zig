//! One teacher a model folder: the hold a --slide server takes, held by another live process, then by a dead one.
const std = @import("std");
const builtin = @import("builtin");
const originals = @import("originals.zig");

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}

/// A child that takes a --slide server's hold on `dir`, says so on `ready`, then lives until it is killed.
fn holdAndWait(dir: []const u8, ready: std.c.fd_t, exit_now: bool) noreturn {
    var why: []const u8 = "";
    const held = originals.serving(std.heap.page_allocator, dir, true, &why) catch std.c._exit(2);
    _ = held orelse std.c._exit(3);
    _ = std.c.write(ready, "h", 1);
    if (exit_now) std.c._exit(0); // ends without releasing: the kernel drops the lock with the process
    while (true) _ = std.c.nanosleep(&.{ .sec = 1, .nsec = 0 }, null);
}

/// A second process's writer hold on `dir`: forked, ready once it holds; `exit_now` ends it at once.
fn otherWriter(dir: []const u8, exit_now: bool) !std.c.pid_t {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.pipe(&fds) != 0) return error.Pipe;
    defer _ = std.c.close(fds[0]);
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        _ = std.c.close(fds[0]);
        holdAndWait(dir, fds[1], exit_now);
    }
    _ = std.c.close(fds[1]);
    var byte: [1]u8 = undefined;
    if (std.c.read(fds[0], &byte, 1) != 1) return error.ChildDidNotHold;
    return pid;
}

test "a --slide server's folder: a second writer and a server are refused while it lives, and its lock dies with it" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var why: []const u8 = "";
    for ([_]bool{ false, true }) |exit_now| {
        const pid = try otherWriter(dir, exit_now);
        var status: c_int = 0;
        if (!exit_now) {
            // a live writer in another process: a second --slide server and a plain server both refused, naming it
            try std.testing.expectError(error.Busy, originals.serving(a, dir, true, &why));
            try std.testing.expect(has(why, dir) and has(why, "teach a copy"));
            try std.testing.expectError(error.Busy, originals.serving(a, dir, false, &why));
            try std.testing.expect(has(why, dir) and has(why, "--slide server"));
            _ = std.c.kill(pid, .KILL); // killed without releasing
        }
        _ = std.c.waitpid(pid, &status, 0);
        // its lock is stale and free: the next writer takes it, and holds it against another
        const next = (try originals.serving(a, dir, true, &why)) orelse return error.TestExpectedHold;
        try std.testing.expectError(error.Busy, originals.serving(a, dir, true, &why));
        next.release();
    }
}
