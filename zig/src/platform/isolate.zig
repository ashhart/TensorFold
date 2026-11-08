//! A test body run in a forked child, so a panic in the code under test fails that test instead of the whole binary.
const std = @import("std");
const builtin = @import("builtin");

pub const Outcome = enum { passed, failed, panicked, hung };

/// `body(args)` in a child: passed, failed (an error, named on stderr), panicked (a signal) or hung (60 s, killed).
pub fn run(comptime body: anytype, args: anytype) error{ SkipZigTest, ForkFailed }!Outcome {
    return if (comptime builtin.os.tag == .windows) error.SkipZigTest else child(body, args);
}

fn child(comptime body: anytype, args: anytype) error{ForkFailed}!Outcome {
    const pid = std.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        @call(.auto, body, args) catch |err| {
            std.debug.print("isolated test body returned {s}\n", .{@errorName(err)});
            std.c._exit(1);
        };
        std.c._exit(0);
    }
    var status: c_int = 0;
    for (0..60_000) |_| {
        if (std.c.waitpid(pid, &status, std.c.W.NOHANG) == pid) {
            if (status & 0x7f != 0) return .panicked;
            return if ((status >> 8) & 0xff == 0) .passed else .failed;
        }
        _ = std.c.nanosleep(&.{ .sec = 0, .nsec = 1_000_000 }, null);
    }
    _ = std.c.kill(pid, .KILL);
    _ = std.c.waitpid(pid, &status, 0);
    return .hung;
}
