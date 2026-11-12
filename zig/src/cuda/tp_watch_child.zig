//! The peer-watcher test child's stderr: a pipe it writes, and the lines it must have logged before it stopped.
const std = @import("std");
const posix = @import("sys").posix;

pub extern "c" fn dup2(c_int, c_int) c_int;
extern "c" fn pipe(*[2]c_int) c_int;

pub fn open() ![2]c_int {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return error.PipeFailed;
    return fds;
}

/// What a reaped child wrote: every byte waiting in the pipe, without waiting for more.
pub fn drain(fd: c_int, buf: []u8) []const u8 {
    var got: usize = 0;
    while (got < buf.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if ((posix.poll(&fds, 0) catch 0) == 0) break;
        const rc = posix.system.read(fd, buf[got..].ptr, buf.len - got);
        if (posix.errno(rc) == .INTR) continue;
        if (posix.errno(rc) != .SUCCESS or rc == 0) break;
        got += @intCast(rc);
    }
    return buf[0..got];
}

/// The watcher's log for each mode: a closed line, a peer's error record, an intentional end.
pub fn said(mode: u8) []const u8 {
    return switch (mode) {
        0 => "[default] (err): rank 0: another rank's process ended (its line closed); " ++
            "this rank stops instead of waiting on it\n",
        1 => "[default] (err): rank 0: rank 1 stopped (peer OOM); this rank stops too\n",
        4 => "[default] (err): rank 0: rank 1 stopped (PeerWarmupTimedOut); this rank stops too\n",
        else => "",
    };
}
