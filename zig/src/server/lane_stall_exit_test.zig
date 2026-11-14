//! --stall-exit: the process itself leaves with the restart status while a round stays open (POSIX, in a fork).
const std = @import("std");
const api = @import("engine_api");
const platform = @import("sys");
const t = std.testing;

/// A real watch thread over a round that never ends; returning at all means it did not exit.
fn stalledRound() u8 {
    var w: api.stall.Watch = .{};
    w.start(std.heap.page_allocator, t.io, .{ .timeout_ns = 50 * std.time.ns_per_ms, .exit = true, .family = "fixture" }) catch return 2;
    const round = w.activity() orelse return 3;
    round.begin(round.ctx);
    std.Io.sleep(t.io, .fromSeconds(4), .awake) catch {};
    return 4;
}

test "lane sentinel exit: --stall-exit ends the process with status 70 once a round outlasts the deadline" {
    const pid = platform.c.fork();
    if (pid < 0) return error.SkipZigTest;
    if (pid == 0) platform.c._exit(stalledRound());
    var status: c_int = 0;
    var waited: u32 = 0;
    const exited = while (waited < 6000) : (waited += 10) {
        if (platform.c.waitpid(pid, &status, platform.c.W.NOHANG) == pid) break true;
        std.Io.sleep(t.io, .fromMilliseconds(10), .awake) catch {};
    } else false;
    if (!exited) {
        _ = platform.c.kill(pid, .KILL);
        _ = platform.c.waitpid(pid, &status, 0);
        return error.TimedOut;
    }
    const s: u32 = @bitCast(status);
    try t.expect(platform.c.W.IFEXITED(s));
    try t.expectEqual(api.stall.exit_code, platform.c.W.EXITSTATUS(s));
}
