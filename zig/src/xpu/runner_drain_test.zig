//! The 27B runner over a fake Level Zero: an error after a queued copy waits for the queue, a stuck queue is fatal.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");

/// A Level Zero that logs each call ('c' copy, 'l' launch, 's' wait) and fails the launch numbered `fail_launch`.
const Fake = struct {
    var calls: [32]u8 = undefined;
    var n: usize = 0;
    var launches: usize = 0;
    var fail_launch: usize = std.math.maxInt(usize);
    var stuck = false; // every wait reports a list that did not drain
    var lost_rc: ?u32 = null;

    fn note(c: u8) void {
        if (n < calls.len) calls[n] = c;
        n += 1;
    }
    fn copy(_: ze.H, _: ?*anyopaque, _: ?*const anyopaque, _: usize, _: ze.H, _: u32, _: ?[*]ze.H) callconv(.c) u32 {
        note('c');
        return 0;
    }
    fn arg(_: ze.H, _: u32, _: usize, _: ?*const anyopaque) callconv(.c) u32 {
        return 0;
    }
    fn launch(_: ze.H, _: ze.H, _: *const ze.GroupCount, _: ze.H, _: u32, _: ?[*]ze.H) callconv(.c) u32 {
        note('l');
        launches += 1;
        return @intFromBool(launches - 1 == fail_launch);
    }
    fn sync(_: ze.H, _: u64) callconv(.c) u32 {
        note('s');
        return @intFromBool(stuck);
    }
    fn lostHook(rc: u32) void {
        lost_rc = rc;
    }

    /// A runner whose device is this fake, with the activations argmaxMasked and rowLogits use at made-up addresses.
    fn runner(r: *q.Runner) void {
        n = 0;
        launches = 0;
        fail_launch = std.math.maxInt(usize);
        stuck = false;
        lost_rc = null;
        var api: ze.Api = undefined;
        api.zeCommandListAppendMemoryCopy = copy;
        api.zeKernelSetArgumentValue = arg;
        api.zeCommandListAppendLaunchKernel = launch;
        api.zeCommandListHostSynchronize = sync;
        r.d = .{ .g = .{ .lib = undefined, .api = api, .driver = null, .device = null, .context = null, .props = undefined }, .list = @ptrFromInt(0x10), .k = std.mem.zeroes(@TypeOf(r.d.k)) };
        for ([_]**anyopaque{ &r.acts.logits, &r.acts.pv, &r.acts.pi, &r.acts.mrows, &r.acts.mbits, &r.acts.mtok }, 1..) |p, i|
            p.* = @ptrFromInt(0x1000 * i);
    }
};

test "a masked argmax whose launch fails waits for the queue before it returns, so its staged rows outlive the copy" {
    var r: q.Runner = undefined;
    Fake.runner(&r);
    Fake.fail_launch = 1;
    const bits = [_]u32{ 0xffff_ffff, 0xffff_ffff };
    var out: [2]u32 = undefined;
    try std.testing.expectError(error.ZeFailed, r.argmaxMasked(&.{ 3, 4 }, &bits, 1, &out));
    try std.testing.expectEqualStrings("cclls", Fake.calls[0..Fake.n]); // the rows, the bits, two launches, then the wait
}

test "a queue that will not drain reaches the fatal hook before a row's logits staging is freed" {
    var r: q.Runner = undefined;
    Fake.runner(&r);
    r.host_logits = try std.testing.allocator.alloc(f32, q.VOCAB);
    defer std.testing.allocator.free(r.host_logits);
    const keep = dev.lost;
    defer dev.lost = keep;
    dev.lost = Fake.lostHook; // the process would end here; the hook returns so the test can look
    Fake.stuck = true;
    _ = try r.rowLogits(0);
    try std.testing.expectEqual(@as(?u32, 1), Fake.lost_rc);
    try std.testing.expectEqualStrings("cs", Fake.calls[0..Fake.n]);
}
