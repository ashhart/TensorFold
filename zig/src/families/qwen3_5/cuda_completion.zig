//! A failed owned forward must finish reading its media before the last owner releases it.
const cuda = @import("cuda");
pub fn release(d: *const cuda.Driver, stream: cuda.Stream, owner: anytype) void {
    @import("qwen_vision").completion.drain(d, stream);
    owner.deinit();
}
const std = @import("std");
const Fixture = struct {
    var pending = false;
    var fail_stream = false;
    var freed: usize = 0;
    var freed_early: usize = 0;
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_stream) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        pending = false;
        return 0;
    }
    fn deinit(_: *@This()) void {
        freed += 1;
        if (pending) freed_early += 1;
    }
};
test "Trainer admission: a failed forward's media waits for stream or context completion before release" {
    for ([_]bool{ false, true }) |fails| {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuStreamSynchronize = Fixture.sync;
        d.api.cuCtxSynchronize = Fixture.context;
        Fixture.pending = true;
        Fixture.fail_stream = fails;
        Fixture.freed = 0;
        Fixture.freed_early = 0;
        var media: Fixture = .{};
        release(&d, .{ .d = &d, .handle = null }, &media);
        try std.testing.expectEqual(@as(usize, 1), Fixture.freed);
        try std.testing.expectEqual(@as(usize, 0), Fixture.freed_early);
        try std.testing.expect(!Fixture.pending);
    }
}
