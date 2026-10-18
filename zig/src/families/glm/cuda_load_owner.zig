//! Own a newly loaded model across the fallible post-load transformation.
const std = @import("std");
pub fn finish(loaded: anytype, ctx: anytype, comptime transform: anytype) !@TypeOf(loaded) {
    var w = loaded;
    errdefer w.deinit();
    try transform(ctx, &w);
    return w;
}
const Fake = struct {
    live: usize = 1,
    queued: bool = false,
    drained: bool = false,
    phase: usize = 0,
    const W = struct {
        f: *Fake,
        fn deinit(w: *@This()) void {
            std.debug.assert(!w.f.queued or w.f.drained);
            w.f.live -= 1;
        }
    };
    fn transform(f: *Fake, _: *W) !void {
        if (f.phase == 0) return error.OutOfMemory;
        f.queued = true;
        // The real tiler drains its stream, or terminates on an unrecoverable context, before releasing scratch.
        defer f.drained = true;
        if (f.phase == 1) return error.EnqueueFailed;
        if (f.phase == 2) return error.SyncFailed;
    }
};
test "loaded weights are freed after a tiling failure at allocation, enqueue or wait, and a retry succeeds" {
    for (0..3) |phase| {
        var f: Fake = .{ .phase = phase };
        const result = finish(Fake.W{ .f = &f }, &f, Fake.transform);
        if (result) |_| return error.ExpectedFailure else |_| {}
        try std.testing.expectEqual(@as(usize, 0), f.live);
        f = .{ .phase = 3 };
        var w = try finish(Fake.W{ .f = &f }, &f, Fake.transform);
        w.deinit();
        try std.testing.expectEqual(@as(usize, 0), f.live);
    }
}

const cuda = @import("cuda");

pub fn drain(s: cuda.Stream, d: *const cuda.Driver) void {
    cuda.completion.drain(d, s, "a GLM stream and its context did not drain");
}
