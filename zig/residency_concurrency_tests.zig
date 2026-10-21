//! A follower may publish snapshots while the lane host's idle ticker flushes the same set.
const std = @import("std");
const mtl = @import("metal");
const a = std.testing.allocator;
const Job = struct {
    owner: *mtl.resident.Resident,
    buffer: mtl.Buffer,
    go: *std.atomic.Value(bool),
    fn run(j: @This()) void {
        while (!j.go.load(.acquire)) std.atomic.spinLoopHint();
        for (0..1000) |_| {
            j.owner.remove(j.buffer);
            j.owner.addNow(&.{j.buffer});
        }
    }
};

test "residency owner shared: two snapshot publishers and an idle ticker serialize membership changes" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const one = try record.device().buffer(64, 0);
        defer one.deinit();
        const two = try record.device().buffer(64, 0);
        defer two.deinit();
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{ one, two }, "shared metal")).?;
        defer owner.deinit();
        var go: std.atomic.Value(bool) = .init(false);
        const first = try std.Thread.spawn(.{}, Job.run, .{Job{ .owner = &owner, .buffer = one, .go = &go }});
        const second = std.Thread.spawn(.{}, Job.run, .{Job{ .owner = &owner, .buffer = two, .go = &go }}) catch |err| {
            go.store(true, .release);
            first.join();
            return err;
        };
        go.store(true, .release);
        var warm = owner.target();
        for (0..1000) |_| mtl.keepalive.Target.tick(&warm);
        first.join();
        second.join();
        owner.addNow(&.{ one, two });
        try record.check();
    }
    try record.allFreed();
}
