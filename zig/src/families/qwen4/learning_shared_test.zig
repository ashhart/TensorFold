//! The real shared learner reaches the native trainer and rejects malformed lessons without publishing rows.
const std = @import("std");
const native = @import("train_test.zig");
const fixture = @import("learning_native_fixture.zig");
const Slide = @import("slide.zig").Slide(fixture.E, native.T);
const a = std.testing.allocator;

test "Qwen4 learning: shared learner token refusal retains an empty pocket and releases its native trainer" {
    var f = try fixture.Fixture.open();
    defer f.deinit();
    const r = try native.R.init(a, f.e);
    defer r.close() catch unreachable;
    f.e.learning = r.listener();
    const held = f.b.held;
    {
        var learner = Slide.Learner.init(a, std.testing.io, .{ .e = f.e, .pockets = &r.pockets, .dir = f.path, .trainer = r });
        defer learner.deinit();
        const tokens = [_]u32{ 3, 5, 7, 1000, 9 };
        const examples = [_]Slide.Example{.{ .ids = &tokens, .start = 3, .decide = 2, .subject = 1 }};
        try learner.begin(.{ .train = &examples, .held = &examples, .steps = 1 });
        try std.testing.expect(learner.trainer != null);
        var result = learner.step();
        var steps: usize = 1;
        while (!result.done and steps < 12) : (steps += 1) result = learner.step();
        try std.testing.expect(result.done and result.report != null);
        try std.testing.expectEqualStrings("TokenOutOfRange", result.report.?.failed);
        for (r.pockets.layers) |layer| try std.testing.expect(layer == null or layer.?.used == 0);
    }
    try std.testing.expectEqual(@as(u64, 0), r.training_reserved);
    try std.testing.expectEqual(held, f.b.held);
    try std.testing.expect(f.e.trace == null and f.e.gpu.trace == null);
}
