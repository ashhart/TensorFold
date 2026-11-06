//! The real shared trainer and owner run through recording storage with production scalar pullback bodies.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const n = @import("learning_numeric.zig");
const a = std.testing.allocator;
const T = @import("learning_train.zig").Trainer(n.E);
const turn: living.lesson_route.Turn = .{ .decide = 0, .whole = true };
const tokens = [_]u32{ 3, 5, 7, 2, 9 };
test "Gemma4 learning: bounded trainer records native nodes and pulls the last answer through local and tied-value layers" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const t = try T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e);
    defer t.deinit();
    _ = try t.step(&tokens, 3, .keys, turn);
    try std.testing.expect(t.normed(0, 0).len == 64);
    const baseline = try t.step(&tokens, 3, .loss, turn);
    t.nudge = .{ .layers = .{ 1, 1 << 4, 1 << 5 }, .rows = .{ 0, 1, 2 } };
    const target = try t.step(&tokens, 3, .target, turn);
    try std.testing.expectEqual(baseline.loss, target.loss);
    for ([_]usize{ 0, 4, 5 }, 0..) |l, key| {
        var mag: f64 = 0;
        for (t.pushAt(0, key, l)) |v| {
            try std.testing.expect(std.math.isFinite(v));
            mag += @abs(v);
        }
        try std.testing.expect(mag > 0);
    }
    try std.testing.expect(f.e.r.observer == null);
    f.e.omit = .key;
    try std.testing.expectError(error.GemmaTrainingTape, t.step(&tokens, 3, .target, turn));
    try std.testing.expect(f.e.r.observer == null);
    f.e.omit = null;
    var bad = tokens;
    bad[2] = 64;
    try std.testing.expectError(error.TokenOutOfRange, t.step(&bad, 3, .keys, turn));
}
test "Gemma4 learning: trainer budget refuses before allocation and failed completion retains every trainer resource" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const before = f.b.held;
    f.e.r.pockets.budget = 1;
    try std.testing.expectError(error.GemmaLearningBudget, T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e));
    try std.testing.expectEqual(before, f.b.held);
    f.e.r.pockets.budget = std.math.maxInt(u64);
    const t = try T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e);
    const held = f.b.held;
    f.b.fail_drain = true;
    t.deinit();
    try std.testing.expect(f.e.r.retained != null and f.e.r.failed);
    try std.testing.expectEqual(held, f.b.held);
    try std.testing.expectError(error.PlantedDrain, f.e.r.prepareClose());
    try std.testing.expectEqual(held, f.b.held);
    f.b.fail_drain = false;
    try f.e.r.prepareClose();
}
test "Gemma4 learning: shared learner's bad-token lesson refuses without publishing and releases training reservation" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const Slide = @import("learning_slide.zig").Slide(n.E);
    var learner = Slide.Learner.init(a, std.testing.io, .{ .e = f.e, .pockets = &f.e.r.pockets, .dir = f.path, .trainer = f.e });
    defer learner.deinit();
    const bad = [_]u32{ 3, 5, 7, 64, 9 };
    const examples = [_]Slide.Example{.{ .ids = &bad, .start = 3, .decide = 2, .subject = 1 }};
    try learner.begin(.{ .train = &examples, .held = &examples, .steps = 1 });
    var result = learner.step();
    var steps: usize = 1;
    while (!result.done and steps < 12) : (steps += 1) result = learner.step();
    try std.testing.expect(result.done and result.report != null);
    try std.testing.expectEqualStrings("TokenOutOfRange", result.report.?.failed);
    try std.testing.expectEqual(@as(u64, 0), f.e.r.pockets.mask());
}

test "Gemma4 learning: a constructor completion failure transfers the fully allocated trainer before unwinding" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    f.e.poison_open = true;
    try std.testing.expectError(error.PlantedDrain, T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e));
    try std.testing.expect(f.e.r.failed and f.e.r.retained != null and f.e.r.training_reserved > 0 and f.b.held > 0);
    try std.testing.expectError(error.PlantedDrain, f.e.r.prepareClose());
    f.b.fail_drain = false;
    try f.e.r.prepareClose();
}
