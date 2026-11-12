//! A keep that fails at any of its allocations, or after its encoder ends, returns the error instead of aborting.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const graph = @import("synthetic_graph.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const autorelease = mtl.objc.Pool.push();
    defer autorelease.pop();
    const model = try graph.createLayers(a, 5);
    defer model.deinit();
    var failing = std.testing.FailingAllocator.init(a, .{});
    var r = try q.decode_round.Runner.init(failing.allocator(), model, 1, 256);
    defer r.deinit();
    const s = q.session.Session{ .runner = &r };
    try r.reset(0);
    try s.prefill(&.{ 1, 2, 3, 4, 5 }, 16);
    var failed: usize = 0;
    var at: usize = 0;
    // each of the keep's allocations fails in turn: the encoder it opened must end before the error returns
    while (true) : (at += 1) {
        var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = &.{ 6, 7 } }}, @intCast(model.config.conv_kernel), 1, @intCast(model.config.vocab));
        defer round.deinit();
        try r.verifyHead(&round, .all);
        failing.fail_index = failing.alloc_index + at;
        const result = r.keep(&round, &.{&.{0}});
        failing.fail_index = std.math.maxInt(usize);
        result catch |err| {
            if (err != error.OutOfMemory) return err;
            failed += 1;
            try r.keep(&round, &.{&.{0}}); // the same round keeps once the allocation succeeds
            continue;
        };
        break;
    }
    if (failed == 0) return error.NoKeepAllocation;
    // a path the window refuses fails after the encoder has ended: the error returns, with no second end
    var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = &.{ 8, 9 } }}, @intCast(model.config.conv_kernel), 1, @intCast(model.config.vocab));
    defer round.deinit();
    try r.verifyHead(&round, .all);
    if (r.keep(&round, &.{&.{1}})) |_| return error.BadPathKept else |err| if (err != error.BadPath) return err;
    std.debug.print("PASS a keep failing at each of its {d} allocations returns OutOfMemory and keeps on retry; a refused path after the encoder ends returns BadPath\n", .{failed});
}
