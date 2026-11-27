//! Review regressions execute the actual trace and wrapped driver against recording CUDA calls.
const std = @import("std");
const cuda = @import("cuda");
const trace = cuda.round_attribution;
const Rig = @import("round_attribution_fake.zig").Rig;
const Fault = @import("round_attribution_fault.zig").Fault;
const t = std.testing;
test "attribution review: cached function names do not grow the permanent arena" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    rig.timing.permanent.deinit();
    rig.timing.permanent = .init(fault.allocator());
    var function: cuda.abi.Function = null;
    const name = "attention_repeated_function_name_long_enough_to_cross_an_arena_block";
    try t.expectEqual(@as(cuda.abi.Result, 0), rig.driver.api.cuModuleGetFunction(&function, null, name));
    const saved = rig.timing.name(function);
    const calls = fault.calls;
    for (0..5000) |_| {
        try t.expectEqual(@as(cuda.abi.Result, 0), rig.driver.api.cuModuleGetFunction(&function, null, name));
    }
    try t.expectEqualStrings(name, rig.timing.name(function));
    try t.expectEqual(calls, fault.calls);
    try t.expectEqual(saved.ptr, rig.timing.name(function).ptr);
    rig.timing.begin();
    try rig.kernel();
    _ = rig.driver.api.cuStreamSynchronize(null);
    try t.expectEqualStrings(name, rig.timing.device.items[0].symbol);
}

const id: [64]u8 = @splat('a');
fn beginRound(rig: Rig) !void {
    const x = rig.timing;
    x.identity = &id;
    x.begin();
    try x.requests.append(x.a, .{ .stream = "s", .prompt_sha256 = &id, .prompt_tokens = 1, .sampling_json = "{}", .cached = 0 });
    try x.frames.append(x.a, .{ .correlation = x.root_scope, .live_rows = 1, .windows = 1, .shape_bucket = 1, .shape_key = 1, .mt2 = false });
    try x.windows.append(x.a, .{ .stream = "s", .kind = "head", .offered_rows = 1, .proposal_nodes = 0, .connected_accepted_depth = 0, .kept_path = &.{0}, .committed_ids = &.{7} });
}
fn cleanRound(rig: Rig) !void {
    try beginRound(rig);
    const scope = trace.scope(.forward);
    try rig.kernel();
    _ = rig.driver.api.cuStreamSynchronize(null);
    scope.end();
    rig.timing.finish(false);
    try t.expect(rig.timing.problem == null);
    try t.expect(!rig.timing.active);
    try t.expectEqual(@as(usize, 0), rig.timing.pending.items.len);
    try t.expectEqual(@as(usize, 1), rig.timing.device.items.len);
    for (rig.timing.pairs.items) |p| try t.expect(!p.inflight and !p.owned and p.free);
}
test "attribution review: projection OOM still drains and recycles pending pairs" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.round_arena.deinit();
    x.round_arena = .init(fault.allocator());
    const desc: trace.meta.Meta = .{ .symbol = "qmmf", .op = .qmmf, .count = 1, .projections = .{ .{ .m = 1, .n = 128, .k = 128, .sk = 1, .cluster = .{ 1, 1, 1 }, .reduce = false, .mode = "nvfp4" }, undefined, undefined, undefined } };
    try beginRound(rig);
    const index = x.before(null, desc).?;
    x.after(index, 0);
    fault.arm(1);
    _ = rig.driver.api.cuStreamSynchronize(null);
    try t.expectEqual(@as(usize, 1), fault.failures);
    try t.expectEqualStrings("projection metadata allocation", x.problem.?);
    try t.expectEqual(@as(usize, 0), x.pending.items.len);
    try t.expect(!x.pairs.items[index].inflight and x.pairs.items[index].free);
    x.finish(false);
    try beginRound(rig);
    x.after(x.before(null, desc), 0);
    _ = rig.driver.api.cuStreamSynchronize(null);
    try t.expectEqual(@as(usize, 1), x.device.items[0].qmmf.len);
    x.finish(false);
    try t.expect(x.problem == null);
    try t.expectEqual(@as(usize, 1), x.pairs.items.len);
    try t.expect(x.pairs.items[index].free);
}

test "attribution review: pool recycling does not allocate or lose a free slot" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    try beginRound(rig);
    try rig.kernel();
    try x.device.ensureTotalCapacity(x.a, 8);
    const calls = fault.calls;
    fault.arm(1);
    x.collect(null);
    try t.expectEqual(@as(usize, 1), x.free.items.len);
    try t.expectEqual(calls, fault.calls);
    fault.remaining = null;
    try t.expect(x.problem == null);
    x.finish(false);
    try cleanRound(rig);
    try t.expectEqual(@as(usize, 1), x.pairs.items.len);
}
test "attribution review: pool reservation OOM destroys the newly created events" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    try beginRound(rig);
    try x.pairs.ensureTotalCapacity(x.a, 1);
    try x.pending.ensureTotalCapacity(x.a, 1);
    const counts = rig.eventCounts();
    fault.arm(1);
    try t.expect(x.before(null, .{ .symbol = "attention", .op = .attention }) == null);
    try t.expectEqual(@as(usize, 1), fault.failures);
    try t.expectEqualStrings("event pool metadata allocation", x.problem.?);
    try t.expectEqual(counts[0] + 2, rig.eventCounts()[0]);
    try t.expectEqual(counts[1] + 2, rig.eventCounts()[1]);
    try t.expectEqual(@as(usize, 0), x.pairs.items.len);
    x.finish(false);
    try cleanRound(rig);
}

comptime {
    _ = @import("round_attribution_lifetime_test.zig");
}
