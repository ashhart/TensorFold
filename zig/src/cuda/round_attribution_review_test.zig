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
test "attribution review: pending OOM leaves the pair recoverable" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    try beginRound(rig);
    try x.host.ensureTotalCapacity(x.a, 8);
    const index = x.before(null, .{ .symbol = "attention", .op = .attention }).?;
    fault.arm(1);
    x.after(index, 0);
    try t.expectEqual(@as(usize, 1), fault.failures);
    try t.expectEqualStrings("pending metadata allocation", x.problem.?);
    try t.expectEqual(@as(usize, 0), x.pending.items.len);
    try t.expect(!x.pairs.items[index].inflight);
    try t.expect(x.pairs.items[index].free);
    x.finish(false);
    try cleanRound(rig);
    try t.expectEqual(@as(usize, 1), x.pairs.items.len);
}
