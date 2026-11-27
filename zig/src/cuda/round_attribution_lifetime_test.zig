//! Published event handles must survive eager work and launchable graphs after metadata allocation failure.
const std = @import("std");
const cuda = @import("cuda");
const Rig = @import("round_attribution_fake.zig").Rig;
const Fault = @import("round_attribution_fault.zig").Fault;
const t = std.testing;
fn capturedFailure(nth: usize, executable: bool) !void {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    x.begin();
    _ = rig.driver.api.cuStreamBeginCapture_v2(null, .thread_local);
    try rig.kernel();
    const first = x.pairs.items[0].begin;
    if (!executable) fault.arm(nth);
    var graph: cuda.abi.Graph = null;
    _ = rig.driver.api.cuStreamEndCapture(null, &graph);
    if (executable) fault.arm(nth);
    var exec: cuda.abi.GraphExec = null;
    _ = rig.driver.api.cuGraphInstantiateWithFlags(&exec, graph, 0);
    _ = rig.driver.api.cuGraphDestroy(graph);
    try t.expectEqual(@as(usize, 1), fault.failures);
    try t.expect(!x.pairs.items[0].free);
    // A still-launchable executable retains its event nodes after source-graph destruction.
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    const i = x.before(null, .{ .symbol = "attention", .op = .attention }).?;
    x.after(i, 0);
    try t.expect(first != x.pairs.items[i].begin);
    _ = rig.driver.api.cuStreamSynchronize(null);
    _ = rig.driver.api.cuGraphExecDestroy(exec);
    try t.expect(!x.pairs.items[0].free);
    try t.expectEqual(@as(usize, 0), rig.unsafeReuses());
}
test "attribution review: live graph handles are quarantined on group list map and executable OOM" {
    for (1..4) |nth| try capturedFailure(nth, false);
    try capturedFailure(1, true);
}
test "attribution review: eager pending capacity fails before publishing an event" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    x.begin();
    fault.arm(1);
    const counts = rig.eventCounts();
    try t.expect(x.before(null, .{ .symbol = "attention", .op = .attention }) == null);
    try t.expectEqual(counts, rig.eventCounts());
    try t.expectEqual(@as(usize, 0), x.pending.items.len);
}
test "attribution review: capture capacity fails before publishing an event" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    x.begin();
    _ = rig.driver.api.cuStreamBeginCapture_v2(null, .thread_local);
    fault.arm(1);
    const counts = rig.eventCounts();
    try t.expect(x.before(null, .{ .symbol = "attention", .op = .attention }) == null);
    try t.expectEqual(counts, rig.eventCounts());
    var graph: cuda.abi.Graph = null;
    _ = rig.driver.api.cuStreamEndCapture(null, &graph);
    _ = rig.driver.api.cuGraphDestroy(graph);
}

test "attribution review: eager event is not reused before its actual stream wait" {
    const rig = try Rig.init();
    defer rig.deinit();
    rig.timing.begin();
    try rig.kernel();
    try rig.kernel();
    try t.expectEqual(@as(usize, 2), rig.timing.pairs.items.len);
    try t.expectEqual(@as(usize, 0), rig.unsafeReuses());
    _ = rig.driver.api.cuStreamSynchronize(null);
    try rig.kernel();
    try t.expectEqual(@as(usize, 2), rig.timing.pairs.items.len);
    try t.expectEqual(@as(usize, 0), rig.unsafeReuses());
}
test "attribution review: same geometry affine graph mutation is refused" {
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.begin();
    const exec = try rig.graph();
    defer _ = rig.driver.api.cuGraphExecDestroy(exec);
    const g = x.execs.get(@intFromPtr(exec)).?;
    const pair = &x.pairs.items[g.pairs[0]];
    pair.desc.op = .affine_g64;
    var params: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
    params.func = @ptrFromInt(80);
    params.grid_x = pair.desc.grid[0];
    params.grid_y = pair.desc.grid[1];
    params.grid_z = pair.desc.grid[2];
    params.block_x = pair.desc.block[0];
    params.block_y = pair.desc.block[1];
    params.block_z = pair.desc.block[2];
    _ = rig.driver.api.cuGraphExecKernelNodeSetParams_v2(exec, null, &params);
    try t.expectEqualStrings("mutated graph projection or geometry needs fresh descriptors", x.problem.?);
    x.active = false;
    _ = rig.driver.api.cuGraphExecKernelNodeSetParams_v2(exec, null, &params);
    try t.expectEqualStrings("mutated graph projection or geometry needs fresh descriptors", x.persistent_problem.?);
}
test "attribution review: graph update registration OOM quarantines its launchable events" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    const x = rig.timing;
    x.a = fault.allocator();
    x.begin();
    _ = rig.driver.api.cuStreamBeginCapture_v2(null, .thread_local);
    try rig.kernel();
    var graph: cuda.abi.Graph = null;
    _ = rig.driver.api.cuStreamEndCapture(null, &graph);
    var info: cuda.abi.ExecUpdateResultInfo = undefined;
    fault.arm(1);
    const exec: cuda.abi.GraphExec = @ptrFromInt(400);
    _ = rig.driver.api.cuGraphExecUpdate_v2(exec, graph, &info);
    try t.expectEqual(@as(usize, 1), fault.failures);
    _ = rig.driver.api.cuGraphDestroy(graph);
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    try rig.kernel();
    try t.expect(!x.pairs.items[0].free);
    try t.expectEqual(@as(usize, 0), rig.unsafeReuses());
    _ = rig.driver.api.cuStreamSynchronize(null);
    _ = rig.driver.api.cuGraphExecDestroy(exec);
}
