//! Host fixtures for complete-round attribution; no driver library or device is opened.
const std = @import("std");
const cuda = @import("cuda");
const trace = cuda.round_attribution;
const m = trace.model;
test {
    _ = @import("round_attribution_affine_test.zig");
}
const t = std.testing;
test "round attribution: overlap is counted once and kernel sums are not round time" {
    const ds = [_]m.Device{
        .{ .correlation = 1, .stream = 1, .phase = .forward, .op = .qmmf, .symbol = "projection", .begin_ns = 10, .end_ns = 50 },
        .{ .correlation = 2, .stream = 1, .phase = .forward, .op = .attention, .symbol = "attention", .begin_ns = 30, .end_ns = 60 },
        .{ .correlation = 3, .stream = 1, .phase = .token_readback, .op = .d2h, .symbol = "readback", .begin_ns = 80, .end_ns = 90 },
    };
    const hs = [_]m.Host{
        .{ .correlation = 4, .phase = .land, .kind = .wait, .begin_ns = 20, .end_ns = 60 },
        .{ .correlation = 5, .phase = .candidate_frame, .kind = .wait, .begin_ns = 40, .end_ns = 70 },
        .{ .correlation = 6, .phase = .state_publication, .kind = .span, .begin_ns = 75, .end_ns = 85 },
        .{ .correlation = 7, .phase = .other, .kind = .attribution_drain, .begin_ns = 95, .end_ns = 110 },
    };
    const b = try m.budgets(t.allocator, 120, 95, 100, &ds, &hs);
    try t.expectEqual(@as(u64, 120), b.complete_round_wall_ns);
    try t.expectEqual(@as(u64, 60), b.device_event_span_union_ns);
    try t.expectEqual(@as(u64, 70), b.kernel_event_span_sum_ns);
    try t.expectEqual(@as(u64, 40), b.device_envelope_gap_ns);
    try t.expectEqual(@as(u64, 50), b.host_wait_union_ns);
    try t.expectEqual(@as(u64, 10), b.state_publication_union_ns);
    try t.expectEqual(@as(u64, 15), b.attribution_drain_union_ns);
    try t.expectEqual(@as(u64, 40), b.device_by_operation_union_ns[@backingInt(m.Op.qmmf)]);
    try t.expectEqual(@as(u64, 50), b.device_by_phase_union_ns[@backingInt(m.Phase.forward)]);
}
test "round attribution: invalid intervals and duplicate launches are refused" {
    try t.expectEqual(@as(u64, 0), try m.unionNs(t.allocator, &.{}));
    try t.expectEqual(@as(u64, 8), try m.unionNs(t.allocator, &.{ .{ .begin = 5, .end = 9 }, .{ .begin = 1, .end = 5 } }));
    try t.expectError(error.BackwardsInterval, m.unionNs(t.allocator, &.{.{ .begin = 2, .end = 1 }}));
    try t.expectError(error.InvalidRoundBoundary, m.budgets(t.allocator, 4, 5, 0, &.{}, &.{}));
    var ds: [2]m.Device = @splat(.{ .correlation = 1, .stream = 1, .phase = .forward, .op = .other, .symbol = "x", .begin_ns = 1, .end_ns = 5 });
    try t.expectError(error.DuplicateCorrelation, m.budgets(t.allocator, 10, 8, 5, &ds, &.{}));
    ds[1].correlation = 2;
    ds[1].end_ns = 6;
    try t.expectError(error.InvalidDeviceInterval, m.budgets(t.allocator, 10, 8, 5, &ds, &.{}));
}
test "round attribution: a receipt preserves committed IDs and connected accepted depth" {
    const windows = [_]m.Window{.{ .stream = "stream", .kind = "draft", .offered_rows = 5, .proposal_nodes = 4, .rolled_back_rows = 2, .connected_accepted_depth = 2, .kept_path = &.{ 0, 2, 4 }, .committed_ids = &.{ 101, 37 } }};
    const id: [64]u8 = @splat('a');
    var r: m.Record = .{ .round = 1, .identity = &id, .complete = true, .problem = null, .requests = &.{.{ .stream = "stream", .prompt_sha256 = &id, .prompt_tokens = 8, .sampling_json = "{}", .cached = 8 }}, .frames = &.{.{ .correlation = 1, .live_rows = 5, .windows = 1, .shape_bucket = 8, .shape_key = 42, .mt2 = true }}, .windows = &windows, .device = &.{.{ .correlation = 3, .stream = 1, .phase = .forward, .op = .other, .symbol = "fixture", .begin_ns = 0, .end_ns = 0 }}, .host = &.{}, .budget = try m.budgets(t.allocator, 100, 90, 0, &.{}, &.{}) };
    const json = try m.json(t.allocator, r);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(m.Record, t.allocator, json, .{});
    defer parsed.deinit();
    try t.expectEqualSlices(u32, &.{ 101, 37 }, parsed.value.windows[0].committed_ids);
    try t.expectEqual(@as(usize, 2), parsed.value.windows[0].connected_accepted_depth);
    r.windows = &.{};
    try t.expectError(error.IncompleteRound, m.validate(r));
    r.windows = &windows;
    r.problem = "lost graph";
    try t.expectError(error.IncompleteRound, m.validate(r));
}
test "round attribution: disabled build returns before driver or allocator access" {
    var d: cuda.Driver = undefined;
    try t.expect((try trace.Trace.install(std.testing.failing_allocator, t.io, &d)) == null);
    try t.expect(trace.owner == null);
}
test "round attribution: capture retains event ownership across graph destruction and repeated replay" {
    const rig = try @import("round_attribution_fake.zig").Rig.init();
    defer rig.deinit();
    const exec = try rig.graph();
    const n = rig.timing.pairs.items.len;
    try t.expectEqual(@as(usize, 1), n);
    try t.expectEqual(@as(usize, 0), rig.launchCount());
    rig.timing.begin();
    const scope = trace.scope(.forward);
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    _ = rig.driver.api.cuStreamSynchronize(null);
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    _ = rig.driver.api.cuStreamSynchronize(null);
    scope.end();
    try t.expect(rig.timing.problem == null);
    try t.expectEqual(@as(usize, 2), rig.timing.device.items.len);
    try t.expect(rig.timing.device.items[0].graph_replay != rig.timing.device.items[1].graph_replay);
    try t.expectEqual(scope.correlation, rig.timing.device.items[0].parent_scope);
    try t.expectEqual(@as(usize, 2), rig.waitCount());
    try t.expectEqual(n, rig.timing.pairs.items.len);
    _ = rig.driver.api.cuGraphExecDestroy(exec);
    try t.expectEqual(n, rig.timing.free.items.len);
}
test "round attribution: uncollected graph reuse and unknown graphs invalidate a receipt" {
    const rig = try @import("round_attribution_fake.zig").Rig.init();
    defer rig.deinit();
    const exec = try rig.graph();
    rig.timing.begin();
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    try t.expectEqualStrings("event reused before existing wait collected it", rig.timing.problem.?);
    _ = rig.driver.api.cuStreamSynchronize(null);
    rig.timing.begin();
    _ = rig.driver.api.cuGraphLaunch(@ptrFromInt(600), null);
    try t.expectEqualStrings("untracked graph replay", rig.timing.problem.?);
    _ = rig.driver.api.cuGraphExecDestroy(exec);
}
test "round attribution: eager launch joins its host submission and releases events at an existing wait" {
    const rig = try @import("round_attribution_fake.zig").Rig.init();
    defer rig.deinit();
    rig.timing.begin();
    const scope = trace.scope(.proposal_hold);
    try rig.kernel();
    _ = rig.driver.api.cuStreamSynchronize(null);
    scope.end();
    try t.expect(rig.timing.problem == null);
    try t.expectEqual(@as(usize, 1), rig.launchCount());
    try t.expectEqual(@as(usize, 1), rig.waitCount());
    const d = rig.timing.device.items[0];
    try t.expectEqual(m.Phase.proposal_hold, d.phase);
    try t.expectEqual(d.correlation, rig.timing.host.items[0].correlation);
    try t.expectEqual(scope.correlation, d.parent_scope);
    try t.expectEqual(@as(usize, 1), rig.timing.free.items.len);
}
test "round attribution: actual QMMF argument layouts, split-K, clusters and reduction are retained" {
    var args: cuda.Args = .{};
    inline for (.{ @as(u64, 1), @as(u64, 2), @as(u64, 3), @as(f32, 1), @as(u64, 4), @as(u64, 99), @as(c_int, 8), @as(c_int, 17408), @as(c_int, 5120), @as(c_int, 4) }) |v| args.add(v);
    const q = trace.meta.kernel("_ZN7tf_qmmf11qmmf_kernelILi0ELi16", .{ 136, 4, 1 }, .{ 128, 1, 1 }, .{ 4, 1, 1 }, args.pointers(), null);
    try t.expectEqual(@as(usize, 1), q.count);
    const p = q.projections[0];
    try t.expectEqual(@as(u32, 17408), p.n);
    try t.expectEqual(@as(u32, 5120), p.k);
    try t.expectEqual(@as(u32, 4), p.sk);
    try t.expectEqual([3]u32{ 4, 1, 1 }, p.cluster);
    try t.expect(p.reduce);
    try t.expectEqualStrings("nvfp4", p.mode);
    const reduced = trace.meta.kernel("_ZN7tf_qmmf13reduce_kernel", .{ 1, 1, 1 }, .{ 256, 1, 1 }, .{ 1, 1, 1 }, args.pointers(), q);
    try t.expectEqual(p, reduced.projections[0]);
    var segs: cuda.qmmf_group.Segs = .{};
    segs.count = 2;
    segs.s[0].n = 5120;
    segs.s[0].sk = 2;
    segs.s[1].n = 1024;
    var grouped: cuda.Args = .{};
    grouped.add(@as(u64, 1));
    grouped.add(segs);
    grouped.add(@as(c_int, 8));
    grouped.add(@as(c_int, 5120));
    const many = trace.meta.kernel("tf_qmmf_group_bf16", .{ 96, 1, 1 }, .{ 128, 1, 1 }, .{ 2, 1, 1 }, grouped.pointers(), null);
    try t.expectEqual(@as(usize, 2), many.count);
    try t.expectEqual(@as(u32, 1024), many.projections[1].n);
    try t.expectEqualStrings("bf16", many.projections[0].mode);
    const unknown = trace.meta.kernel("unknown_qmmf", .{ 1, 1, 1 }, .{ 1, 1, 1 }, .{ 1, 1, 1 }, args.pointers(), null);
    try t.expectEqual(@as(usize, 0), unknown.count);
}
test "round attribution: source attestation is a full commit and tree" {
    const hex: [40]u8 = @splat('a');
    const id = hex ++ ":" ++ hex;
    try t.expect(trace.sourceValid(id));
    try t.expect(!trace.sourceValid("main"));
    var bad = id.*;
    bad[0] = 'z';
    try t.expect(!trace.sourceValid(&bad));
}

test "round attribution: round entry binds its CUDA context before recording the origin" {
    const rig = try @import("round_attribution_fake.zig").Rig.init();
    defer rig.deinit();
    rig.unbind();
    try t.expect(!rig.bound());
    rig.timing.begin();
    try t.expect(rig.bound());
}

extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
test "round attribution: traced policy must replay untraced costs before any allocation" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    _ = setenv("TF_CUDA_ROUND_TRACE", "1", 1);
    defer _ = setenv("TF_CUDA_ROUND_TRACE", "0", 1);
    _ = unsetenv("TF_CUDA_ROUND_POLICY_REPLAY");
    var driver: cuda.Driver = undefined;
    try t.expectError(error.AttributionNeedsUntracedPolicyReplay, trace.Trace.install(std.testing.failing_allocator, t.io, &driver));
}
