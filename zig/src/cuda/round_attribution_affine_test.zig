//! Actual affine launch descriptors without loading a device or changing any arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const meta = cuda.round_attribution.meta;
const t = std.testing;
test "round attribution: actual MLX prompt and verify symbols decode affine-g64 projections" {
    var args: cuda.Args = .{};
    for (0..5) |_| args.add(@as(u64, 0));
    args.add(@as(c_int, 4096));
    args.add(@as(c_int, 17408));
    args.add(@as(c_int, 5120));
    args.add(@as(c_int, 17408));
    args.add(@as(c_int, 5120));
    args.add(@as(c_int, 9));
    const out = meta.kernel(cuda.qlinear.symbols.prefill22, .{ 4352, 1, 1 }, .{ 128, 1, 1 }, .{ 1, 1, 1 }, args.pointers(), null);
    try t.expectEqualStrings("affine_g64", @tagName(out.op));
    try t.expectEqual(@as(usize, 1), out.count);
    try t.expectEqual(@as(u32, 4096), out.projections[0].m);
    try t.expectEqual(@as(u32, 17408), out.projections[0].n);
    try t.expectEqualStrings("mlx_affine4_g64", out.projections[0].mode);
}

test "round attribution: MLX member widths and local K splits survive every verify width" {
    const linear = cuda.qlinear;
    for (1..129) |rows| {
        var parts: linear.Parts = std.mem.zeroes(linear.Parts);
        parts.count = 2;
        parts.p[0] = .{ .w = 0, .scales = 0, .biases = 0, .out = 0, .n = 48, .npad = 128, .sk = 4, .tiles = 1, .first = 0 };
        parts.p[1] = .{ .w = 0, .scales = 0, .biases = 0, .out = 0, .n = 6144, .npad = 6144, .sk = 1, .tiles = 96, .first = 1 };
        var args: cuda.Args = .{};
        args.add(@as(u64, 0));
        args.add(@as(u64, 0));
        args.add(parts);
        inline for (.{ @as(c_int, @intCast(rows)), @as(c_int, 5120), @as(c_int, 5120), @as(c_int, 1), @as(c_int, 4) }) |v| args.add(v);
        const symbol = if (rows <= 16) linear.symbols.group else if (rows <= 32) linear.symbols.verify32 else if (rows <= 64) linear.symbols.verify64 else linear.symbols.verify128;
        const out = meta.kernel(symbol, .{ 96, 1, 1 }, .{ 128, 1, 1 }, .{ 4, 1, 1 }, args.pointers(), null);
        try t.expectEqual(@as(usize, 2), out.count);
        try t.expectEqual(@as(u32, @intCast(rows)), out.projections[0].m);
        try t.expectEqual(@as(u32, 48), out.projections[0].n);
        try t.expectEqual(@as(u32, 6144), out.projections[1].n);
        try t.expectEqualStrings("cluster_k_order", out.projections[0].reduction);
        try t.expectEqualStrings("none", out.projections[1].reduction);
        try t.expectEqual([3]u32{ 4, 1, 1 }, out.projections[0].cluster);
        if (rows <= 16) {
            var many: linear.GemvManyParts = std.mem.zeroes(linear.GemvManyParts);
            many.count = 2;
            for (0..2) |i| many.p[i].p = .{ .w = 0, .scales = 0, .biases = 0, .out = 0, .n = parts.p[i].n, .npad = parts.p[i].npad, .sk = parts.p[i].sk, .tiles = 1 };
            var lane: cuda.Args = .{};
            lane.add(@as(u64, 0));
            lane.add(@as(u64, 0));
            lane.add(many);
            lane.add(@as(c_int, @intCast(rows)));
            lane.add(@as(c_int, 5120));
            lane.add(@as(c_int, 5120));
            const decoded = meta.kernel("tf_lane_gemv_many", .{ 96, 1, 1 }, .{ 128, 1, 1 }, .{ 1, 1, 1 }, lane.pointers(), null);
            try t.expectEqual(@as(usize, 2), decoded.count);
            try t.expectEqualStrings("cta_k_order", decoded.projections[0].reduction);
            try t.expectEqual(@as(u32, 4), decoded.projections[0].sk);
        }
    }
}

test "round attribution: MLX promptMany and single lane ABI and invalid descriptors fail closed" {
    const linear = cuda.qlinear;
    var parts: linear.PrefillParts = std.mem.zeroes(linear.PrefillParts);
    parts.count = 2;
    parts.p[0].n = 48;
    parts.p[0].npad = 128;
    parts.p[1].n = 6144;
    parts.p[1].npad = 6144;
    var args: cuda.Args = .{};
    args.add(@as(u64, 0));
    args.add(parts);
    args.add(@as(c_int, 4096));
    args.add(@as(c_int, 5120));
    const out = meta.kernel(linear.symbols.prefill_many, .{ 1568, 1, 1 }, .{ 256, 1, 1 }, .{ 1, 1, 1 }, args.pointers(), null);
    try t.expectEqual(@as(usize, 2), out.count);
    try t.expectEqual(@as(u32, 48), out.projections[0].n);
    try t.expectEqual(@as(u32, 6144), out.projections[1].n);
    var p: linear.GemvPart = .{ .w = 0, .scales = 0, .biases = 0, .out = 0, .n = 5120, .npad = 5120, .sk = 4, .tiles = 80 };
    for ([_]i32{ 5120, 5121, -64 }) |k| {
        var lane: cuda.Args = .{};
        lane.add(@as(u64, 0));
        lane.add(@as(u64, 0));
        lane.add(p);
        lane.add(@as(c_int, 8));
        lane.add(k);
        lane.add(k);
        const decoded = meta.kernel(linear.symbols.gemv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, .{ 1, 1, 1 }, lane.pointers(), null);
        try t.expectEqual(@as(usize, if (k == 5120) 1 else 0), decoded.count);
    }
    p.sk = 0;
    var invalid: cuda.Args = .{};
    invalid.add(@as(u64, 0));
    invalid.add(@as(u64, 0));
    invalid.add(p);
    invalid.add(@as(c_int, 8));
    invalid.add(@as(c_int, 5120));
    try t.expectEqual(@as(usize, 0), meta.kernel(linear.symbols.gemv, .{ 1, 1, 1 }, .{ 128, 1, 1 }, .{ 1, 1, 1 }, invalid.pointers(), null).count);
    try t.expectEqual(@as(usize, 0), meta.kernel("unknown_tf_qmm_prefill", .{ 1, 1, 1 }, .{ 1, 1, 1 }, .{ 1, 1, 1 }, null, null).count);
}

test "round attribution: MLX descriptors are copied through capture and attributed once per fused launch" {
    const rig = try @import("round_attribution_fake.zig").Rig.init();
    defer rig.deinit();
    var function: cuda.abi.Function = null;
    _ = rig.driver.api.cuModuleGetFunction(&function, null, "tf_lane_gemv_many");
    var parts: cuda.qlinear.GemvManyParts = std.mem.zeroes(cuda.qlinear.GemvManyParts);
    parts.count = 2;
    for (0..2) |i| parts.p[i].p = .{ .w = 0, .scales = 0, .biases = 0, .out = 0, .n = if (i == 0) 48 else 6144, .npad = if (i == 0) 128 else 6144, .sk = if (i == 0) 4 else 1, .tiles = 1 };
    var args: cuda.Args = .{};
    args.add(@as(u64, 0));
    args.add(@as(u64, 0));
    args.add(parts);
    args.add(@as(c_int, 8));
    args.add(@as(c_int, 5120));
    args.add(@as(c_int, 5120));
    _ = rig.driver.api.cuStreamBeginCapture_v2(null, .thread_local);
    _ = rig.driver.api.cuLaunchKernel(function, 96, 1, 1, 128, 1, 1, 35328, null, args.pointers(), null);
    var graph: cuda.abi.Graph = null;
    _ = rig.driver.api.cuStreamEndCapture(null, &graph);
    var exec: cuda.abi.GraphExec = null;
    _ = rig.driver.api.cuGraphInstantiateWithFlags(&exec, graph, 0);
    _ = rig.driver.api.cuGraphDestroy(graph);
    rig.timing.begin();
    const scope = cuda.round_attribution.scope(.forward);
    _ = rig.driver.api.cuGraphLaunch(exec, null);
    _ = rig.driver.api.cuStreamSynchronize(null);
    scope.end();
    try t.expect(rig.timing.problem == null);
    try t.expectEqual(@as(usize, 1), rig.timing.device.items.len);
    const event = rig.timing.device.items[0];
    try t.expectEqual(@as(usize, 2), event.affine_g64.len);
    try t.expectEqual(@as(usize, 0), event.qmmf.len);
    try t.expect(event.graph_replay != null);
    try t.expectEqual(@as(u32, 48), event.affine_g64[0].n);
    try t.expectEqual(@as(usize, 1), rig.waitCount());
    _ = rig.driver.api.cuGraphExecDestroy(exec);
}
