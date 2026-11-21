const std = @import("std");
const deployment = @import("deployment_fit.zig");

const checkpoint: [32]u8 = @splat(9);

fn model() deployment.Model {
    return .{ .checkpoint = checkpoint, .max_context = 4096, .max_streams = 8, .stream_fixed = 10, .stream_token = 2 };
}

fn placements() [2]deployment.Placement {
    return .{
        .{ .name = "resident", .checkpoint = checkpoint, .resident = 6000, .scratch = 500 },
        .{ .name = "ssd", .checkpoint = checkpoint, .resident = 3500, .scratch = 500, .expert_pool = 300, .source_disk = 4000 },
    };
}

test "deployment fit preserves full service across exact SSD fallback and emits native controls" {
    const host: deployment.Host = .{ .memory_limit = 8000, .memory_reserve = 500, .disk_free = 5000, .disk_floor = 500 };
    const request: deployment.Request = .{ .context = 512, .streams = 2, .exact_context = true, .fixed_streams = true };
    const selected = try deployment.fit(host, model(), &placements(), request);
    try std.testing.expectEqualStrings("ssd", selected.placement.name);
    try std.testing.expect(selected.full_service);
    try std.testing.expectEqual(@as(u32, 512), selected.context);
    try std.testing.expectEqual(@as(u32, 2), selected.streams);
    try std.testing.expectEqual(@as(u64, 3500 + 500 + 2 * (10 + 2 * 512)), selected.memory_used);

    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try selected.describe(&out.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const receipt = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 300), receipt.get("ssd_experts_bytes").?.integer);
    try std.testing.expectEqual(@as(i64, 4000), receipt.get("source_disk_bytes").?.integer);
    try std.testing.expect(receipt.get("full_service").?.bool);
}

test "deployment fit searches best context then streams before resident preference" {
    const host: deployment.Host = .{ .memory_limit = 5000, .memory_reserve = 100, .disk_free = 0 };
    var candidates = [_]deployment.Placement{.{ .name = "resident", .checkpoint = checkpoint, .resident = 2500, .scratch = 500 }};
    const selected = try deployment.fit(host, model(), &candidates, .{ .context = 1024, .streams = 4 });
    try std.testing.expectEqualStrings("resident", selected.placement.name);
    try std.testing.expectEqual(@as(u32, 768), selected.context);
    try std.testing.expectEqual(@as(u32, 1), selected.streams);
    try std.testing.expect(!selected.full_service);

    const exact = try deployment.fit(host, model(), &candidates, .{ .context = 768, .streams = 4, .exact_context = true });
    try std.testing.expectEqual(@as(u32, 1), exact.streams);
    const fixed = try deployment.fit(host, model(), &candidates, .{ .context = 1024, .streams = 2, .fixed_streams = true });
    try std.testing.expectEqual(@as(u32, 256), fixed.context);
}

test "deployment fit skips unavailable device and SSD placements without changing tensor identity" {
    var candidates = placements();
    candidates[1].available = false;
    const host: deployment.Host = .{ .memory_limit = 8000, .memory_reserve = 500, .disk_free = 5000, .disk_floor = 500 };
    const selected = try deployment.fit(host, model(), &candidates, .{ .context = 512, .streams = 2 });
    try std.testing.expectEqualStrings("resident", selected.placement.name);
    try std.testing.expectEqual(@as(u32, 256), selected.context);
    candidates[0].available = false;
    try std.testing.expectError(error.NoDeploymentFit, deployment.fit(host, model(), &candidates, .{ .context = 512, .streams = 2 }));

    candidates = placements();
    candidates[1].checkpoint[0] +%= 1;
    try std.testing.expectError(error.CheckpointMismatch, deployment.fit(host, model(), &candidates, .{ .context = 512, .streams = 2 }));
}

test "deployment fit enforces disk floor, exact dimensions and bounded arithmetic" {
    var host: deployment.Host = .{ .memory_limit = 8000, .memory_reserve = 500, .disk_free = 4499, .disk_floor = 500 };
    var candidates = [_]deployment.Placement{placements()[1]};
    try std.testing.expectError(error.NoDeploymentFit, deployment.fit(host, model(), &candidates, .{ .context = 512, .streams = 2, .exact_context = true, .fixed_streams = true }));
    host.disk_free = 4500;
    _ = try deployment.fit(host, model(), &candidates, .{ .context = 512, .streams = 2, .exact_context = true, .fixed_streams = true });

    var huge = model();
    huge.stream_token = std.math.maxInt(u64);
    try std.testing.expectError(error.NoDeploymentFit, deployment.fit(host, huge, &candidates, .{ .context = 512, .streams = 2 }));
    var wide = model();
    wide.max_context = 131073;
    wide.max_streams = 128;
    try std.testing.expectError(error.FitCensusTooLarge, deployment.fit(host, wide, &placements(), .{ .context = 131073, .streams = 128 }));
}

test "deployment fit refuses malformed geometry and placement ledgers" {
    var bad = model();
    bad.max_context = 0;
    try std.testing.expectError(error.BadModelGeometry, deployment.fit(.{ .memory_limit = 1, .memory_reserve = 0 }, bad, &placements(), .{ .context = 1, .streams = 1 }));
    var candidates = placements();
    candidates[1].expert_pool = 0;
    try std.testing.expectError(error.BadPlacement, deployment.fit(.{ .memory_limit = 8000, .memory_reserve = 0 }, model(), &candidates, .{ .context = 512, .streams = 2 }));
}
