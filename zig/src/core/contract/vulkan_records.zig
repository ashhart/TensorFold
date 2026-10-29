//! Vulkan launch declarations and unmeasured served-device rows; HIP's record tables use this same audit shape.
const std = @import("std");
const limits = @import("limits.zig");

pub const Record = struct { launch: limits.VulkanLaunch, control_gate: ?u8 = null };

/// Live-device LIMITS (M1 Max, M1, M2 Max; Mesa 26.3 Honeykrisp; same values, a row a chip); run logs in the PR.
fn measuredRow(chip: []const u8) limits.VulkanDeviceLimits {
    return .{
        .chip = chip,
        .max_compute_work_group_invocations = 1024,
        .max_compute_work_group_size = .{ 1024, 1024, 64 },
        .max_compute_shared_memory_size = 32768,
        .max_storage_buffer_range = 2147483647,
        .min_storage_buffer_offset_alignment = 16,
        .subgroup_size = 32,
    };
}

pub const devices = [_]limits.VulkanDeviceLimits{
    measuredRow("G13C (M1 Max)"),
    measuredRow("G13G (M1)"),
    measuredRow("G14C (M2 Max)"),
};

pub const records = [_]Record{
    .{ .launch = .{ .name = "qmm_bf16", .local_size = .{ 256, 1, 1 } } },
    .{ .launch = .{ .name = "control:qmm_fmax_nonfinite", .local_size = .{ 256, 1, 1 } }, .control_gate = 1 },
    .{ .launch = .{ .name = "control:atomic_sum", .local_size = .{ 256, 1, 1 } }, .control_gate = 2 },
    .{ .launch = .{ .name = "control:scribble_input", .local_size = .{ 256, 1, 1 } }, .control_gate = 2 },
    .{ .launch = .{ .name = "control:drop_tail", .local_size = .{ 256, 1, 1 } }, .control_gate = 3 },
    .{ .launch = .{ .name = "control:fp16_accum", .local_size = .{ 256, 1, 1 } }, .control_gate = 4 },
    .{ .launch = .{ .name = "control:over_limit", .local_size = .{ 2048, 1, 1 } }, .control_gate = 5 },
};

/// Every row must be measured before records can be considered audited.
pub fn audit(row: limits.VulkanDeviceLimits) !usize {
    var issues: usize = 0;
    for (records) |record| {
        if (!try limits.vulkanFits(record.launch, row)) issues += 1;
    }
    return issues;
}

test "an unmeasured chip is refused, never treated as permissive" {
    try std.testing.expectError(error.Unmeasured, audit(.{ .chip = "not yet measured" }));
    var partial = measuredRow("partial");
    partial.subgroup_size = null;
    try std.testing.expectError(error.Unmeasured, audit(partial));
}

test "every served chip fits every real launch and rejects only the planted over-limit control" {
    for (devices) |device| try std.testing.expectEqual(@as(usize, 1), try audit(device));
    for (devices) |device| for (records) |record| {
        const fits = try limits.vulkanFits(record.launch, device);
        try std.testing.expectEqual(record.control_gate != 5, fits);
    };
}

test "the planted controls stay registered, one per gate, in gate order" {
    const expected_gates = [_]u8{ 1, 2, 2, 3, 4, 5 };
    var control_index: usize = 0;
    for (records) |record| if (record.control_gate) |gate| {
        try std.testing.expectEqual(expected_gates[control_index], gate);
        control_index += 1;
    };
    try std.testing.expectEqual(expected_gates.len, control_index);
}
