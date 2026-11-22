//! The real pool round and shared keep encode against a recording device with byte sentinels.
const std = @import("std");
const metal = @import("metal");
const gpu = @import("families/qwen27/gdn_pool_gpu.zig");
const abi = @import("families/qwen27/gdn_contract.zig");
const gdn = @import("families/qwen27/gdn.zig");
const Plan = @import("families/qwen27/gdn_plan.zig").Plan;
const Window = @import("families/qwen27/state.zig").Window;
const Config = @import("families/qwen27/config.zig").Config;

test {
    _ = @import("families/qwen27/gdn_test.zig");
    _ = @import("families/qwen27/attention_plan.zig");
    _ = @import("families/qwen27/attention_test.zig");
    _ = @import("families/qwen27/gdn_conv_inplace_test.zig");
}

test "qwen27 speed: shared in-place verify then empty and full keeps restore state and retain conv histories" {
    const a = std.testing.allocator;
    var c: Config = undefined;
    c.k_heads = 1;
    c.v_heads = 1;
    c.dk = 128;
    c.dv = 128;
    c.conv_kernel = 2;
    c.layers = 1;
    c.eps = 1e-6;
    const shape = try abi.Shape.init(c);
    const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
    var window = try Window.init(a, 9, &.{-1}, 2, 64);
    defer window.deinit();
    for ([_]bool{ false, true }) |empty_second| {
        var pool: gpu.Pool = .{
            .budget = try @import("families/qwen27/gdn_pool.zig").Budget.init(shape, 1, 2, 2),
            .buffers = undefined,
            .kernels = undefined,
            .scratch = gpu.Pool.emptyScratch(a),
        };
        for (&pool.buffers, 0..) |*b, i| {
            b.* = .{ .id = i + 1, .data = try a.alignedAlloc(u8, .@"16", 2 * shape.state * 4) };
            @memset(b.data, 0);
        }
        defer for (pool.buffers) |b| a.free(b.data);
        pool.kernels = undefined;
        pool.kernels.pipelines = .{ .pre = .{ .id = 0 }, .tree = .{ .id = 1 }, .replay = .{ .id = 2 }, .post = .{ .id = 3 }, .conv_commit = .{ .id = 4 }, .tree_slots = .{ .{ .id = 10 }, .{ .id = 11 }, .{ .id = 12 }, .{ .id = 13 }, .{ .id = 14 } } };
        pool.kernels.publish_conv = .{ .id = 8 };
        pool.kernels.conv_in_place = .{ .id = 9 };
        const bytes = 2 * shape.state * 4;
        var metadata: gpu.Metadata = undefined;
        inline for (@typeInfo(gpu.Metadata).@"struct".field_names, 0..) |field, i| {
            const buffer = metal.Buffer{ .id = 100 + i, .data = try a.alignedAlloc(u8, .@"16", bytes) };
            @field(metadata, field) = .{ .buffer = buffer };
        }
        defer inline for (@typeInfo(gpu.Metadata).@"struct".field_names) |field| a.free(@field(metadata, field).buffer.data);
        const state = try pool.committed(0, 0, true);
        @memset(@constCast(state), 0x12);
        const other = try pool.committed(0, 1, true);
        @memset(@constCast(other), 0x34);
        const original = try a.dupe(u8, state);
        defer a.free(original);
        const second = try a.dupe(u8, other);
        defer a.free(second);
        const history = try pool.committed(0, 0, false);
        @memset(@constCast(history), 0x56);
        const other_history = try pool.committed(0, 1, false);
        @memset(@constCast(other_history), 0x78);
        const old_history = try a.dupe(u8, history);
        defer a.free(old_history);
        var plan = try Plan.init(a, try shape.params(2, 2, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} },
            .{ .window = window, .state_slot = 1, .next_slot = 1, .path = &.{0} },
        });
        defer plan.deinit();
        const scratch = metal.Buffer{ .id = 999, .data = try a.alignedAlloc(u8, .@"16", bytes) };
        defer a.free(scratch.data);
        const output = metal.Buffer{ .id = 1000, .data = try a.alignedAlloc(u8, .@"16", bytes) };
        defer a.free(output.data);
        const span = gdn.Span{ .buffer = scratch, .bytes = scratch.length() };
        const refs = gpu.Refs{ .z = .{ .buffer = scratch }, .a = .{ .buffer = scratch }, .b = .{ .buffer = scratch }, .output = .{ .buffer = output }, .scalars = .{ .conv = span, .a_log = span, .dt = span, .norm = span, .kinds = kinds } };
        var record: metal.Recorder = .{};
        const e = metal.ComputeEncoder{ .record = &record };
        var allocations: metal.Allocations = .{};
        defer pool.deinitScratch();
        try pool.prepare(.{ .allocations = &allocations }, plan, false);
        try pool.forward(0, .{ .host = plan, .metadata = metadata }, refs, e);
        try std.testing.expect(!std.mem.eql(u8, original, state));
        var keep = try Plan.init(a, try shape.params(2, 2, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{} },
            .{ .window = window, .state_slot = 1, .next_slot = 1, .path = if (empty_second) &.{} else &.{0} },
        });
        defer keep.deinit();
        const saved_state = try a.dupe(u8, state);
        defer a.free(saved_state);
        const saved_other = try a.dupe(u8, other);
        defer a.free(saved_other);
        const saved_history = try a.dupe(u8, history);
        defer a.free(saved_history);
        const saved_other_history = try a.dupe(u8, other_history);
        defer a.free(saved_other_history);
        pool.conv_in_place = true;
        const before = record;
        try (gpu.DevicePlan{ .host = keep, .metadata = metadata }).metadata.upload(keep);
        try pool.commitLayers(.{ .host = keep, .metadata = metadata }, e);
        try std.testing.expectEqual(before.launches[8], record.launches[8]);
        try std.testing.expectEqual(before.launches[9] + 1, record.launches[9]);
        try std.testing.expectEqual(before.barriers + 1, record.barriers);
        try std.testing.expectEqual(@as(u64, 1), pool.conv_in_place_commits);
        const on_state = try a.dupe(u8, state);
        defer a.free(on_state);
        const on_other = try a.dupe(u8, other);
        defer a.free(on_other);
        const on_history = try a.dupe(u8, history);
        defer a.free(on_history);
        const on_other_history = try a.dupe(u8, other_history);
        defer a.free(on_other_history);
        @memcpy(@constCast(state), saved_state);
        @memcpy(@constCast(other), saved_other);
        @memcpy(@constCast(history), saved_history);
        @memcpy(@constCast(other_history), saved_other_history);
        record.restores = before.restores;
        pool.conv_in_place = false;
        try pool.commit(0, .{ .host = keep, .metadata = metadata }, e);
        try std.testing.expectEqualSlices(u8, on_state, state);
        try std.testing.expectEqualSlices(u8, on_other, other);
        try std.testing.expectEqualSlices(u8, on_history, history);
        try std.testing.expectEqualSlices(u8, on_other_history, other_history);
        try std.testing.expectEqualSlices(u8, original, state);
        try std.testing.expectEqual(empty_second, std.mem.eql(u8, second, other));
        try std.testing.expectEqualSlices(u8, old_history, history);
        for (other_history) |byte| try std.testing.expectEqual(@as(u8, if (empty_second) 0x78 else 0), byte);
        try std.testing.expectEqual(@as(u32, 9), (try window.keep(&.{})).length);
        try std.testing.expectEqual(@as(u32, 10), (try window.keep(&.{0})).length);
        try std.testing.expectEqual(@as(usize, if (empty_second) 2 else 1), record.restores);
    }
}
