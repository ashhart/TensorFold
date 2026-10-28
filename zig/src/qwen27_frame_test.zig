//! Recording allocations exercise the actual frame's switching, stable outputs and cleanup.
const std = @import("std");
const metal = @import("metal");
const gpu = @import("families/qwen27/gpu_frame.zig");
const Config = @import("families/qwen27/config.zig").Config;

fn config() Config {
    return .{ .hidden = 64, .intermediate = 128, .layers = 2, .vocab = 8, .heads = 2, .kv_heads = 1, .head_dim = 32, .k_heads = 1, .v_heads = 1, .dk = 128, .dv = 128, .conv_kernel = 2, .eps = 1e-6, .rope_dims = 32, .rope_theta = 1e4 };
}

test "qwen27 bank: narrow rounds bind narrow activations without changing the wide prompt or output owners" {
    var allocations: metal.Allocations = .{};
    var frame = try gpu.Frame.init(.{ .allocations = &allocations }, config(), 2048);
    defer frame.deinit();
    const wide = frame.get(.mixed).buffer;
    const logits = frame.get(.logits).buffer.id;
    const positions = frame.get(.positions).buffer.id;
    const ids: [2048]u32 = @splat(0);
    const pos: [2048]i32 = @splat(0);
    const calls = allocations.calls;
    var narrow: ?usize = null;
    for ([_]usize{ 1, 512, 513, 2048, 128, 2048, 1 }) |rows| {
        try frame.setRows(ids[0..rows], pos[0..rows]);
        const buffer = frame.get(.mixed).buffer;
        const wanted = gpu.allocationSizes(config(), if (rows <= 512) 512 else 2048);
        try std.testing.expectEqual(wanted[@backingInt(gpu.Field.mixed)], buffer.length());
        if (rows <= 512) {
            try std.testing.expect(buffer.id != wide.id);
            if (narrow) |id| try std.testing.expectEqual(id, buffer.id) else narrow = buffer.id;
        } else try std.testing.expectEqual(wide.id, buffer.id);
        try std.testing.expectEqual(logits, frame.get(.logits).buffer.id);
        try std.testing.expectEqual(positions, frame.get(.positions).buffer.id);
        try std.testing.expectEqual(@as(u32, 2048), frame.capacity);
        try std.testing.expectEqual(@as(i32, @intCast(rows)), frame.get(.dims).buffer.slice(i32, 8)[0]);
        try std.testing.expectEqual(calls, allocations.calls);
    }
    const last = frame.get(.mixed).buffer.id;
    try std.testing.expectError(error.BadFrameRows, frame.setRows(&.{}, &.{}));
    try std.testing.expectError(error.BadFrameRows, frame.setRows(&.{0}, &.{}));
    try std.testing.expectEqual(last, frame.get(.mixed).buffer.id);
}

test "qwen27 bank: both activation banks are inventoried once and every failed allocation unwinds" {
    var count: usize = 0;
    {
        var allocations: metal.Allocations = .{};
        var frame = try gpu.Frame.init(.{ .allocations = &allocations }, config(), 2048);
        var list: std.ArrayList(metal.Buffer) = .empty;
        defer list.deinit(std.testing.allocator);
        try frame.appendBuffers(std.testing.allocator, &list);
        var total: usize = 0;
        var wired: usize = 0;
        for (list.items, 0..) |buffer, i| {
            for (list.items[0..i]) |prior| try std.testing.expect(prior.id != buffer.id);
            total += buffer.length();
            wired += std.mem.alignForward(usize, @max(buffer.length(), 16), 16384);
        }
        try std.testing.expectEqual(allocations.bytes, total);
        try std.testing.expectEqual(@import("families/qwen27/prompt_memory.zig").frameBytes(config(), 2048), wired);
        count = allocations.calls;
        frame.deinit();
        try std.testing.expectEqual(@as(usize, 0), allocations.live);
    }
    for (0..count) |fail| {
        var allocations: metal.Allocations = .{ .fail = fail };
        try std.testing.expectError(error.NoBuffer, gpu.Frame.init(.{ .allocations = &allocations }, config(), 2048));
        try std.testing.expectEqual(@as(usize, 0), allocations.bytes);
        try std.testing.expectEqual(@as(usize, 0), allocations.live);
    }
    var allocations: metal.Allocations = .{};
    var frame = try gpu.Frame.init(.{ .allocations = &allocations }, config(), 512);
    defer frame.deinit();
    try std.testing.expectEqual(std.enums.values(gpu.Field).len, allocations.calls);
}
