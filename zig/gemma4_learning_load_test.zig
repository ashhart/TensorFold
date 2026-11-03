//! The real learned-checkpoint proxy preserves original packed columns and owns every compacted buffer on failures.
const std = @import("std");
const M = @import("metal");
const ckpt = @import("src/core/checkpoint_metal.zig");
const cfg = @import("src/families/gemma4/config.zig");
const Proxy = @import("src/families/gemma4/learning_weights.zig").Proxy;
const matrix = @import("living").dense_matrix;
const a = std.testing.allocator;
fn run(allocator: std.mem.Allocator) !void {
    M.reset();
    defer for (M.releases[0..M.made]) |count| std.testing.expectEqual(@as(usize, 1), count) catch unreachable;
    var checkpoint = ckpt.Checkpoint.init(a);
    defer checkpoint.deinit();
    var originals: [9]M.Buffer = undefined;
    var made: usize = 0;
    defer for (originals[0..made]) |b| b.deinit();
    for (std.enums.values(matrix.Role)) |role| inline for (std.enums.values(matrix.Part)) |part| {
        const rows: usize = if (role == .down) 64 else 128;
        const cols: usize = (if (role == .down) @as(usize, 128) else 64) / (if (part == .weight) @as(usize, 8) else 64);
        const size = rows * cols * (if (part == .weight) @as(usize, 4) else 2);
        const buffer = try M.Device.buffer(.{}, size, M.ResourceOptions.shared);
        originals[made] = buffer;
        made += 1;
        for (buffer.contents()[0..size], 0..) |*v, i| v.* = @truncate(i * 37 + i / 17);
        const stem = if (role == .gate) "gate_proj" else if (role == .up) "up_proj" else "down_proj";
        const key = try std.fmt.allocPrint(a, "language_model.model.layers.0.mlp.{s}.{s}", .{ stem, @tagName(part) });
        errdefer a.free(key);
        var shape: @TypeOf(@as(ckpt.Tensor, undefined).shape) = @splat(1);
        shape[0] = rows;
        shape[1] = cols;
        try checkpoint.tensors.put(a, key, .{ .buffer = buffer, .offset = 0, .bytes = size, .dtype = if (part == .weight) .u32 else .bf16, .shape = shape, .rank = 2 });
    };
    var geo = cfg.dense31b;
    geo.hidden = 64;
    geo.intermediate = 64;
    var proxy: Proxy = .{ .a = allocator, .device = .{}, .checkpoint = &checkpoint, .config = .{ .geo = geo, .context = 2048, .learned = 64 } };
    defer proxy.deinit();
    var keys = checkpoint.tensors.keyIterator();
    while (keys.next()) |key| {
        const original = try checkpoint.get(key.*);
        const compact = try proxy.get(key.*);
        const again = try proxy.get(key.*);
        try std.testing.expectEqual(compact.buffer.id, again.buffer.id);
        const down = std.mem.indexOf(u8, key.*, ".down_proj.") != null;
        if (down) {
            const old_stride = original.bytes / 64;
            const new_stride = compact.bytes / 64;
            try std.testing.expectEqual(old_stride / 2, new_stride);
            for (0..64) |row| try std.testing.expectEqualSlices(u8, original.host(u8)[row * old_stride ..][0..new_stride], compact.host(u8)[row * new_stride ..][0..new_stride]);
        } else {
            try std.testing.expectEqual(@as(usize, 64), compact.shape[0]);
            try std.testing.expectEqual(original.buffer.id, compact.buffer.id);
            try std.testing.expectEqualSlices(u8, original.host(u8)[0 .. original.bytes / 2], compact.host(u8));
        }
    }
}
test "Gemma4 learning: real Metal load proxy keeps base bytes and handles every allocator failure without double release" {
    try std.testing.checkAllAllocationFailures(a, run, .{});
}
