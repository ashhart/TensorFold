//! Byte identity and refusal tests through the CUDA tower's real checkpoint reader, without CUDA libraries.
const std = @import("std");
const weights = @import("qwen_cuda_weights.zig");
const fixture = @import("qwen_cuda_weights_fixture.zig");
const diagnostics = @import("qwen_weights_diagnostics");

fn run(a: std.mem.Allocator, dir: []const u8, fail_at: ?usize) !void {
    var sink: fixture.Sink = .{ .a = a, .fail_at = fail_at };
    defer sink.deinit();
    try weights.load(a, std.testing.io, dir, fixture.config, &sink);
    try expectCopies(&sink);
}

fn expectCopies(sink: *const fixture.Sink) !void {
    try std.testing.expectEqual(fixture.tensors.len, sink.copies.items.len);
    try std.testing.expectEqual(@as(u64, 1), sink.patch_w);
    try std.testing.expectEqual(@as(u64, 2), sink.patch_b);
    try std.testing.expectEqual(@as(u64, 3), sink.pos);
    const fields = .{ "norm1_w", "norm1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "norm2_w", "norm2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b" };
    for (sink.blocks, 0..) |block, row| inline for (fields, 4..) |field, i| try std.testing.expectEqual(@as(u64, i + row * 12), @field(block, field));
    try std.testing.expectEqual(@as(u64, 28), sink.merger_norm_w);
    try std.testing.expectEqual(@as(u64, 29), sink.merger_norm_b);
    try std.testing.expectEqual(@as(u64, 30), sink.fc1_w);
    try std.testing.expectEqual(@as(u64, 31), sink.fc1_b);
    try std.testing.expectEqual(@as(u64, 32), sink.fc2_w);
    try std.testing.expectEqual(@as(u64, 33), sink.fc2_b);
    for (fixture.tensors, sink.copies.items, 0..) |tensor, bytes, id| {
        try std.testing.expectEqual(fixture.count(tensor.shape) * 2, bytes.len);
        for (0..bytes.len / 2) |i| {
            const source = if (id == 0) (i / 1536) * 1536 + (i % 512) * 3 + (i / 512) % 3 else i;
            try std.testing.expectEqual(fixture.value(id, source), std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
        }
    }
}

fn matching(indexed: bool) !void {
    var mlx = std.testing.tmpDir(.{});
    defer mlx.cleanup();
    var hf = std.testing.tmpDir(.{});
    defer hf.cleanup();
    try fixture.write(mlx, .{ .indexed = indexed });
    try fixture.write(hf, .{ .indexed = indexed, .hf = true });
    var mlx_buf: [96]u8 = undefined;
    var hf_buf: [96]u8 = undefined;
    const mlx_dir = try fixture.directory(mlx, &mlx_buf);
    const hf_dir = try fixture.directory(hf, &hf_buf);
    var old: fixture.Sink = .{ .a = std.testing.allocator };
    defer old.deinit();
    var direct: fixture.Sink = .{ .a = std.testing.allocator };
    defer direct.deinit();
    try weights.load(std.testing.allocator, std.testing.io, mlx_dir, fixture.config, &old);
    try weights.load(std.testing.allocator, std.testing.io, hf_dir, fixture.config, &direct);
    try expectCopies(&old);
    try expectCopies(&direct);
    for (old.copies.items, direct.copies.items) |a, b| try std.testing.expectEqualSlices(u8, a, b);
}

test "Qwen tower weights: existing MLX names and every byte match HF in one shard" {
    try matching(false);
}

test "Qwen tower weights: indexed MLX fallback and sharded HF preserve every tensor" {
    try matching(true);
}

test "Qwen tower weights: malformed selected shape and dtype fail without a fallback" {
    for ([_]fixture.Options{ .{ .hf = true, .wrong_dtype = true }, .{ .hf = true, .wrong_patch_shape = true }, .{ .wrong_dtype = true }, .{ .wrong_patch_shape = true } }) |options| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try fixture.write(tmp, options);
        var buf: [96]u8 = undefined;
        try std.testing.expectError(error.UnexpectedTensor, run(std.testing.allocator, try fixture.directory(tmp, &buf), null));
        try diagnostics.expect(1);
    }
}

test "Qwen tower weights: missing selected tensor and unused selected tensors fail closed" {
    for ([_]bool{ false, true }) |hf| {
        for ([_]bool{ false, true }) |extra| {
            var tmp = std.testing.tmpDir(.{});
            defer tmp.cleanup();
            try fixture.write(tmp, .{ .hf = hf, .extra = extra, .missing = if (extra) null else 12 });
            var buf: [96]u8 = undefined;
            const result = run(std.testing.allocator, try fixture.directory(tmp, &buf), null);
            try std.testing.expectError(if (extra) error.UnusedVisionTensors else error.MissingTensor, result);
            try diagnostics.expect(1);
        }
    }
}

test "Qwen tower weights: HF patch keeps the authored precedence when both namespaces exist" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(tmp, .{ .hf = true, .mixed = true });
    var buf: [96]u8 = undefined;
    try run(std.testing.allocator, try fixture.directory(tmp, &buf), null);
}

test "Qwen tower weights: every copy refusal and MLX retry release their host owners" {
    for ([_]bool{ false, true }) |hf| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try fixture.write(tmp, .{ .hf = hf, .indexed = true });
        var buf: [96]u8 = undefined;
        const dir = try fixture.directory(tmp, &buf);
        for (0..fixture.tensors.len) |i| try std.testing.expectError(error.InjectedCopyFailure, run(std.testing.allocator, dir, i));
        try run(std.testing.allocator, dir, null);
    }
}

test "Qwen tower weights: allocation failures balance the fallback reader and transpose scratch" {
    for ([_]bool{ false, true }) |hf| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try fixture.write(tmp, .{ .hf = hf, .indexed = true });
        var buf: [96]u8 = undefined;
        const dir = try fixture.directory(tmp, &buf);
        var n: usize = 0;
        while (true) : (n += 1) {
            var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = n, .resize_fail_index = 0 });
            run(failing.allocator(), dir, null) catch |err| {
                if (err != error.OutOfMemory or !failing.has_induced_failure) return err;
            };
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            if (!failing.has_induced_failure) break;
        }
    }
}
