//! The GGUF subset's own bounds: tensor offsets and metadata arrays, refused rather than wrapping or recursing.

const std = @import("std");
const gguf = @import("glp_container.zig");

test "a tensor offset that would wind past the address space is refused, not wrapped" {
    const bytes = try std.testing.allocator.alloc(u8, 64);
    defer std.testing.allocator.free(bytes);
    const dims = [_]u64{8};
    const file: gguf.File = .{ .entries = &.{}, .tensors = &.{}, .data_start = 8, .alignment = 32, .bytes = bytes };
    const t: gguf.Tensor = .{ .name = "direction.10", .dims = &dims, .kind = gguf.f32_type, .offset = std.math.maxInt(u64) - 15 };
    // data_start + offset is eight bytes short of the end, where a start + size check would wrap into the file.
    try std.testing.expect(file.bytesOf(t) == null);
    const inside: gguf.Tensor = .{ .name = "direction.10", .dims = &dims, .kind = gguf.f32_type, .offset = 0 };
    try std.testing.expectEqual(@as(usize, 32), file.bytesOf(inside).?.len);
}

test "metadata arrays nested past the limit are refused" {
    // Four arrays inside one another parse; eight do not recurse.
    for ([_]u8{ 4, 8 }) |depth| {
        const bytes = try nested(depth);
        defer std.testing.allocator.free(bytes);
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        var problem: gguf.Problem = .{};
        if (depth == 4) {
            _ = try gguf.parse(arena.allocator(), bytes, &problem);
            continue;
        }
        try std.testing.expectError(error.BadGguf, gguf.parse(arena.allocator(), bytes, &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem.text(), "nests arrays deeper than") != null);
    }
}

/// A container whose only metadata value is `depth` arrays inside one another.
fn nested(depth: u8) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const tmp = arena.allocator();
    var value: gguf.Value = .{ .u32v = 1 };
    for (0..depth) |_| {
        const items = try tmp.alloc(gguf.Value, 1);
        items[0] = value;
        // The tag is read before the assignment: result-location semantics make `value` the new union already.
        const elem = std.meta.activeTag(value);
        value = .{ .array = .{ .elem = elem, .items = items } };
    }
    const entries = [_]gguf.Entry{.{ .key = "general.notes", .value = value }};
    return gguf.write(std.testing.allocator, &entries, &.{});
}
