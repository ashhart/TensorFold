//! A planted late mutation proves capture ordering and first-node attribution without a GPU.
const std = @import("std");
const capture = @import("node_capture.zig");
const B = @import("engine_test.zig").B;
fn read(_: *B, buffer: B.Buffer, out: []u8) !void {
    @memcpy(out, buffer[0..out.len]);
}
test "Qwen4 node capture: copies precede reuse and one wrong bit identifies the first differing node per row" {
    const a = std.testing.allocator;
    var b: B = .{};
    const input = try b.allocate(8);
    defer b.release(input);
    const bits = std.mem.bytesAsSlice(u16, input);
    @memcpy(bits, &[_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080 });
    var cap: capture.Capture(B, read) = .{ .a = a, .backend = &b };
    defer cap.close() catch unreachable;
    const sink = cap.sink();
    try sink.emit(sink.ctx, "first", input, 0, 2, 2, .bf16, 0);
    bits[2] ^= 1;
    try sink.emit(sink.ctx, "second", input, 0, 2, 2, .bf16, 0);
    const nodes = [_]capture.Node{
        .{ .name = "first", .position = 0, .kind = .bf16, .values = &.{ 1, 2 } },
        .{ .name = "first", .position = 1, .kind = .bf16, .values = &.{ 3, 4 } },
        .{ .name = "second", .position = 0, .kind = .bf16, .values = &.{ 1, 2 } },
        .{ .name = "second", .position = 1, .kind = .bf16, .values = &.{ 3, 4 } },
    };
    const oracle: capture.Oracle = .{ .nodes = &nodes };
    const summary = try cap.report(oracle, false);
    try std.testing.expectEqual(@as(usize, 1), summary.different);
    try std.testing.expectEqual(@as(usize, 4), summary.checked);
    try std.testing.expect(summary.first[0] == null);
    try std.testing.expectEqual(@as(?usize, 1), summary.first[1]);
    try std.testing.expectError(error.Qwen4TraceShape, sink.emit(sink.ctx, "bad", input, 0, 33, 1, .bf16, 0));
    try std.testing.expectError(error.Qwen4TraceMissing, cap.report(.{ .nodes = &(nodes ++ [_]capture.Node{.{ .name = "missing", .position = 0, .kind = .bf16, .values = &.{1} }}) }, false));
}
