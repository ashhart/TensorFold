//! The lone driver's first commit and the prompt slice it must copy after a possible relocation.
const std = @import("std");
const Stream = @import("lanes").Stream;

/// Commits a lone stream's first token; its prompt as the context holds it after the commit, which can move it.
pub fn commitFirst(gpa: std.mem.Allocator, stream: *Stream, token: u32) ![]const u32 {
    _ = try stream.commit(gpa, &.{token}, &.{});
    return stream.prompt();
}

test "a lone stream's prompt is read after its first token's commit moves the context" {
    const gpa = std.testing.allocator;
    const tokens = [_]u32{ 11, 12, 13, 14, 15, 16, 17 };
    var stream = try Stream.init(gpa, .{ .id = "lone", .prompt = &tokens, .max_new = 4, .eos = &.{} });
    defer stream.deinit(gpa);
    stream.context.shrinkAndFree(gpa, stream.context.items.len); // full: the first token must grow it
    var failing = std.testing.FailingAllocator.init(gpa, .{ .resize_fail_index = 0 }); // and not in place
    const before = stream.context.items.ptr;
    const prompt = try commitFirst(failing.allocator(), &stream, 99);
    try std.testing.expect(stream.context.items.ptr != before);
    try std.testing.expectEqual(stream.context.items.ptr, prompt.ptr);
    try std.testing.expectEqualSlices(u32, &tokens, prompt);
    try std.testing.expectEqualSlices(u32, &.{99}, stream.emitted());
}
