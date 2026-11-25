//! The working-set trim the checkpoint's release takes on Windows, exercised on the host.
const std = @import("std");
const builtin = @import("builtin");
const working_set = @import("sys").working_set;

test "release trims the process working set and leaves every byte it was given as it was" {
    // elsewhere the trim is a no-op by design: safetensors.File.release has the page cache advice there
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const page = std.heap.pageSize();
    const bytes = try std.testing.allocator.alloc(u8, 4 * page);
    defer std.testing.allocator.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 37 +% 11);
    const kept = try std.testing.allocator.dupe(u8, bytes);
    defer std.testing.allocator.free(kept);
    // twice, as a load calls it a step and then a tensor: a trim takes out no byte the process holds
    working_set.release();
    working_set.release();
    try std.testing.expectEqualSlices(u8, kept, bytes);
}
