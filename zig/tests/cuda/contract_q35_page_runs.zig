//! The exact portable helper included by both CUDA cache formats, with a table-read regression witness.
const std = @import("std");
const PageRun = extern struct { ids: [*]const c_int, tokens: c_int, logical: c_int, stride: u64, base: u64 };
extern fn tf_test_q35_page_base(run: *PageRun, pos: c_int) u64;

test "Qwen page run: one ID read per encountered page, shared by K and V" {
    var ids = [_]c_int{ 7, 2, 9 };
    var run = PageRun{ .ids = &ids, .tokens = 48, .logical = -1, .stride = 1024, .base = 0 };
    try std.testing.expectEqual(@as(u64, 7 * 1024), tf_test_q35_page_base(&run, 47));
    ids[0] = 99; // a second table read in this run would expose the planted value
    try std.testing.expectEqual(@as(u64, 7 * 1024), tf_test_q35_page_base(&run, 47));
    try std.testing.expectEqual(@as(u64, 2 * 1024), tf_test_q35_page_base(&run, 48));
    ids[1] = 98;
    try std.testing.expectEqual(@as(u64, 2 * 1024), tf_test_q35_page_base(&run, 49));
    try std.testing.expectEqual(@as(u64, 9 * 1024), tf_test_q35_page_base(&run, 96));
}

test "Qwen page run: shuffled pages at chunk and page boundaries keep the original physical addresses" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 16, 48, 128, 256, 512, 1024 }) |tokens| {
        const ids = try a.alloc(c_int, (65537 + tokens - 1) / tokens);
        defer a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(ids.len - 1 - i);
        for ([_]usize{ 127, 128, 129, 8191, 8192, 8193, 16384, 32768, 65536 }) |keys| {
            const chunk = @max(128, (keys + 8191) / 8192 * 128);
            var first: usize = 0;
            while (first < keys) : (first += chunk) {
                var run = PageRun{ .ids = ids.ptr, .tokens = @intCast(tokens), .logical = -1, .stride = 0x12345600, .base = 0 };
                for (first..@min(first + chunk, keys)) |pos| {
                    const original = @as(u64, @intCast(ids[pos / tokens])) * run.stride;
                    try std.testing.expectEqual(original, tf_test_q35_page_base(&run, @intCast(pos)));
                    try std.testing.expectEqual(original, tf_test_q35_page_base(&run, @intCast(pos))); // value reuses key's ID
                }
            }
        }
    }
}
