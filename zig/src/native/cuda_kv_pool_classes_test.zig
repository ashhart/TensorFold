//! TF_GLM_KV_SPLIT's owner classes in the CUDA pool's plan: whole classes, one short of a window refused by name.
const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("core").kv_pool;
const pool = @import("cuda_kv_pool.zig");

const Owned = struct {
    pub const snap_save = 1;
    pub const window_rows = 8;
};

test "kv pages: owner classes floor the plan's pages to the classes, and a class short of a window is refused with its numbers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    var l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0, .classes = 2 };
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4, .prompt_cache_gib = 4 };
    const p = (try pool.plan(Owned, arena.allocator(), l, 40 << 30, o, 8192, &problem, false, true, 0, null)).?;
    try std.testing.expectEqual(@as(usize, 4 * 17), p.pages); // even already: two classes of 34, two windows each
    l.classes = 3;
    try std.testing.expectEqual(@as(usize, 66), (try pool.plan(Owned, arena.allocator(), l, 40 << 30, o, 8192, &problem, false, true, 0, null)).?.pages);
    l.classes = 8; // 8 classes of 8 pages, a window of 17
    try std.testing.expectEqual(@as(?pool.Plan, null), try pool.plan(Owned, arena.allocator(), l, 40 << 30, o, 8192, &problem, false, true, 0, null));
    try std.testing.expectEqualStrings("the KV pages split into 8 owner classes of 8 pages, and one stream at a 8192-token window takes 17: lower --context, raise --parallel to 8 or more, or serve without TF_GLM_KV_SPLIT", problem);
}
