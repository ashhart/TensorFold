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
    try std.testing.expectEqualStrings("the KV pages split into 8 owner classes of 8 pages, and one stream at a 8192-token window takes 17, and the trash page one more: lower --context or --prompt-cache-gib, raise --parallel to 8 or more, or serve without TF_GLM_KV_SPLIT", problem);
}

test "CUDA cache default under owner classes: the cache the unsplit pool leaves, set aside first, the pages the rest" {
    // GLM at TP=2, a 1M window, 8 streams, 30 GiB of room: 13,056 bytes a token unsplit, 6,912 with the split
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    var o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 8 };
    const unsplit: kv_pool.Layout = .{ .fixed_bytes = 378 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    var split = unsplit;
    split.token_bytes = 6_912;
    split.classes = 2;
    split.whole_bytes = 13_056;
    const bu = pool.budgetOf(Owned, unsplit, 30 << 30, o, 1 << 20, false, true, 0, null);
    const pu = (try pool.plan(Owned, arena.allocator(), unsplit, 30 << 30, o, 1 << 20, &problem, false, true, 0, null)).?;
    try std.testing.expectEqual(@as(u64, 2_043_936_768), bu.cache); // 1.90 GiB after two whole windows
    try std.testing.expectEqual(@as(usize, 4_098), pu.pages); // 2.10M tokens
    var one = unsplit;
    one.whole_bytes = 13_056; // one class: the field changes nothing
    try std.testing.expect(std.meta.eql(bu, pool.budgetOf(Owned, one, 30 << 30, o, 1 << 20, false, true, 0, null)));
    const bs = pool.budgetOf(Owned, split, 30 << 30, o, 1 << 20, false, true, 0, null);
    const ps = (try pool.plan(Owned, arena.allocator(), split, 30 << 30, o, 1 << 20, &problem, false, true, 0, null)).?;
    try std.testing.expectEqual(bu.cache, bs.cache); // the unsplit cache, not the 0.40 GiB four whole windows would leave
    try std.testing.expect(bs.asked);
    try std.testing.expectEqual(bu.cache, ps.copied);
    try std.testing.expectEqual(@as(usize, 7_740), ps.pages); // 3.96M tokens, two classes of 3,870
    o.prompt_cache_gib = 6; // a named cache is the same split or not
    try std.testing.expectEqual(@as(u64, 6 << 30), pool.budgetOf(Owned, split, 30 << 30, o, 1 << 20, false, true, 0, null).cache);
}

test "CUDA cache default under owner classes: lowered only as far as each class holds a window, when the unsplit one starves them" {
    // a 1M window, 8 streams, 18.2 GiB of room: the unsplit cache left each class 1,941 pages of a window's 2,049
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 8 };
    const room: u64 = 18_200 << 20;
    var split: kv_pool.Layout = .{ .fixed_bytes = 391 << 20, .token_bytes = 6_912, .page_tokens = 512, .snap_bytes = 0, .classes = 2, .whole_bytes = 13_056 };
    var unsplit = split;
    unsplit.token_bytes = 13_056;
    unsplit.classes = 1;
    const window: u64 = 1 << 20;
    const bu = pool.budgetOf(Owned, unsplit, room, o, window, false, true, 0, null);
    var starved = bu; // the unsplit cache as before: refused
    starved.asked = true;
    try std.testing.expectError(error.ClassTooSmall, kv_pool.planClasses(split, starved));
    const bs = pool.budgetOf(Owned, split, room, o, window, false, true, 0, null);
    try std.testing.expect(bs.cache > 0 and bs.cache < bu.cache);
    const ps = (try pool.plan(Owned, arena.allocator(), split, room, o, window, &problem, false, true, 0, null)).?;
    try std.testing.expect(ps.pages / 2 >= split.pages(window + 8) + 1); // each class a window and the trash page
    var more = bs; // and no more than that taken: a page's bytes more of cache would starve a class again
    more.cache += split.pageBytes();
    try std.testing.expectError(error.ClassTooSmall, kv_pool.planClasses(split, more));
}
