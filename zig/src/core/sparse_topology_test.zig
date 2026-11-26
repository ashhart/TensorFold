//! sparse_topology on a model shaped like Kolibri 1.1's config lists, with the follow rule left as a parameter.
const std = @import("std");
const st = @import("sparse_topology.zig");
const testing = std.testing;
const gpa = testing.allocator;

// config at e9515166: sparse layers 4, 9, ..., 49; KV sources 9/14/49 -> 4, 34/39/44 -> 29; index owners 4, 14, 19, 29
const sparse = [_]u32{ 4, 9, 14, 19, 24, 29, 34, 39, 44, 49 };
const sources = [_]st.Source{ .{ .layer = 9, .source = 4 }, .{ .layer = 14, .source = 4 }, .{ .layer = 49, .source = 4 }, .{ .layer = 34, .source = 29 }, .{ .layer = 39, .source = 29 }, .{ .layer = 44, .source = 29 } };
const owners = [_]u32{ 4, 14, 19, 29 };

test "sparse topology: KV, selection and index-key owners stay apart (layers 14, 24, 49)" {
    const follow = try st.followLatest(gpa, &sparse, &owners);
    defer gpa.free(follow);
    const t = try st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow });
    defer gpa.free(t);
    // 14: its own indexer on borrowed KV
    try testing.expectEqual(@as(u32, 4), t[14].kv_owner);
    try testing.expectEqual(@as(?u32, 14), t[14].select_owner);
    try testing.expectEqual(@as(?u32, null), t[14].cache_slot);
    // 24: its own KV, no indexer: it follows an owner
    try testing.expectEqual(@as(u32, 24), t[24].kv_owner);
    try testing.expect(t[24].cache_slot != null);
    try testing.expectEqual(@as(?u32, 19), t[24].select_owner);
    // 49: KV from 4, selection from 29 under "latest owner": the two maps differ, and the table keeps both
    try testing.expectEqual(@as(u32, 4), t[49].kv_owner);
    try testing.expectEqual(@as(?u32, 29), t[49].select_owner);
    try testing.expectEqual(@as(?u32, 29), t[49].index_key_owner); // final reuse: the owner scored on its own cache
    // dense and sliding layers: no selection; one slot each KV owner
    try testing.expectEqual(@as(?u32, null), t[0].select_owner);
    var slots: usize = 0;
    for (t) |e| slots += @intFromBool(e.cache_slot != null);
    try testing.expectEqual(@as(usize, 50 - sources.len), slots);
}

test "sparse topology: the follow rule is a parameter; leaving one sparse layer out is refused, not guessed" {
    // the other candidate for 49: follow its KV source's indexer (4)
    var follow = try st.followLatest(gpa, &sparse, &owners);
    defer gpa.free(follow);
    for (follow) |*f| if (f.layer == 49) {
        f.owner = 4;
    };
    const t = try st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow });
    defer gpa.free(t);
    try testing.expectEqual(@as(?u32, 4), t[49].select_owner);
    try testing.expectError(error.UnsettledSelectOwner, st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow[0 .. follow.len - 1] }));
}

test "sparse topology: planted bad lists are refused by name" {
    const base: st.Lists = .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners };
    var l = base;
    l.kv_source = &.{.{ .layer = 4, .source = 9 }};
    try testing.expectError(error.LaterSource, st.build(gpa, l));
    l = base;
    l.kv_source = &.{ .{ .layer = 9, .source = 4 }, .{ .layer = 14, .source = 9 } };
    try testing.expectError(error.ChainedSource, st.build(gpa, l));
    l = base;
    l.index_owners = &.{ 4, 5 };
    try testing.expectError(error.OwnerNotSparse, st.build(gpa, l));
    l = base;
    l.select_from = &.{.{ .layer = 9, .owner = 24 }};
    try testing.expectError(error.FollowerOfNonOwner, st.build(gpa, l));
    l = base;
    l.select_from = &.{.{ .layer = 9, .owner = 14 }};
    try testing.expectError(error.LaterOwner, st.build(gpa, l));
    l = base;
    l.select_from = &.{.{ .layer = 14, .owner = 4 }};
    try testing.expectError(error.FollowerIsOwner, st.build(gpa, l));
    l = base;
    l.select_from = &.{.{ .layer = 10, .owner = 4 }};
    try testing.expectError(error.FollowerNotSparse, st.build(gpa, l));
}

test "sparse topology: pool reuse rescores on the follower's own keys; sizes per owner and row tiles" {
    const follow = try st.followLatest(gpa, &sparse, &owners);
    defer gpa.free(follow);
    const t = try st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow, .reuse = .pool });
    defer gpa.free(t);
    try testing.expectEqual(@as(?u32, 29), t[49].pool_owner);
    try testing.expectEqual(@as(?u32, 4), t[49].index_key_owner);
    // 8,192 prompt rows over 65,536 keys would be 2 GiB of fp32 scores: 256 MiB tiles take 1,024 rows
    try testing.expectEqual(@as(usize, 1024), st.tileRows(65536, 256 << 20));
    try testing.expectEqual(@as(usize, 1), st.tileRows(1 << 30, 16));
    try testing.expectEqual(@as(usize, 8192 * (2184 + 16384) * 4), st.keptBytes(8192, 2184, 16384));
}

test "sparse topology: an index cache apart from the KV cache is named, and followers read their owner's" {
    const follow = try st.followLatest(gpa, &sparse, &owners);
    defer gpa.free(follow);
    // layer 19 keeps its index keys in layer 14's index cache (a model that shares index caches but not KV)
    const t = try st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow, .index_cache = &.{.{ .layer = 19, .source = 14 }} });
    defer gpa.free(t);
    try testing.expectEqual(@as(?u32, 14), t[19].index_key_owner);
    try testing.expectEqual(@as(u32, 19), t[19].kv_owner); // its KV stays its own
    try testing.expectEqual(@as(?u32, 14), t[24].index_key_owner); // 24 follows 19: final picks scored on 14's cache
    // planted: a later layer's index cache, a non-indexer, the same layer twice
    const bad = [_][]const st.Source{ &.{.{ .layer = 19, .source = 29 }}, &.{.{ .layer = 24, .source = 4 }}, &.{ .{ .layer = 19, .source = 4 }, .{ .layer = 19, .source = 14 } } };
    const want = [_]anyerror{ error.LaterSource, error.OwnerNotSparse, error.Repeated };
    for (bad, want) |b, w| try testing.expectError(w, st.build(gpa, .{ .layers = 50, .sparse = &sparse, .kv_source = &sources, .index_owners = &owners, .select_from = follow, .index_cache = b }));
}
