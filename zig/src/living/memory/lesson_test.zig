//! A lesson is published whole: its staged units are unseen by routing until publishParent, then live as one version.
const std = @import("std");
const safety = @import("safety_test.zig");
const t = std.testing;

const more = [_]u32{ 40, 41, 42, 43 };
const ask = [_]u32{ 40, 41, 42 };

fn hits(b: anytype, mock: *safety.Mock, q: []const u32) !bool {
    const snap = (try b.routeAndCompose(q)) orelse return false;
    mock.backend().dropSnap(mock, snap.saved);
    return true;
}

test "bank lesson: staged units change no route, index or count until their lesson is published, then one version" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, false);
    defer b.deinit();
    const v0 = b.version;
    const clusters = b.cluster_index.n_clusters;
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true, .staged = true });
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true, .staged = true });
    try t.expectEqual(v0, b.version);
    try t.expectEqual(@as(usize, 1), b.units.items.len);
    try t.expectEqual(clusters, b.cluster_index.n_clusters);
    try t.expect(!try hits(b, &mock, &ask)); // half a lesson, or a whole unpublished one, is never routed
    try t.expectEqual(@as(u32, 2), try b.publishParent(8));
    try t.expectEqual(v0 + 1, b.version);
    try t.expectEqual(@as(usize, 3), b.units.items.len);
    try t.expect(try hits(b, &mock, &ask));
    try t.expectEqual(@as(u32, 0), try b.publishParent(8)); // nothing left staged: no new version
    try t.expectEqual(v0 + 1, b.version);
}

test "bank lesson: a staged lesson forgotten before publication is never seen and frees its units" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, false);
    defer b.deinit();
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true, .staged = true });
    try t.expectEqual(@as(u32, 1), b.forgetParent(8));
    try t.expectEqual(@as(usize, 0), b.staged.items.len);
    try t.expectEqual(@as(u32, 0), try b.publishParent(8));
    try t.expect(!try hits(b, &mock, &ask));
}
