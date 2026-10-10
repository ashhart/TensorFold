//! Smoke test (Reviewer3): negative controls. blobName() does neutralize '/'
//! in hub paths that carry no digest, and a plain hub path stays inside the
//! snapshot dir — so the traversal in the sibling tests needs ".." specifically.
//!   zig test --dep pull_file -Mroot=smoke_tests/r3_negative_controls.zig \
//!            -Mpull_file=zig/src/cli/pull_file.zig
const std = @import("std");
const pull_file = @import("pull_file");
const hub = @import("hub");

test "blobName neutralizes '/' and a plain path stays inside" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root_rel = try std.fmt.allocPrint(a, ".tf-r3-neg-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root_rel) catch {};
    const cwd = try std.process.currentPathAlloc(io, a);
    const root = try std.fmt.allocPrint(a, "{s}/{s}", .{ cwd, root_rel });

    // No '/' remains after blobName's sanitization.
    const name = try pull_file.blobName(a, .{ .path = "sub/dir/weights.bin", .size = 1 });
    try std.testing.expect(std.mem.indexOfScalar(u8, name, '/') == null);

    // A well-formed path writes inside the snapshot dir only.
    const snap = try std.fmt.allocPrint(a, "{s}/snap", .{root});
    try std.Io.Dir.cwd().createDirPath(io, snap);
    try pull_file.linkIntoSnapshot(a, io, snap, "sub/dir/weights.bin", "blobname");
    const inside = try std.fmt.allocPrint(a, "{s}/sub/dir/weights.bin", .{snap});
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    _ = try std.Io.Dir.readLinkAbsolute(io, inside, &link_buf);

    // isRepoIdLike rejects traversal-shaped repo ids (the input that IS validated).
    try std.testing.expect(!hub.isRepoIdLike("../../etc"));
    try std.testing.expect(!hub.isRepoIdLike("org/na..me/../x"));
    try std.testing.expect(hub.isRepoIdLike("Org/Flash"));
}
