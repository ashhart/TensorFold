//! Smoke test (Reviewer3, post-fix): pull_file.linkIntoSnapshot refuses a hub
//! file path with traversal segments. The hub tree JSON (`path`) once reached
//! std.fs.path.join unvalidated; the sink now rejects empty, absolute and
//! "."/".." component paths before any write. Run:
//!   zig test --dep pull_file -Mroot=smoke_tests/r3_linkIntoSnapshot_traversal.zig \
//!            -Mpull_file=zig/src/cli/pull_file.zig
const std = @import("std");
const pull_file = @import("pull_file");

test "a hub path with .. is refused and plants no symlink" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-r3-link-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();
    const cwd = try std.process.currentPathAlloc(io, a);

    const snap = try std.fmt.allocPrint(a, "{s}/{s}/snap", .{ cwd, root });
    try w.createDirPath(io, snap);
    const outside = try std.fmt.allocPrint(a, "{s}/{s}/outside/evil", .{ cwd, root });

    // Malicious hub tree entries: "../outside/evil", "/etc", "a/../b" and "".
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    for ([_][]const u8{ "../outside/evil", "/etc/evil", "a/../b", "" }) |bad| {
        try std.testing.expectError(error.SnapshotPath, pull_file.linkIntoSnapshot(a, io, snap, bad, "blobname"));
    }
    const n = std.Io.Dir.readLinkAbsolute(io, outside, &link_buf) catch return;
    _ = n;
    return error.TestUnexpectedResult; // nothing may be planted outside the snapshot dir
}

test "a hub path without traversal still links into the snapshot dir" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-r3-link-ok-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();
    const cwd = try std.process.currentPathAlloc(io, a);

    const snap = try std.fmt.allocPrint(a, "{s}/{s}/snap", .{ cwd, root });
    try w.createDirPath(io, snap);

    // A real hub tree entry: nested directories and a dotted name are fine.
    try pull_file.linkIntoSnapshot(a, io, snap, "deep/nested/model.1-of-2.safetensors", "blobname");

    const planted = try std.fmt.allocPrint(a, "{s}/deep/nested/model.1-of-2.safetensors", .{snap});
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try std.Io.Dir.readLinkAbsolute(io, planted, &link_buf);
    try std.testing.expect(std.mem.endsWith(u8, link_buf[0..n], "blobs/blobname"));
}
