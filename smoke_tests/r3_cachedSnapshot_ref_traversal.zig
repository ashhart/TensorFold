//! Smoke test (Reviewer3, post-fix): hub.cachedSnapshot refuses a refs/main
//! revision that is not a 40-hex git sha. The revision once joined into the
//! snapshot path raw, so a "../.." ref (or a raw JSON "sha" pull.zig had
//! written) resolved the snapshot outside the cache. The same primitive is
//! pull.zig's `std.fs.path.join(&.{ root, "snapshots", sha })`, where the JSON
//! "sha" is now required to be 40 hex. Run:
//!   zig test --dep hub --dep native_engines \
//!            -Mroot=smoke_tests/r3_cachedSnapshot_ref_traversal.zig \
//!            -Mhub=zig/src/cli/hub.zig \
//!            -Mnative_engines=smoke_tests/r3_native_engines_stub.zig
const std = @import("std");
const hub = @import("hub");

test "a refs/main revision outside the sha space resolves no snapshot" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-r3-ref-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();

    // An "escaped" directory outside the repo's cache, holding a config.json.
    try w.createDirPath(io, try std.fmt.allocPrint(a, "{s}/escaped-dir", .{root}));
    try w.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/escaped-dir/config.json", .{root}), .data = "{\"model_type\": \"qwen4_exp\"}" });
    try w.createDirPath(io, try std.fmt.allocPrint(a, "{s}/models--Org--Name/refs", .{root}));
    try w.createDirPath(io, try std.fmt.allocPrint(a, "{s}/models--Org--Name/snapshots", .{root}));

    // Traversal segments, a short sha, an uppercase-wrong sha and a non-sha all resolve nothing.
    for ([_][]const u8{ "../../escaped-dir", "d9d758fb", "d9d758fbcouldbefortycharacters", "main" }) |bad| {
        try w.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/models--Org--Name/refs/main", .{root}), .data = bad });
        const snapshot = try hub.cachedSnapshot(a, io, root, "Org/Name");
        try std.testing.expectEqual(@as(?[]const u8, null), snapshot);
    }
    const resolved = try std.fs.path.resolve(a, &.{try std.fmt.allocPrint(a, "{s}/escaped-dir", .{root})});
    const repo_dir = try std.fmt.allocPrint(a, "{s}/models--Org--Name/snapshots", .{root});
    try std.testing.expect(!std.mem.startsWith(u8, resolved, try std.fs.path.resolve(a, &.{repo_dir})));
}

test "a 40-hex refs/main revision still resolves the snapshot" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-r3-ref-ok-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();

    const sha = "d9d758fb83953437f7263256b0d96157e2a348b8";
    try w.createDirPath(io, try std.fmt.allocPrint(a, "{s}/models--Org--Name/refs", .{root}));
    try w.createDirPath(io, try std.fmt.allocPrint(a, "{s}/models--Org--Name/snapshots/{s}", .{ root, sha }));
    try w.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/models--Org--Name/snapshots/{s}/config.json", .{ root, sha }), .data = "{\"model_type\": \"qwen4_exp\"}" });
    try w.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/models--Org--Name/refs/main", .{root}), .data = sha });

    const snapshot = (try hub.cachedSnapshot(a, io, root, "Org/Name")) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.endsWith(u8, snapshot, sha));
}
