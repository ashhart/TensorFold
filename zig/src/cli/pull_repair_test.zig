//! Published snapshots lose completion evidence when a same-revision repair fails.
const std = @import("std");
const pull = @import("pull.zig");
const FakeHub = @import("pull_test.zig").FakeHub;
const TestClient = @import("pull_test_client.zig").Client;

fn refuseWeight(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    if (!std.mem.startsWith(u8, head, "GET ") or std.mem.indexOf(u8, head, FakeHub.weights_path) == null) return null;
    fake.sendAll(conn, "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    return false;
}

test "pull revokes completion before a failed repair of a published snapshot" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-repair-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", try fake.endpoint(a));
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const sizes: pull.Sizes = .{ .threshold = 1 << 20 };
    const spec = "Org/Flash";
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out.writer, &err.writer, &env, root, spec, sizes));

    const repo = try std.fs.path.join(a, &.{ root, "models--Org--Flash" });
    const ref_path = try std.fs.path.join(a, &.{ repo, "refs/main" });
    const snapshot = try std.fs.path.join(a, &.{ repo, "snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1" });
    try std.Io.Dir.cwd().deleteFile(io, try std.fs.path.join(a, &.{ repo, "blobs", fake.weights_sha_hex }));
    fake.route = refuseWeight;
    try std.testing.expectEqual(@as(u8, 1), try pull.runWith(a, io, &out.writer, &err.writer, &env, root, spec, sizes));
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ snapshot, "config.json" }), .{});
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, ref_path, .{}));

    fake.route = null;
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out.writer, &err.writer, &env, root, spec, sizes));
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(64));
    try std.testing.expectEqualStrings("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1", ref);
}
