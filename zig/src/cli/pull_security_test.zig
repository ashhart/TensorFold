//! Bounded Hub metadata and digest-checked cache reuse against a local fake Hub.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");
const TestClient = @import("pull_test_client.zig").Client;
const FakeHub = @import("pull_test.zig").FakeHub;

fn testEnv(a: Allocator, endpoint: []const u8) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", endpoint);
    return env;
}

fn bodyPath(head: []const u8) []const u8 {
    const start = std.mem.indexOfScalar(u8, head, ' ') orelse return "";
    const rest = head[start + 1 ..];
    const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
    return rest[0..end];
}

fn compressedOversize(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const path = bodyPath(head);
    const revision = std.mem.eql(u8, path, "/api/models/Org/Flash/revision/main");
    const tree = std.mem.eql(u8, path, "/api/models/Org/Flash/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1?recursive=true");
    const config = std.mem.eql(u8, path, "/Org/Flash/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/config.json");
    const mode = fake.range_starts.load(.monotonic);
    if (!((mode == 0 and revision) or (mode == 1 and tree) or (mode == 2 and config))) return null;
    var header: [256]u8 = undefined;
    const text = std.fmt.bufPrint(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{fake.weights.len}) catch return false;
    fake.sendAll(conn, text);
    fake.sendAll(conn, fake.weights);
    return true;
}

test "security: decompressed revision, tree and config JSON have byte caps" {
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
    var compressed = try std.Io.Writer.Allocating.initCapacity(a, 4096);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var gzip = try std.compress.flate.Compress.init(&compressed.writer, &window, .gzip, .default);
    try gzip.writer.splatByteAll('A', 17 << 20);
    try gzip.finish();
    fake.weights = compressed.written();
    fake.route = compressedOversize;
    const root = try std.fmt.allocPrint(a, ".tf-pull-caps-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake.endpoint(a));
    for (0..3) |mode| {
        fake.range_starts.store(mode, .monotonic);
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        if (mode == 2) {
            try std.testing.expectError(error.HubTooLarge, pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
        } else {
            try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
            try std.testing.expect(std.mem.indexOf(u8, err_out.written(), "decompressed byte cap") != null);
        }
    }
}

test "security: poisoned cached LFS and git blobs download again and verify" {
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
    const root = try std.fmt.allocPrint(a, ".tf-pull-cache-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake.endpoint(a));
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    const blobs = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs" });
    const weight_blob = try std.fs.path.join(a, &.{ blobs, fake.weights_sha_hex });
    const config_blob = try std.fs.path.join(a, &.{ blobs, &fake.config_git_hex });
    const bad_weights = try a.dupe(u8, FakeHub.weights_body);
    bad_weights[0] ^= 1;
    const bad_config = try a.dupe(u8, FakeHub.config_body);
    bad_config[0] ^= 1;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = weight_blob, .data = bad_weights });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_blob, .data = bad_config });
    var repaired: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &repaired.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 2), fake.weight_requests.load(.monotonic));
    const snapshot = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1" });
    const weights = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" }), a, .limited(1 << 20));
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "config.json" }), a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, weights);
    try std.testing.expectEqualStrings(FakeHub.config_body, config);
    var cached: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &cached.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 2), fake.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, cached.written(), "weights.safetensors: cached") != null);
}
