//! Bounded Hub metadata and digest-checked cache reuse against a local fake Hub.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");
const TestClient = @import("pull_test_client.zig").Client;
const FakeHub = @import("pull_test.zig").FakeHub;
const ranged = @import("pull_parts.zig");
const runs = @import("pull_runs.zig");

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

fn gitHex(body: []const u8) [40]u8 {
    var git = std.crypto.hash.Sha1.init(.{});
    var head: [32]u8 = undefined;
    git.update(std.fmt.bufPrint(&head, "blob {d}\x00", .{body.len}) catch unreachable);
    git.update(body);
    return std.fmt.bytesToHex(git.finalResult(), .lower);
}

fn reflectedConfig(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    if (!std.mem.endsWith(u8, bodyPath(head), "/config.json")) return null;
    var header: [192]u8 = undefined;
    const text = std.fmt.bufPrint(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{fake.config.len}) catch return false;
    fake.sendAll(conn, text);
    var at: usize = 0;
    while (at < fake.config.len) : (at += 4) fake.sendAll(conn, fake.config[at..@min(at + 4, fake.config.len)]);
    return true;
}

test "security: a valid reflected config is refused before any blob write, including encoded and split forms" {
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
    fake.route = reflectedConfig;
    const secret = "fixture-reflected-token";
    for ([_][]const u8{ secret, "%66ixture-reflected-token", "%2566ixture-reflected-token" }, 0..) |reflection, i| {
        fake.config = try std.fmt.allocPrint(a, "{{\"model_type\": \"qwen4_exp\", \"note\": \"{s}\"}}", .{reflection});
        fake.config_git_hex = gitHex(fake.config);
        var env = try testEnv(a, try fake.endpoint(a));
        try env.put("HF_TOKEN", secret);
        const root = try std.fmt.allocPrint(a, ".tf-pull-reflected-config-{d}-{d}", .{ std.Io.Clock.awake.now(io).toNanoseconds(), i });
        defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
        if (i == 2) {
            const blobs = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs" });
            try std.Io.Dir.cwd().createDirPath(io, blobs);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ blobs, &fake.config_git_hex }), .data = fake.config });
        }
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
        const blob = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", &fake.config_git_hex });
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, blob, .{}));
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.incomplete", .{blob}), .{}));
        try std.testing.expect(std.mem.indexOf(u8, out.written(), secret) == null);
        try std.testing.expect(std.mem.indexOf(u8, err_out.written(), secret) == null);
    }
}

test "security: a ranged body reflection never reaches its artifact" {
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
    const secret = "fixture-range-token";
    fake.weights = try std.mem.concat(a, u8, &.{ "safe-prefix-", secret, "-safe-suffix" });
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const file = ranged.File{
        .url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake.endpoint(a), FakeHub.weights_path }),
        .path = try std.fs.path.join(a, &.{ root, "weights.ranges" }),
        .size = fake.weights.len,
    };
    const auth = @import("pull_auth.zig").Auth{ .bearer = "Bearer fixture-range-token" };
    try std.testing.expectError(error.HubCredentialBody, ranged.fetch(a, io, &.{file}, 7, auth, null));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, file.path, .{}));
}

test "security: an EXLR8 run reflection never reaches its artifact" {
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
    const secret = "fixture-exlr8-token";
    fake.weights = try std.mem.concat(a, u8, &.{ "safe-prefix-", secret, "-safe-suffix" });
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const target = runs.Target{
        .url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake.endpoint(a), FakeHub.weights_path }),
        .path = try std.fs.path.join(a, &.{ root, "width.exlr8" }),
        .size = fake.weights.len,
    };
    const one = runs.Run{ .file = 0, .first = 0, .end = fake.weights.len, .units = .{ 0, 0 } };
    const Accept = struct {
        fn all(_: *const anyopaque, _: runs.Run, _: []const u8) bool {
            return true;
        }
    };
    const context: u8 = 0;
    const check = runs.Check{ .ctx = &context, .run = Accept.all };
    const auth = @import("pull_auth.zig").Auth{ .bearer = "Bearer fixture-exlr8-token" };
    try std.testing.expectError(error.HubCredentialBody, runs.fetch(a, io, &.{target}, &.{one}, check, auth, null));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, target.path, .{}));
}

test "security: artifact containment spans every byte split without refusing benign prefixes" {
    const a = std.testing.allocator;
    const secret = "fixture-unit-token";
    const Auth = @import("pull_auth.zig").Auth;
    for ([_][]const u8{ "safe-fixture-unit-token-end", "safe-%66ixture-unit-token-end", "safe-%252566ixture-unit-token-end" }) |body| {
        for (0..body.len) |cut| {
            var guard: Auth.Body = .init(.{ .bearer = "Bearer fixture-unit-token" });
            defer guard.deinit(a);
            var written: std.ArrayList(u8) = .empty;
            defer written.deinit(a);
            var refused = false;
            if (guard.feed(a, body[0..cut])) |safe| {
                try written.appendSlice(a, safe);
                guard.commit(safe.len);
            } else |err| {
                try std.testing.expectEqual(error.HubCredentialBody, err);
                refused = true;
            }
            if (!refused) {
                if (guard.feed(a, body[cut..])) |safe| {
                    try written.appendSlice(a, safe);
                    guard.commit(safe.len);
                } else |err| {
                    try std.testing.expectEqual(error.HubCredentialBody, err);
                    refused = true;
                }
            }
            try std.testing.expect(refused);
            var matcher = @import("sys").credential.Matcher.init(a, written.items);
            defer matcher.deinit();
            try std.testing.expect(try matcher.spanFrom(0, secret) == null);
        }
    }
    for ([_][]const u8{ "fixture-unit-prefix", "100% raw %66ixture-other", "%66ixture-unit-%" }) |benign| {
        var guard: Auth.Body = .init(.{ .bearer = "Bearer fixture-unit-token" });
        defer guard.deinit(a);
        var written: std.ArrayList(u8) = .empty;
        defer written.deinit(a);
        for (benign) |byte| {
            const safe = try guard.feed(a, &.{byte});
            try written.appendSlice(a, safe);
            guard.commit(safe.len);
        }
        const tail = try guard.finish(a);
        try written.appendSlice(a, tail);
        try std.testing.expectEqualSlices(u8, benign, written.items);
    }
}
