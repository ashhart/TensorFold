//! The Hugging Face token: its sources in order, the one origin it goes to, and a fake gated hub with its CDN.
const std = @import("std");
const Allocator = std.mem.Allocator;
const auth = @import("pull_auth.zig");
const pull = @import("pull.zig");
const single = @import("pull_file.zig");
const ranged = @import("pull_parts.zig");
const exlr8_pull = @import("pull_exlr8.zig");
const FakeHub = @import("pull_test.zig").FakeHub;
const Map = std.process.Environ.Map;

const token = "hf_ExampleGatedToken0001";
const wrong = "hf_ExampleOtherToken0002";
const gated_sha = "fffffffffffffffffffffffffffffffffffffff6";

/// What the gated routes saw: hub requests with the token and without it, CDN requests, and CDN requests with one.
const Seen = struct {
    hub_token: std.atomic.Value(u32) = .init(0),
    hub_other: std.atomic.Value(u32) = .init(0),
    cdn: std.atomic.Value(u32) = .init(0),
    cdn_auth: std.atomic.Value(u32) = .init(0),
};
var seen: Seen = .{};

/// Org/Gated's files need the token, its weights redirect to the CDN host; Org/Private is 401; /denied is 403 anywhere.
fn gated(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return null;
    var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
    _ = parts.next() orelse return null;
    const target = parts.next() orelse return null;
    const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
    const authorization = header(head, "authorization");
    if (std.mem.eql(u8, path, "/denied")) return refuse(fake, conn, "403 Forbidden");
    // the CDN is this fake under another host name: its own weights route serves the bytes, never to a token
    if (std.mem.startsWith(u8, header(head, "host") orelse "", "localhost:")) {
        _ = seen.cdn.fetchAdd(1, .monotonic);
        if (authorization == null) return null;
        _ = seen.cdn_auth.fetchAdd(1, .monotonic);
        return refuse(fake, conn, "400 Bad Request");
    }
    const good = authorization != null and std.mem.eql(u8, authorization.?, "Bearer " ++ token);
    _ = (if (good) &seen.hub_token else &seen.hub_other).fetchAdd(1, .monotonic);
    if (std.mem.startsWith(u8, path, "/api/models/Org/Private/")) return refuse(fake, conn, "401 Unauthorized");
    if (std.mem.eql(u8, path, "/loop")) return moved(fake, conn, "/loop");
    if (std.mem.eql(u8, path, "/moved")) return moved(fake, conn, "/Org/Gated/resolve/" ++ gated_sha ++ "/config.json");
    if (std.mem.eql(u8, path, "/api/models/Org/Gated/revision/main")) {
        fake.respond(conn, "{\"sha\": \"" ++ gated_sha ++ "\"}");
        return true;
    }
    var buf: [512]u8 = undefined;
    if (std.mem.eql(u8, path, "/api/models/Org/Gated/tree/" ++ gated_sha)) {
        const tree = "[{{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"oid\": \"1111111111111111111111111111111111111111\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}]";
        fake.respond(conn, std.fmt.bufPrint(&buf, tree, .{ &fake.config_git_hex, fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len }) catch return false);
        return true;
    }
    if (!std.mem.startsWith(u8, path, "/Org/Gated/resolve/" ++ gated_sha ++ "/")) return null;
    if (authorization == null) return refuse(fake, conn, "401 Unauthorized");
    if (!good) return refuse(fake, conn, "403 Forbidden");
    if (std.mem.endsWith(u8, path, "/config.json")) {
        fake.respond(conn, fake.config);
        return true;
    }
    return moved(fake, conn, std.fmt.bufPrint(&buf, "http://localhost:{d}{s}", .{ fake.port, FakeHub.weights_path }) catch return false);
}

/// A redirect to `location`, the connection kept.
fn moved(fake: *FakeHub, conn: std.Io.net.Stream, location: []const u8) bool {
    var buf: [256]u8 = undefined;
    const text = "HTTP/1.1 302 Found\r\nLocation: {s}\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
    fake.sendAll(conn, std.fmt.bufPrint(&buf, text, .{location}) catch return false);
    return true;
}

var redirect_secret_seen = false;
var tree_secret_seen = false;

fn reflectedRedirect(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const target = requestTarget(head);
    if (std.mem.eql(u8, target, "/reflect")) return moved(fake, conn, "/collect?credential=" ++ token);
    if (std.mem.startsWith(u8, target, "/collect")) {
        redirect_secret_seen = true;
        fake.respond(conn, "{}");
        return true;
    }
    return null;
}

fn reflectedTree(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const target = requestTarget(head);
    if (std.mem.eql(u8, target, "/api/models/Org/Flash/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1?recursive=true")) {
        var buf: [512]u8 = undefined;
        const tree = std.fmt.bufPrint(&buf, "[{{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"{s}\", \"size\": 0}}]", .{ &fake.config_git_hex, fake.config.len, token }) catch return false;
        fake.respond(conn, tree);
        return true;
    }
    if (std.mem.indexOf(u8, target, token) != null) {
        tree_secret_seen = true;
        fake.respond(conn, "");
        return true;
    }
    return null;
}

/// A refusal with a body the hub might send, on a connection that then closes.
fn refuse(fake: *FakeHub, conn: std.Io.net.Stream, status: []const u8) bool {
    const body = "{\"error\": \"Access to this repo is restricted\"}";
    var buf: [192]u8 = undefined;
    const text = "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}";
    fake.sendAll(conn, std.fmt.bufPrint(&buf, text, .{ status, body.len, body }) catch return false);
    return false;
}

fn header(head: []const u8, name: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, head, "\r\n");
    _ = it.next();
    while (it.next()) |h| {
        const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), name)) return std.mem.trim(u8, h[colon + 1 ..], " ");
    }
    return null;
}

/// The fake under the name localhost, the CDN's host; a system that cannot reach it skips the test.
fn cdnReachable(io: std.Io, port: u16) !void {
    const host = try std.Io.net.HostName.init("localhost");
    const probe = host.connect(io, port, .{ .mode = .stream }) catch return error.SkipZigTest;
    probe.close(io);
}

fn gatedEnv(a: Allocator, fake: *FakeHub, with: ?[]const u8) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", try fake.endpoint(a));
    if (with) |t| try env.put("HF_TOKEN", t);
    return env;
}

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}

fn requestTarget(head: []const u8) []const u8 {
    const start = std.mem.indexOfScalar(u8, head, ' ') orelse return "";
    const rest = head[start + 1 ..];
    return rest[0 .. std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len];
}

/// No file under `root` holds `secret`: not a blob, a snapshot link's target, a ref nor a partial download.
fn expectNoFileHolds(a: Allocator, io: std.Io, root: []const u8, secret: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(a);
    defer walker.deinit();
    var files: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        files += 1;
        try std.testing.expect(!has(try entry.dir.readFileAlloc(io, entry.basename, a, .limited(1 << 20)), secret));
    }
    try std.testing.expect(files > 0);
}

/// Org/Gated pulled with the token at `sizes`: its files land whole, and no output or file shows the token.
fn pullGated(a: Allocator, io: std.Io, fake: *FakeHub, cache: []const u8, sizes: pull.Sizes) !void {
    const env = try gatedEnv(a, fake, token);
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out.writer, &err_out.writer, &env, cache, "Org/Gated", sizes));
    const snapshot = try std.fs.path.join(a, &.{ cache, "models--Org--Gated/snapshots", gated_sha });
    const weights = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" }), a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, fake.weights, weights);
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "config.json" }), a, .limited(1 << 20));
    try std.testing.expectEqualStrings(fake.config, config);
    try std.testing.expect(!has(out.written(), token) and !has(err_out.written(), token));
    try expectNoFileHolds(a, io, cache, token);
    // every hub request carried the token, and no CDN request did
    try std.testing.expect(seen.hub_token.load(.monotonic) > 0);
    try std.testing.expectEqual(@as(u32, 0), seen.hub_other.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), seen.cdn_auth.load(.monotonic));
}

test "security: a gated repo comes down with HF_TOKEN, sent to the hub's host and never after the CDN redirect" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = gated;
    fake.start();
    defer fake.stopServer();
    try cdnReachable(io, fake.port);
    seen = .{};
    const root = try std.fmt.allocPrint(a, ".tf-pull-gated-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try pullGated(a, io, &fake, root, .{});
    try std.testing.expectEqual(@as(u32, 1), seen.cdn.load(.monotonic)); // the weights, from the CDN
}

test "security: a hub cannot reflect HF_TOKEN or controls through model_type errors" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-reflected-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try gatedEnv(a, &fake, token);
    var shown: std.ArrayList(u8) = .empty;
    for ([_][]const u8{ token, token ++ "\\u001b[31m" }) |model_type| {
        fake.config = try std.fmt.allocPrint(a, "{{\"model_type\": \"{s}\"}}", .{model_type});
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
        try shown.appendSlice(a, out.written());
        try shown.appendSlice(a, err_out.written());
    }
    try std.testing.expect(!has(shown.items, token));
    try std.testing.expect(!has(shown.items, "\x1b"));
    try std.testing.expect(has(shown.items, "[redacted]"));
}

test "security: a reflected hub token never enters a redirect request target" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedRedirect;
    fake.start();
    defer fake.stopServer();
    redirect_secret_seen = false;
    const hub: auth.Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ token, .source = "HF_TOKEN" };
    var client = @import("pull_http.zig").client(a, std.testing.io);
    defer client.deinit();
    var buf: [8 << 10]u8 = undefined;
    var req: std.http.Client.Request = undefined;
    _ = hub.send(&client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "{s}/reflect", .{hub.endpoint})), .{}, &buf) catch |err| {
        try std.testing.expectEqual(error.HubCredentialRedirect, err);
        try std.testing.expect(!redirect_secret_seen);
        return;
    };
    req.deinit();
    return error.TestExpectedError;
}

test "security: a reflected hub token in a tree path is refused before output or URL use" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedTree;
    fake.start();
    defer fake.stopServer();
    tree_secret_seen = false;
    const root = try std.fmt.allocPrint(a, ".tf-pull-tree-secret-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try gatedEnv(a, &fake, token);
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expect(!tree_secret_seen);
    try std.testing.expect(!has(out.written(), token) and !has(err_out.written(), token));
}

test "a gated repo's large file comes down as pieces, each redirected to the CDN without the token" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = gated;
    fake.start();
    defer fake.stopServer();
    try cdnReachable(io, fake.port);
    seen = .{};
    const root = try std.fmt.allocPrint(a, ".tf-pull-gated-pieces-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try pullGated(a, io, &fake, root, .{ .threshold = 4096, .piece = 1250 });
    try std.testing.expectEqual(@as(u32, 4), seen.cdn.load(.monotonic)); // four pieces, four redirects
    // a piece the hub refuses is not asked for again
    const before = seen.hub_other.load(.monotonic);
    const url = try std.fmt.allocPrint(a, "{s}/Org/Gated/resolve/{s}/weights.safetensors", .{ try fake.endpoint(a), gated_sha });
    const file: ranged.File = .{ .url = url, .path = try std.fs.path.join(a, &.{ root, "refused.ranges" }), .size = fake.weights.len };
    try std.testing.expectError(error.HubDenied, ranged.fetch(a, io, &.{file}, 1 << 20, .{ .endpoint = try fake.endpoint(a) }, null));
    try std.testing.expectEqual(before + 1, seen.hub_other.load(.monotonic));
}

test "security: a gated or private repo is refused by name, and a token without access is called that" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = gated;
    fake.start();
    defer fake.stopServer();
    seen = .{};
    const root = try std.fmt.allocPrint(a, ".tf-pull-refused-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const anonymous = try gatedEnv(a, &fake, null);
    for ([_][]const u8{ "Org/Gated", "Org/Private" }) |repo| {
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &anonymous, root, repo));
        const text = err_out.written();
        try std.testing.expect(has(text, try std.fmt.allocPrint(a, "{s} was refused: it is gated or private, or no repo has that name", .{repo})));
        try std.testing.expect(has(text, "accept its terms on 127.0.0.1"));
        try std.testing.expect(has(text, "set HF_TOKEN or log in with `hf auth login`"));
    }
    const other = try gatedEnv(a, &fake, wrong);
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &other, root, "Org/Gated"));
    try std.testing.expect(has(err_out.written(), "Org/Gated is gated or private, and the token from HF_TOKEN lacks access"));
    try std.testing.expect(!has(err_out.written(), wrong) and !has(out.written(), wrong));
    // each gated refusal took the revision, the tree and config.json, the private one its revision; none the CDN
    try std.testing.expectEqual(@as(u32, 7), seen.hub_other.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), seen.cdn.load(.monotonic));
    // a download the hub refuses is not tried again
    const public_hub: auth.Auth = .{ .endpoint = try fake.endpoint(a) };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    const url = try std.fmt.allocPrint(a, "{s}/Org/Gated/resolve/{s}/config.json", .{ public_hub.endpoint, gated_sha });
    try std.testing.expectError(error.HubDenied, single.downloadRetrying(a, io, &client, public_hub, &out.writer, url, root, .{ .path = "config.json", .size = fake.config.len }));
    try std.testing.expectEqual(@as(u32, 8), seen.hub_other.load(.monotonic));
}

fn expectToken(a: Allocator, io: std.Io, env: *const Map, value: []const u8, source: []const u8) !void {
    var problem: []const u8 = "";
    const t = (try auth.token(a, io, env, &problem)).?;
    try std.testing.expectEqualStrings(value, t.value);
    try std.testing.expectEqualStrings(source, t.source);
}

test "the token is HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or the one file the hf tools read: HF_TOKEN_PATH, HF_HOME's, home's" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    for ([_][]const u8{ "hf-home", "home/.cache/huggingface", "profile/.cache/huggingface" }) |d| try tmp.dir.createDirPath(io, d);
    // the files' own whitespace and line ends are trimmed
    try tmp.dir.writeFile(io, .{ .sub_path = "path-token", .data = "  hf_FromTokenPath\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hf-home/token", .data = "hf_FromHfHome\r\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "home/.cache/huggingface/token", .data = "\thf_FromHomeCache \n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "profile/.cache/huggingface/token", .data = "hf_FromProfileCache" });
    const path_token = try std.fs.path.join(a, &.{ dir, "path-token" });
    var env: Map = .init(a);
    try env.put("HF_TOKEN", " hf_FromEnv\n");
    try env.put("HUGGING_FACE_HUB_TOKEN", "hf_FromOldEnv");
    try env.put("HF_TOKEN_PATH", path_token);
    try env.put("HF_HOME", try std.fs.path.join(a, &.{ dir, "hf-home" }));
    try env.put("HOME", try std.fs.path.join(a, &.{ dir, "home" }));
    try expectToken(a, io, &env, "hf_FromEnv", "HF_TOKEN");
    try env.put("HF_TOKEN", " \t\n"); // blank is unset
    try expectToken(a, io, &env, "hf_FromOldEnv", "HUGGING_FACE_HUB_TOKEN");
    _ = env.swapRemove("HUGGING_FACE_HUB_TOKEN");
    try expectToken(a, io, &env, "hf_FromTokenPath", path_token);
    // only the one file is read: a missing or blank one means no token, not the next folder's
    var problem: []const u8 = "";
    try env.put("HF_TOKEN_PATH", try std.fs.path.join(a, &.{ dir, "missing" }));
    try std.testing.expect(try auth.token(a, io, &env, &problem) == null);
    _ = env.swapRemove("HF_TOKEN_PATH");
    try expectToken(a, io, &env, "hf_FromHfHome", try std.fs.path.join(a, &.{ dir, "hf-home", "token" }));
    try tmp.dir.writeFile(io, .{ .sub_path = "hf-home/token", .data = "\n" });
    try std.testing.expect(try auth.token(a, io, &env, &problem) == null);
    _ = env.swapRemove("HF_HOME");
    try expectToken(a, io, &env, "hf_FromHomeCache", try std.fs.path.join(a, &.{ dir, "home", ".cache", "huggingface", "token" }));
    // Windows sets USERPROFILE, not HOME, as the cache folder reads it
    _ = env.swapRemove("HOME");
    try env.put("USERPROFILE", try std.fs.path.join(a, &.{ dir, "profile" }));
    try expectToken(a, io, &env, "hf_FromProfileCache", try std.fs.path.join(a, &.{ dir, "profile", ".cache", "huggingface", "token" }));
    // an empty value is unset, so no file named token is read where pull runs
    for ([_][]const u8{ "HF_TOKEN_PATH", "HF_HOME", "USERPROFILE" }) |name| try env.put(name, "");
    try std.testing.expect(try auth.token(a, io, &env, &problem) == null);
    var none: Map = .init(a);
    try std.testing.expect(try auth.token(a, io, &none, &problem) == null);
}

test "security: a token pull cannot send is refused by its source, and never shown" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "two-lines", .data = "hf_ExampleLineOne\nhf_ExampleLineTwo\n" });
    const big: [5000]u8 = @splat('h');
    try tmp.dir.writeFile(io, .{ .sub_path = "big", .data = &big });
    const Case = struct { name: []const u8, value: []const u8, says: []const u8 };
    const cases = [_]Case{
        .{ .name = "HF_TOKEN", .value = "hf_Example Spaced", .says = "the Hugging Face token in HF_TOKEN has a space or control character" },
        .{ .name = "HUGGING_FACE_HUB_TOKEN", .value = "hf_Example\x01Control", .says = "in HUGGING_FACE_HUB_TOKEN has a space or control character" },
        .{ .name = "HF_TOKEN_PATH", .value = try std.fs.path.join(a, &.{ dir, "two-lines" }), .says = "two-lines has a space or control character" },
        .{ .name = "HF_TOKEN_PATH", .value = try std.fs.path.join(a, &.{ dir, "big" }), .says = "big is larger than a token" },
        .{ .name = "HF_TOKEN_PATH", .value = dir, .says = "cannot be read (" },
        .{ .name = "HF_TOKEN_PATH", .value = "hf_Example\x1btoken", .says = "HF_TOKEN_PATH has a control character" }, // a folder: the error's name is the system's
    };
    for (cases) |c| {
        var env: Map = .init(a);
        try env.put(c.name, c.value);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expect(try auth.Auth.init(a, io, &env, &err_out.writer) == null);
        const text = err_out.written();
        try std.testing.expect(has(text, c.says));
        try std.testing.expect(!has(text, "hf_Example") and !has(text, "hhhh"));
    }
    // pull stops there, before it asks the hub anything (here a closed loopback port, never the network)
    var env: Map = .init(a);
    try env.put("HF_ENDPOINT", "http://127.0.0.1:9");
    try env.put("HF_TOKEN", "hf_Example Spaced");
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, dir, "Org/Name"));
    try std.testing.expectEqualStrings("", out.written());
    try std.testing.expect(has(err_out.written(), "in HF_TOKEN has a space") and !has(err_out.written(), "hf_Example"));
}

test "security: endpoint userinfo never appears in connection errors" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const user = "fixture-endpoint-user";
    const password = "fixture-endpoint-password";
    var env: Map = .init(a);
    try env.put("HF_ENDPOINT", try std.fmt.allocPrint(a, "http://{s}:{s}@127.0.0.1:9", .{ user, password }));
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, dir, "Org/Name"));
    for ([_][]const u8{ user, password }) |secret| {
        try std.testing.expect(!has(out.written(), secret));
        try std.testing.expect(!has(err_out.written(), secret));
    }
    try std.testing.expect(has(err_out.written(), "HF_ENDPOINT"));
}

test "security: the token goes only to the endpoint's own scheme, host and port" {
    const bearer = "Bearer hf_ExampleToken";
    const public_hub: auth.Auth = .{ .bearer = bearer, .source = "HF_TOKEN" };
    for ([_][]const u8{ "https://huggingface.co/api/models/Org/Name", "https://HuggingFace.co/Org/Name/resolve/main/config.json", "https://huggingface.co:443/x" }) |url| {
        const v = public_hub.header(try std.Uri.parse(url));
        try std.testing.expect(v == .override and std.mem.eql(u8, v.override, bearer));
    }
    // another host, a subdomain, a lookalike, plain HTTP and another port each lose it
    for ([_][]const u8{ "https://cdn-lfs.huggingface.co/x", "https://huggingface.co.example.com/x", "http://huggingface.co/x", "https://huggingface.co:8443/x", "https://example.com/huggingface.co" }) |url| {
        try std.testing.expect(public_hub.header(try std.Uri.parse(url)) == .default);
    }
    const mirror: auth.Auth = .{ .endpoint = "https://mirror.example.com/hf", .bearer = bearer, .source = "HF_TOKEN" };
    try std.testing.expect(mirror.header(try std.Uri.parse("https://mirror.example.com/hf/api/models/Org/Name")) == .override);
    try std.testing.expect(mirror.header(try std.Uri.parse("https://huggingface.co/api/models/Org/Name")) == .default);
    const anonymous: auth.Auth = .{};
    try std.testing.expect(anonymous.header(try std.Uri.parse("https://huggingface.co/api/models/Org/Name")) == .default);
}

test "security: a redirect on the hub keeps the token, another host's 403 is no refusal, and three hops is the most" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = gated;
    fake.start();
    defer fake.stopServer();
    try cdnReachable(io, fake.port);
    seen = .{};
    const hub: auth.Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ token, .source = "HF_TOKEN" };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    var buf: [8 << 10]u8 = undefined;
    var req: std.http.Client.Request = undefined;
    // the gated file answers only to the token, so the relative redirect kept it
    const kept = try hub.send(&client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "{s}/moved", .{hub.endpoint})), .{}, &buf);
    try std.testing.expectEqual(std.http.Status.ok, kept.head.status);
    req.deinit();
    try std.testing.expectError(error.HubDenied, hub.send(&client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "{s}/denied", .{hub.endpoint})), .{}, &buf));
    const elsewhere = try hub.send(&client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "http://localhost:{d}/denied", .{fake.port})), .{}, &buf);
    try std.testing.expectEqual(std.http.Status.forbidden, elsewhere.head.status);
    req.deinit();
    try std.testing.expectError(error.TooManyHttpRedirects, hub.send(&client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "{s}/loop", .{hub.endpoint})), .{}, &buf));
    try std.testing.expectEqual(@as(u32, 0), seen.hub_other.load(.monotonic));
}

test "an EXLR8 node's whole files of a gated repo resolve with the token, and come from the CDN without it" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = gated;
    fake.start();
    defer fake.stopServer();
    try cdnReachable(io, fake.port);
    seen = .{};
    const dir = try std.fmt.allocPrint(a, ".tf-pull-gated-exlr8-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    var sha256: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&sha256, fake.weights_sha_hex);
    var git_sha1: [20]u8 = undefined;
    _ = try std.fmt.hexToBytes(&git_sha1, &fake.config_git_hex);
    const whole = [_]single.Entry{
        .{ .path = "config.json", .size = fake.config.len, .git_sha1 = git_sha1 },
        .{ .path = "weights.safetensors", .size = fake.weights.len, .sha256 = sha256 },
    };
    const hub: auth.Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ token, .source = "HF_TOKEN" };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u32, 2), try exlr8_pull.fetchWhole(a, io, &client, &out.writer, &err_out.writer, hub, "Org/Gated", gated_sha, dir, &whole));
    const cwd = std.Io.Dir.cwd();
    try std.testing.expectEqualStrings(fake.config, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20)));
    try std.testing.expectEqualSlices(u8, fake.weights, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.safetensors" }), a, .limited(1 << 20)));
    // the weights' link was resolved on the hub with the token, then fetched from the CDN without it
    try std.testing.expect(seen.hub_token.load(.monotonic) >= 2);
    try std.testing.expectEqual(@as(u32, 0), seen.hub_other.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 1), seen.cdn.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), seen.cdn_auth.load(.monotonic));
}
