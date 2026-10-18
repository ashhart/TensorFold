//! Login, stored credentials, refusals and the public-checkpoint check against a loopback receiver.
const std = @import("std");
const json = @import("json");
const receipt = @import("benchmark_receipt.zig");
const upload = @import("benchmark_upload.zig");
const fake = @import("benchmark_fake.zig");
const receipts = @import("benchmark_receipt_test.zig");

const token = "synthetic-contributor-token-0001";
const now_ms: i64 = 1767225600000 + 60_000;

const World = struct {
    a: std.mem.Allocator,
    env: std.process.Environ.Map,
    home: []const u8,

    fn init(a: std.mem.Allocator, tmp: *std.testing.TmpDir) !World {
        return .{ .a = a, .env = .init(a), .home = try fake.tmpPath(a, tmp) };
    }

    fn opts(w: *const World, f: *fake.Fake) !upload.Options {
        return .{ .api = try f.url(w.a, ""), .hf_base = try f.url(w.a, "/api/models/"), .environ = &w.env, .home = w.home, .now_ms = now_ms };
    }

    fn auth(w: *const World) ![]const u8 {
        return std.fs.path.join(w.a, &.{ w.home, ".tensorfold", "benchmarks", "auth.json" });
    }
};

test "login stores the token 0600 for its API, never shows it, and logout removes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const w = try World.init(a, &tmp);
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "GET", .path = "/me", .replies = &.{try fake.jsonReply(a, "200 OK", "{\"display_name\": \"example-contributor\"}")} }});
    defer f.stop();
    const o = try w.opts(f);
    var problem: []const u8 = "";
    try std.testing.expectEqualStrings("example-contributor", (try upload.login(a, std.testing.io, o, token, &problem)).?);
    try std.testing.expectEqualStrings("Bearer " ++ token, f.header(0, "authorization").?);
    const path = try w.auth();
    const st = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{});
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode() & 0o777)));
    const want = try std.fmt.allocPrint(a, "{{\"api\": \"{s}\", \"token\": \"{s}\"}}", .{ o.api, token });
    try std.testing.expectEqualStrings(want, try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(4096)));
    try std.testing.expectEqualStrings(token, (try upload.credential(a, std.testing.io, o, &problem)).?);
    try upload.logout(a, std.testing.io, w.home);
    try upload.logout(a, std.testing.io, w.home);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, path, .{}));
    try std.testing.expect(try upload.login(a, std.testing.io, o, "short-token", &problem) == null);
    try std.testing.expectEqualStrings("Invalid benchmark upload token", problem);
}

test "stored credentials refuse broad permissions, another receiver and a malformed file" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const w = try World.init(a, &tmp);
    const path = try w.auth();
    try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path).?);
    const o: upload.Options = .{ .api = "https://example.test/api", .environ = &w.env, .home = w.home, .now_ms = now_ms };
    var problem: []const u8 = "";
    try std.testing.expect(try upload.credential(a, io, o, &problem) == null);
    try std.testing.expectEqualStrings("An upload token is required; run tensorfold benchmark login", problem);
    const cases = [_]struct { []const u8, std.posix.mode_t, []const u8, []const u8 }{
        .{ "{\"api\": \"https://example.test/api\", \"token\": \"" ++ token ++ "\"}", 0o644, "https://example.test/api", "broad file permissions" },
        .{ "{\"api\": \"https://example.test/api\", \"token\": \"" ++ token ++ "\"}", 0o600, "https://another.example.test/api", "different API" },
        .{ "[]", 0o600, "https://example.test/api", "Invalid benchmark credentials" },
    };
    for (cases) |c| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = c[0], .flags = .{ .permissions = .fromMode(c[1]) } });
        try std.Io.Dir.cwd().setFilePermissions(io, path, .fromMode(c[1]), .{});
        var with = o;
        with.api = c[2];
        try std.testing.expect(try upload.credential(a, io, with, &problem) == null);
        try std.testing.expect(std.mem.indexOf(u8, problem, c[3]) != null);
    }
}

test "a refused upload shows a fixed text and never the receiver's, and a redirect is not followed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var w = try World.init(a, &tmp);
    try w.env.put("TENSORFOLD_BENCHMARK_TOKEN", token);
    const public = try fake.jsonReply(a, "200 OK", "{\"private\": false}");
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "GET", .path = "/api/models/example-org/test-model", .replies = &.{public} },
        .{ .method = "POST", .path = "/submissions", .replies = &.{
            try fake.jsonReply(a, "401 Unauthorized", "{\"error\": {\"message\": \"private upstream details must not be printed\"}}"),
            &.{.{ .bytes = "HTTP/1.1 307 Temporary Redirect\r\nLocation: /untrusted-destination\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" }},
            try fake.jsonReply(a, "202 Accepted", "{\"id\": \"synthetic-submission\", \"status\": \"pending\", \"results_url\": null}"),
        } },
    });
    defer f.stop();
    const o = try w.opts(f);
    const v = try receipts.legacy(a);
    var problem: []const u8 = "";
    try std.testing.expect(try upload.upload(a, std.testing.io, o, v, &problem) == null);
    try std.testing.expectEqualStrings("Upload token was refused; run tensorfold benchmark login again", problem);
    try std.testing.expect(try upload.upload(a, std.testing.io, o, v, &problem) == null);
    try std.testing.expectEqualStrings("Benchmark receiver refused the request, HTTP 307", problem);
    const ack = (try upload.upload(a, std.testing.io, o, v, &problem)).?;
    try std.testing.expect(std.mem.eql(u8, ack.id, "synthetic-submission") and std.mem.eql(u8, ack.status, "pending") and ack.results_url == null);
    var posts: usize = 0;
    for (0..f.count()) |i| if (std.mem.eql(u8, f.request(i).method, "POST")) {
        posts += 1;
        try std.testing.expectEqualStrings("/submissions", f.request(i).path);
        try std.testing.expectEqualStrings("Bearer " ++ token, f.header(i, "authorization").?);
        try std.testing.expectEqualStrings(try receipt.canonical(a, v), f.request(i).body);
    };
    try std.testing.expectEqual(@as(usize, 3), posts);
}

test "only a public checkpoint passes, checked without credentials for the model and the drafter" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var w = try World.init(a, &tmp);
    try w.env.put("HF_TOKEN", "synthetic-hf-token-not-forwarded");
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "GET", .path = "/api/models/example-org/test-model", .replies = &.{
            &.{.{ .bytes = "HTTP/1.1 404 Not Found\r\nContent-Length: 15\r\nConnection: close\r\n\r\nprivate details" }},
            try fake.jsonReply(a, "200 OK", "{\"private\": true}"),
            try fake.jsonReply(a, "200 OK", "{}"),
            try fake.jsonReply(a, "200 OK", "{\"private\": false}"),
        } },
        .{ .method = "GET", .path = "/api/models/example-org/test-draft", .replies = &.{try fake.jsonReply(a, "200 OK", "{\"private\": false}")} },
    });
    defer f.stop();
    const o = try w.opts(f);
    const v = try receipts.legacy(a);
    var problem: []const u8 = "";
    for (0..3) |_| {
        try std.testing.expect(!try upload.verifyPublic(a, std.testing.io, o, v, &problem));
        try std.testing.expectEqualStrings("Only publicly accessible Hugging Face checkpoint IDs can be published", problem);
    }
    try receipts.set(a, v, &.{ "model", "drafter_repo_id" }, "\"example-org/test-draft\"");
    try std.testing.expect(try upload.verifyPublic(a, std.testing.io, o, v, &problem));
    try std.testing.expectEqual(@as(usize, 5), f.count());
    for (0..f.count()) |i| {
        try std.testing.expect(f.header(i, "authorization") == null and f.header(i, "cookie") == null);
        try std.testing.expectEqualStrings("GET", f.request(i).method);
    }
    try std.testing.expectEqualStrings("/api/models/example-org/test-draft", f.request(4).path);
}

test "the receiver's reply may be 1 MiB and not one byte more" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const w = try World.init(a, &tmp);
    const head = "{\"display_name\": \"big\", \"pad\": \"";
    var bodies: [2][]const fake.Write = undefined;
    for (&bodies, [_]usize{ receipt.max_bytes, receipt.max_bytes + 1 }) |*b, size| {
        const text = try a.alloc(u8, size);
        @memset(text, 'x');
        @memcpy(text[0..head.len], head);
        @memcpy(text[size - 2 ..], "\"}");
        b.* = try fake.jsonReply(a, "200 OK", text);
    }
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "GET", .path = "/me", .replies = &bodies }});
    defer f.stop();
    const o = try w.opts(f);
    var problem: []const u8 = "";
    try std.testing.expectEqualStrings("big", (try upload.login(a, std.testing.io, o, token, &problem)).?);
    try std.testing.expect(try upload.login(a, std.testing.io, o, token, &problem) == null);
    try std.testing.expectEqualStrings("Benchmark receiver response is too large", problem);
}

test "proxy variables are never used, so the token cannot reach a proxy in clear text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var w = try World.init(a, &tmp);
    const closed = try fake.Fake.start(std.testing.allocator, &.{});
    const dead_port = closed.port;
    closed.stop();
    const api = try std.fmt.allocPrint(a, "https://0.0.0.0:{d}/api", .{dead_port});
    const refused: []const fake.Write = &.{.{ .bytes = "HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" }};
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "CONNECT", .path = try std.fmt.allocPrint(a, "0.0.0.0:{d}", .{dead_port}), .replies = &.{refused} },
        .{ .method = "GET", .path = try std.fmt.allocPrint(a, "{s}/me", .{api}), .replies = &.{try fake.jsonReply(a, "200 OK", "{\"display_name\": \"proxy\"}")} },
    });
    defer f.stop();
    for ([_][]const u8{ "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY" }) |k| try w.env.put(k, try f.url(a, ""));
    const o: upload.Options = .{ .api = api, .hf_base = try std.fmt.allocPrint(a, "https://0.0.0.0:{d}/api/models/", .{dead_port}), .environ = &w.env, .home = w.home, .now_ms = now_ms };
    var problem: []const u8 = "";
    try std.testing.expect(try upload.login(a, std.testing.io, o, token, &problem) == null);
    try std.testing.expectEqualStrings("Could not reach the benchmark receiver; your local receipt is saved", problem);
    try std.testing.expect(!try upload.verifyPublic(a, std.testing.io, o, try receipts.legacy(a), &problem));
    try std.testing.expectEqualStrings("Could not reach huggingface.co to verify public checkpoint metadata; the receipt remains local", problem);
    try std.testing.expectEqual(@as(usize, 0), f.count());
}
