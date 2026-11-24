//! KP50 hub regressions: EXLR8 path containment and credential reflection/redaction.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");
const auth_mod = @import("pull_auth.zig");
const Auth = auth_mod.Auth;
const exlr8_pull = @import("pull_exlr8.zig");
const runs = @import("pull_runs.zig");
const single = @import("pull_file.zig");
const FakeHub = @import("pull_test.zig").FakeHub;
const Map = std.process.Environ.Map;
const credential = @import("sys").credential;

const redaction_marker = "[redacted]";

const sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1";
const plain_token = "fixture-token";
const literal_token = "fixture%2Ftoken";

fn target(head: []const u8) []const u8 {
    const start = std.mem.indexOfScalar(u8, head, ' ') orelse return "";
    const rest = head[start + 1 ..];
    return rest[0 .. std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len];
}

fn method(head: []const u8) []const u8 {
    return head[0 .. std.mem.indexOfScalar(u8, head, ' ') orelse return ""];
}

fn redirect(fake: *FakeHub, conn: std.Io.net.Stream, location: []const u8) bool {
    var buf: [8192]u8 = undefined;
    fake.sendAll(conn, std.fmt.bufPrint(&buf, "HTTP/1.1 302 Found\r\nLocation: {s}\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n", .{location}) catch return false);
    return true;
}

fn encoded(a: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (text) |c| try out.print(a, "%{X:0>2}", .{c});
    return out.items;
}

fn envWithToken(a: Allocator, fake: *FakeHub, token: []const u8) !Map {
    var env: Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", try fake.endpoint(a));
    try env.put("HF_TOKEN", token);
    return env;
}

fn expectAbsent(text: []const u8, secret: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

fn expectFinalContainment(a: Allocator, out: []const u8, err_out: []const u8, secret: []const u8) !void {
    for ([_][]const u8{ out, err_out }) |text| {
        var matcher = credential.Matcher.init(a, text);
        defer matcher.deinit();
        try std.testing.expect((try matcher.spanFrom(0, secret)) == null);
    }
}

// Pull configures one active token, so marker synthesis stays within that one-token API.
fn markerModelType(a: Allocator, io: std.Io, fake: *FakeHub, root: []const u8, secret: []const u8, model_type: []const u8) !void {
    const env = try envWithToken(a, fake, secret);
    fake.config = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{model_type});
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    try expectFinalContainment(a, out.written(), err_out.written(), secret);
}

fn expectNoFile(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    return error.UnsafeHubPathCreatedFile;
}

// Finding 1: even a valid LFS response must not make an unsafe path reachable.
var traversal_requests: std.atomic.Value(u32) = .init(0);

fn traversalFixture(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const prefix = "/Org/Flash/resolve/" ++ sha ++ "/";
    if (!std.mem.startsWith(u8, target(head), prefix)) return null;
    _ = traversal_requests.fetchAdd(1, .monotonic);
    if (std.mem.eql(u8, method(head), "HEAD")) {
        fake.sendAll(conn, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n");
        return true;
    }
    var buf: [256]u8 = undefined;
    fake.sendAll(conn, std.fmt.bufPrint(&buf, "HTTP/1.1 206 Partial Content\r\nContent-Length: {d}\r\nContent-Range: bytes 0-{d}/{d}\r\nConnection: keep-alive\r\n\r\n", .{ fake.weights.len, fake.weights.len - 1, fake.weights.len }) catch return false);
    fake.sendAll(conn, fake.weights);
    return true;
}

test "security KP50-1: EXLR8 LFS paths obey snapshotPathOk before filesystem or URL use" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = traversalFixture;
    fake.start();
    defer fake.stopServer();
    const hub: Auth = .{ .endpoint = try fake.endpoint(a) };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    var digest: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&digest, fake.weights_sha_hex);

    const stamp = std.Io.Clock.awake.now(io).toNanoseconds();
    const sandbox = try std.fmt.allocPrint(a, ".tf-pull-kp50-paths-{d}", .{stamp});
    defer std.Io.Dir.cwd().deleteTree(io, sandbox) catch {};
    const names = [_][]const u8{
        "../outside",               "absolute",                           "a/../b",                      "..\\outside",
        ".tensorfold-save-pending", "nested/.TensorFold-Original/graphs", "weights.tensorfold-rollback",
    };
    for (names, 0..) |name, i| {
        const case_root = try std.fmt.allocPrint(a, "{s}/case-{d}", .{ sandbox, i });
        const dir = try std.fs.path.join(a, &.{ case_root, "destination" });
        const absolute = try std.fmt.allocPrint(a, "/{s}/case-{d}/absolute-outside", .{ sandbox, i });
        const path = if (std.mem.eql(u8, name, "absolute")) absolute else name;
        const entry = single.Entry{ .path = path, .size = fake.weights.len, .sha256 = digest };
        const final = try std.fs.path.join(a, &.{ dir, path });
        traversal_requests.store(0, .monotonic);
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        const result = exlr8_pull.fetchWhole(a, io, &client, &out.writer, &err_out.writer, hub, "Org/Flash", sha, dir, &.{entry});
        if (result) |_| {
            try expectNoFile(io, final);
            if (std.mem.eql(u8, name, "absolute")) try expectNoFile(io, absolute);
            try std.testing.expectEqual(@as(u32, 0), traversal_requests.load(.monotonic));
            return error.UnsafeExlr8PathAccepted;
        } else |err| {
            try std.testing.expectEqual(error.SnapshotPath, err);
        }
        try expectNoFile(io, final);
        if (std.mem.eql(u8, name, "../outside")) {
            try expectNoFile(io, try std.fs.path.join(a, &.{ case_root, "outside" }));
        }
        if (std.mem.eql(u8, name, "absolute")) try expectNoFile(io, absolute);
        try std.testing.expectEqual(@as(u32, 0), traversal_requests.load(.monotonic));
    }
}

// Finding 4: runs.resolve handles this redirect itself rather than in Auth.send.
var head_location: []const u8 = "";
var requests_after_head: std.atomic.Value(u32) = .init(0);

fn headRedirectFixture(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    if (std.mem.eql(u8, target(head), "/head") and std.mem.eql(u8, method(head), "HEAD")) {
        return redirect(fake, conn, head_location);
    }
    _ = requests_after_head.fetchAdd(1, .monotonic);
    fake.respond(conn, "{}");
    return true;
}

test "security KP50-4: EXLR8 HEAD redirects refuse credentials in every absolute URI component" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = headRedirectFixture;
    fake.start();
    defer fake.stopServer();
    const hub: Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ plain_token, .source = "HF_TOKEN" };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    const coded = try encoded(a, plain_token);
    for ([_][]const u8{ plain_token, coded }) |secret| {
        const authority = try std.fmt.allocPrint(a, "127.0.0.1:{d}", .{fake.port});
        const locations = [_][]const u8{
            try std.fmt.allocPrint(a, "http://{s}@{s}/cdn", .{ secret, authority }),
            try std.fmt.allocPrint(a, "http://{s}/cdn/{s}", .{ authority, secret }),
            try std.fmt.allocPrint(a, "http://{s}/cdn?credential={s}", .{ authority, secret }),
            try std.fmt.allocPrint(a, "http://{s}/cdn#{s}", .{ authority, secret }),
        };
        for (locations) |location| {
            head_location = location;
            requests_after_head.store(0, .monotonic);
            const result = runs.resolve(a, &client, hub, try std.fmt.allocPrint(a, "{s}/head", .{hub.endpoint}));
            var caught: ?anyerror = null;
            if (result) |_| {} else |err| caught = err;
            try std.testing.expectEqual(@as(u32, 0), requests_after_head.load(.monotonic));
            try std.testing.expect(caught != null);
            try std.testing.expectEqual(error.HubCredentialRedirect, caught.?);
        }
    }
}

// Finding 5 records raw request targets and all displayed text for each attack.
const Attack = struct {
    secret: []const u8,
    payload: []const u8,
    variants: [4][]const u8,
};

fn mixedDepthAttack() Attack {
    return .{
        .secret = literal_token,
        .payload = "%2566ixture%2Ftoken",
        .variants = .{ "%66ixture/token", "fixture/token", "%66ixture%2Ftoken", "%2566ixture/token" },
    };
}

var attack_payload: []const u8 = "";
var attack_requests: std.atomic.Value(u32) = .init(0);
var request_log: [4096]u8 = undefined;
var request_log_len: usize = 0;

fn noteAttack(head: []const u8) void {
    _ = attack_requests.fetchAdd(1, .monotonic);
    const text = target(head);
    request_log_len = @min(text.len, request_log.len);
    @memcpy(request_log[0..request_log_len], text[0..request_log_len]);
}

fn reflectedRedirectFixture(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    if (std.mem.eql(u8, target(head), "/head") and std.mem.eql(u8, method(head), "HEAD"))
        return redirect(fake, conn, std.fmt.allocPrint(fake.allocator, "/collect?credential={s}", .{attack_payload}) catch return false);
    if (std.mem.eql(u8, target(head), "/reflect")) {
        return redirect(fake, conn, std.fmt.allocPrint(fake.allocator, "/collect?credential={s}", .{attack_payload}) catch return false);
    }
    if (std.mem.startsWith(u8, target(head), "/collect")) {
        noteAttack(head);
        fake.respond(conn, "{}");
        return true;
    }
    return null;
}

fn reflectedTreeFixture(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool {
    const at = target(head);
    if (std.mem.eql(u8, at, "/api/models/Org/Flash/tree/" ++ sha ++ "?recursive=true")) {
        var buf: [4096]u8 = undefined;
        const tree = std.fmt.bufPrint(&buf, "[{{\"type\":\"file\",\"oid\":\"{s}\",\"path\":\"config.json\",\"size\":{d}}},{{\"type\":\"file\",\"path\":\"weights-{s}\",\"size\":0}}]", .{ &fake.config_git_hex, fake.config.len, attack_payload }) catch return false;
        fake.respond(conn, tree);
        return true;
    }
    if (std.mem.startsWith(u8, at, "/Org/Flash/resolve/" ++ sha ++ "/weights-")) {
        noteAttack(head);
        fake.respond(conn, "");
        return true;
    }
    return null;
}

fn resetAttack(payload: []const u8) void {
    attack_payload = payload;
    attack_requests.store(0, .monotonic);
    request_log_len = 0;
}

fn expectAttackAbsent(attack: Attack, out: []const u8, err_out: []const u8) !void {
    try expectAbsent(request_log[0..request_log_len], attack.payload);
    try expectAbsent(out, attack.payload);
    try expectAbsent(err_out, attack.payload);
    for (attack.variants) |variant| {
        try expectAbsent(request_log[0..request_log_len], variant);
        try expectAbsent(out, variant);
        try expectAbsent(err_out, variant);
    }
}

fn redirectAttack(a: Allocator, client: *std.http.Client, hub: Auth, attack: Attack) !void {
    resetAttack(attack.payload);
    var buf: [8192]u8 = undefined;
    var req: std.http.Client.Request = undefined;
    const result = hub.send(client, &req, .GET, try std.Uri.parse(try std.fmt.allocPrint(a, "{s}/reflect", .{hub.endpoint})), .{}, &buf);
    var caught: ?anyerror = null;
    if (result) |_| req.deinit() else |err| caught = err;
    try expectAttackAbsent(attack, request_log[0..request_log_len], "");
    try std.testing.expectEqual(@as(u32, 0), attack_requests.load(.monotonic));
    try std.testing.expect(caught != null);
    try std.testing.expectEqual(error.HubCredentialRedirect, caught.?);
}

fn treeAttack(a: Allocator, io: std.Io, fake: *FakeHub, root: []const u8, attack: Attack) !void {
    resetAttack(attack.payload);
    const env = try envWithToken(a, fake, attack.secret);
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    const status = try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash");
    try expectAttackAbsent(attack, out.written(), err_out.written());
    try std.testing.expectEqual(@as(u32, 0), attack_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u8, 1), status);
}

test "security KP50-5: literal-percent and nested-encoded tokens are contained in redirects" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedRedirectFixture;
    fake.start();
    defer fake.stopServer();
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();

    const plain = [4][]const u8{
        plain_token,                                 try encoded(a, plain_token),
        try encoded(a, try encoded(a, plain_token)), try encoded(a, try encoded(a, try encoded(a, plain_token))),
    };
    const literal = [4][]const u8{
        literal_token,                                 try encoded(a, literal_token),
        try encoded(a, try encoded(a, literal_token)), try encoded(a, try encoded(a, try encoded(a, literal_token))),
    };
    const attacks = [_]Attack{
        .{ .secret = plain_token, .payload = plain[0], .variants = plain },
        .{ .secret = plain_token, .payload = "%66ixture-token", .variants = plain },
        .{ .secret = plain_token, .payload = plain[1], .variants = plain },
        .{ .secret = plain_token, .payload = plain[2], .variants = plain },
        .{ .secret = plain_token, .payload = plain[3], .variants = plain },
        .{ .secret = literal_token, .payload = literal[0], .variants = literal },
        .{ .secret = literal_token, .payload = "%66ixture%2Ftoken", .variants = literal },
        .{ .secret = literal_token, .payload = literal[1], .variants = literal },
        .{ .secret = literal_token, .payload = literal[2], .variants = literal },
        .{ .secret = literal_token, .payload = literal[3], .variants = literal },
    };
    for (attacks) |attack| {
        const hub: Auth = .{ .endpoint = try fake.endpoint(a), .bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{attack.secret}), .source = "HF_TOKEN" };
        try redirectAttack(a, &client, hub, attack);
    }
}

test "security KP50-5: literal-percent and nested-encoded tokens are contained in tree paths" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedTreeFixture;
    fake.start();
    defer fake.stopServer();

    const plain = [4][]const u8{
        plain_token,                                 try encoded(a, plain_token),
        try encoded(a, try encoded(a, plain_token)), try encoded(a, try encoded(a, try encoded(a, plain_token))),
    };
    const literal = [4][]const u8{
        literal_token,                                 try encoded(a, literal_token),
        try encoded(a, try encoded(a, literal_token)), try encoded(a, try encoded(a, try encoded(a, literal_token))),
    };
    const attacks = [_]Attack{
        .{ .secret = plain_token, .payload = plain[0], .variants = plain },
        .{ .secret = plain_token, .payload = "%66ixture-token", .variants = plain },
        .{ .secret = plain_token, .payload = plain[1], .variants = plain },
        .{ .secret = plain_token, .payload = plain[2], .variants = plain },
        .{ .secret = plain_token, .payload = plain[3], .variants = plain },
        .{ .secret = literal_token, .payload = literal[0], .variants = literal },
        .{ .secret = literal_token, .payload = literal[1], .variants = literal },
        .{ .secret = literal_token, .payload = "%66ixture%2Ftoken", .variants = literal },
        .{ .secret = literal_token, .payload = literal[2], .variants = literal },
        .{ .secret = literal_token, .payload = literal[3], .variants = literal },
    };
    for (attacks, 0..) |attack, i| {
        const root = try std.fmt.allocPrint(a, ".tf-pull-kp50-tree-{d}-{d}", .{ std.Io.Clock.awake.now(io).toNanoseconds(), i });
        defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
        try treeAttack(a, io, &fake, root, attack);
    }
}

test "security KP50-6: percent-encoded credentials are redacted from model_type" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const once = try encoded(a, plain_token);
    const twice = try encoded(a, once);
    const thrice = try encoded(a, twice);
    const root = try std.fmt.allocPrint(a, ".tf-pull-kp50-model-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try envWithToken(a, &fake, plain_token);
    for ([_][]const u8{ once, twice, thrice }) |model_type| {
        fake.config = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{model_type});
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
        for ([_][]const u8{ plain_token, once, twice, thrice }) |variant| {
            try expectAbsent(out.written(), variant);
            try expectAbsent(err_out.written(), variant);
        }
        try std.testing.expect(std.mem.indexOf(u8, err_out.written(), "[redacted]") != null);
    }
    const encoded_percent = "%2525";
    fake.config = "{\"model_type\":\"%2525\"}";
    const percent_env = try envWithToken(a, &fake, "%25");
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &percent_env, root, "Org/Flash"));
    for ([_][]const u8{ out.written(), err_out.written() }) |text| {
        try expectAbsent(text, encoded_percent);
        try expectAbsent(text, "[redacted]25");
    }
    try std.testing.expect(std.mem.indexOf(u8, err_out.written(), "[redacted]") != null);
}

// KR66 finding 2: independent character depths must not defeat containment.
test "security KR66-2: mixed-depth hub credentials are refused in automatic redirects" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedRedirectFixture;
    fake.start();
    defer fake.stopServer();
    var client = @import("pull_http.zig").client(a, std.testing.io);
    defer client.deinit();
    const hub: Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ literal_token, .source = "HF_TOKEN" };
    try redirectAttack(a, &client, hub, mixedDepthAttack());
}

test "security KR66-2: mixed-depth hub credentials are refused in tree paths" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedTreeFixture;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr66-mixed-tree-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try treeAttack(a, io, &fake, root, mixedDepthAttack());
}

test "security KR66-2: mixed-depth hub credentials are refused in EXLR8 HEAD redirects" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = headRedirectFixture;
    fake.start();
    defer fake.stopServer();
    const hub: Auth = .{ .endpoint = try fake.endpoint(a), .bearer = "Bearer " ++ literal_token, .source = "HF_TOKEN" };
    var client = @import("pull_http.zig").client(a, std.testing.io);
    defer client.deinit();
    const attack = mixedDepthAttack();
    resetAttack(attack.payload);
    const authority = try std.fmt.allocPrint(a, "127.0.0.1:{d}", .{fake.port});
    head_location = try std.fmt.allocPrint(a, "http://{s}/storage?credential={s}", .{ authority, attack.payload });
    requests_after_head.store(0, .monotonic);
    const result = runs.resolve(a, &client, hub, try std.fmt.allocPrint(a, "{s}/head", .{hub.endpoint}));
    var caught: ?anyerror = null;
    if (result) |_| {} else |err| caught = err;
    try std.testing.expect(caught != null);
    try std.testing.expectEqual(error.HubCredentialRedirect, caught.?);
    try std.testing.expectEqual(@as(u32, 0), requests_after_head.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), attack_requests.load(.monotonic));
    try expectAttackAbsent(attack, "", "");
}

test "security KR66-2: mixed-depth hub credentials are redacted from model_type" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const attack = mixedDepthAttack();
    resetAttack(attack.payload);
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr66-mixed-model-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try envWithToken(a, &fake, literal_token);
    fake.config = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{attack.payload});
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    try expectAttackAbsent(attack, out.written(), err_out.written());
    try std.testing.expectEqual(@as(u32, 0), attack_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), fake.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, err_out.written(), "[redacted]") != null);
}

test "security KR67-2: exact redaction-marker HF_TOKEN cannot survive model_type output" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr67-marker-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try markerModelType(a, io, &fake, root, redaction_marker, redaction_marker);
}

test "security KR67-2: redaction-marker substring HF_TOKEN cannot survive model_type output" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr67-substring-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try markerModelType(a, io, &fake, root, "redacted", redaction_marker);
}

test "security KR67-2: raw percent and nested-encoded redaction markers are contained in model_type output" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr67-encoded-marker-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const once = try encoded(a, redaction_marker);
    const twice = try encoded(a, once);
    for ([_][]const u8{ redaction_marker, once, twice }) |model_type| {
        try markerModelType(a, io, &fake, root, redaction_marker, model_type);
    }
}

test "security KR69-4: a registered family and raw or nested model types contain the HF token" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.start();
    defer fake.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-kr69-family-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const secret = "qwen4_exp";
    const env = try envWithToken(a, &fake, secret);
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    try expectFinalContainment(a, out.written(), err_out.written(), secret);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "Qwen3.8 Flash Next ([redacted])") != null);

    const once = try encoded(a, secret);
    const twice = try encoded(a, once);
    for ([_][]const u8{ once, twice }) |model_type| {
        try markerModelType(a, io, &fake, root, secret, model_type);
    }
}

test "security KR71: exact two-thirds prefixes fail closed at every hub boundary" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake = FakeHub.open() catch return error.SkipZigTest;
    fake.route = reflectedRedirectFixture;
    fake.start();
    defer fake.stopServer();
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    const raw = "abcdef";
    const once = try encoded(a, raw);
    const twice = try encoded(a, once);
    const attacks = [_]Attack{
        .{ .secret = "abcdefghi", .payload = raw, .variants = .{ raw, once, twice, raw } },
        .{ .secret = "abcdefghi", .payload = once, .variants = .{ raw, once, twice, raw } },
        .{ .secret = "abcdefghi", .payload = twice, .variants = .{ raw, once, twice, raw } },
        .{ .secret = "prefix-[redacted]-suffix", .payload = "prefix-prefix-[redacted]-suffix", .variants = .{ "prefix-prefix-[redacted]-suffix", "prefix-[redacted]", "prefix-[redacted]-suffix", "prefix-[redacted]" } },
    };
    for (attacks, 0..) |attack, index| {
        const hub: Auth = .{ .endpoint = try fake.endpoint(a), .bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{attack.secret}), .source = "HF_TOKEN" };
        try redirectAttack(a, &client, hub, attack);
        resetAttack(attack.payload);
        const result = runs.resolve(a, &client, hub, try std.fmt.allocPrint(a, "{s}/head", .{hub.endpoint}));
        try std.testing.expectError(error.HubCredentialRedirect, result);
        try std.testing.expectEqual(@as(u32, 0), attack_requests.load(.monotonic));
        try expectAttackAbsent(attack, request_log[0..request_log_len], "");
        const root = try std.fmt.allocPrint(a, ".tf-pull-kr71-{d}", .{index});
        defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
        fake.route = reflectedTreeFixture;
        try treeAttack(a, io, &fake, root, attack);
        fake.route = reflectedRedirectFixture;
        resetAttack(attack.payload);
        const env = try envWithToken(a, &fake, attack.secret);
        fake.config = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{attack.payload});
        var out: std.Io.Writer.Allocating = .init(a);
        var err_out: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
        try expectAttackAbsent(attack, out.written(), err_out.written());
        try std.testing.expect(std.mem.indexOf(u8, err_out.written(), credential.withheld) != null);
    }
    fake.config = FakeHub.config_body;
    const env = try envWithToken(a, &fake, "qwen4_expX");
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, ".tf-pull-kr71-registered", "Org/Flash"));
    defer std.Io.Dir.cwd().deleteTree(io, ".tf-pull-kr71-registered") catch {};
    for ([_][]const u8{ out.written(), err_out.written() }) |text| try expectAbsent(text, "qwen4_exp");
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "Qwen3.8 Flash Next ([withheld])") != null);
}
