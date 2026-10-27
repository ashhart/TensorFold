//! Key-file scopes: label syntax, exact route grants, stateless Responses, rotation and conflicting duplicates.
const std = @import("std");
const json = @import("json");
const auth = @import("auth.zig");
const scope = @import("auth_scope.zig");
const platform = @import("sys");
const t = std.testing;

pub const KeyFile = struct {
    tmp: t.TmpDir,
    path: [:0]u8,

    pub fn init(content: []const u8) !KeyFile {
        if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // refusal by mode bits; NTFS keeps ACLs
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const path = try std.fmt.allocPrintSentinel(t.allocator, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
        errdefer t.allocator.free(path);
        var file: KeyFile = .{ .tmp = tmp, .path = path };
        try file.write(content);
        try t.expectEqual(@as(c_int, 0), platform.c.fchmodat(platform.posix.AT.FDCWD, path, 0o600, 0));
        return file;
    }

    pub fn write(f: *KeyFile, content: []const u8) !void {
        try f.tmp.dir.writeFile(t.io, .{ .sub_path = "keys", .data = content });
    }

    pub fn deinit(f: *KeyFile) void {
        f.tmp.cleanup();
        t.allocator.free(f.path);
    }
};

test "auth scopes: plain labels keep every route and bracketed sets are strict" {
    const cases = [_]struct { text: []const u8, label: []const u8, expected: scope.Scopes }{
        .{ .text = "ops", .label = "ops", .expected = scope.full },
        .{ .text = "a_b.c-9[admin]", .label = "a_b.c-9", .expected = scope.full },
        .{ .text = "app[inference]", .label = "app", .expected = .{ .inference = true } },
        .{ .text = "monitor[read]", .label = "monitor", .expected = .{ .read = true } },
        .{ .text = "app[inference, read]", .label = "app", .expected = .{ .inference = true, .read = true } },
        .{ .text = "app[read,inference]", .label = "app", .expected = .{ .inference = true, .read = true } },
    };
    for (cases) |case| {
        const parsed = try scope.parseLabel(case.text);
        try t.expectEqualStrings(case.label, parsed.name);
        try t.expect(case.expected.eql(parsed.scopes));
    }
    const bad_scopes = [_][]const u8{
        "x[]",      "x[read,]", "x[,read]",       "x[read,read]", "x[Read]",       "x[all]",
        "x[infer]", "x[read",   "x[read][admin]", "x[[read]]",    "x[admin,read]", "x[read]extra",
    };
    for (bad_scopes) |bad| try t.expectError(error.InvalidScopes, scope.parseLabel(bad));
    const oversized: [65]u8 = @splat('a');
    for ([_][]const u8{ "", "[read]", "bad label[read]", "bad]", &oversized }) |bad|
        try t.expectError(error.InvalidLabel, scope.parseLabel(bad));
}

const inference_routes = [_][]const u8{
    "/v1/completions", "/completions", "/v1/chat/completions", "/chat/completions",      "/v1/responses",
    "/responses",      "/v1/messages", "/messages",            "/messages/count_tokens", "/v1/messages/count_tokens",
    "/v1/tokenize",    "/tokenize",    "/v1/detokenize",       "/detokenize",            "/v1/audio/speech",
    "/v1/embeddings",  "/decisions",   "/v1/decisions",        "/systemone",             "/v1/systemone",
};
const read_routes = [_][]const u8{ "/metrics", "/v1/metrics", "/memory", "/v1/memory", "/stats", "/dashboard", "/v1/benchmark/manifest" };
const private_routes = [_][]const u8{
    "/v1/slide/learn",   "/v1/slide/graph",     "/v1/slide/fact/1",  "/v1/recall",           "/v1/imprint",
    "/v1/imprint/load",  "/v1/imprint/resume",  "/v1/responses/id",  "/responses/id",        "/x/models",
    "/x/completions",    "/x/chat/completions", "/x/systemone",      "/v1/completions/more", "/v1/../imprint/load",
    "/v1/%63ompletions", "/V1/COMPLETIONS",     "/v1/completions#x", "http://h/v1/models",   "/future/private",
    "/slide/unknown.js", "/v1/completions/..",  "/./v1/completions",
};

test "auth scopes: route grants are exact and every other method or resource is refused" {
    const infer: scope.Scopes = .{ .inference = true };
    const read: scope.Scopes = .{ .read = true };
    const both: scope.Scopes = .{ .inference = true, .read = true };
    for (inference_routes) |path| {
        try t.expect(scope.allows(infer, "POST", path) and scope.allows(both, "POST", path));
        try t.expect(!scope.allows(read, "POST", path) and !scope.allows(infer, "GET", path));
        for ([_][]const u8{ "/", "///", "?x=1", "/?stream=true" }) |tail| {
            const variant = try std.mem.concat(t.allocator, u8, &.{ path, tail });
            defer t.allocator.free(variant);
            try t.expect(scope.allows(infer, "POST", variant)); // the dispatcher's own route: query and slashes cut
        }
    }
    for (read_routes) |path| {
        try t.expect(scope.allows(read, "GET", path) and scope.allows(both, "GET", path));
        try t.expect(!scope.allows(infer, "GET", path) and !scope.allows(read, "POST", path));
    }
    for ([_][]const u8{ "/models", "/v1/models", "/v1/voices" }) |path| {
        try t.expect(scope.allows(read, "GET", path) and scope.allows(infer, "GET", path));
        try t.expect(!scope.allows(both, "POST", path));
    }
    for (private_routes) |path| {
        for ([_][]const u8{ "GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS" }) |method| {
            try t.expect(!scope.allows(both, method, path));
            try t.expect(scope.allows(scope.full, method, path));
        }
    }
    for ([_][]const u8{ "DELETE", "PUT", "PATCH", "HEAD", "post", "OPTIONS", "CONNECT", "" }) |method| {
        try t.expect(!scope.allows(both, method, "/v1/completions"));
        try t.expect(!scope.allows(both, method, "/v1/models"));
    }
    try t.expect(!scope.allows(.{}, "GET", "/v1/models"));
}

fn denial(scopes: scope.Scopes, text: []const u8) !?scope.Denial {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    return scope.responseDenial(scopes, (try json.parse(arena.allocator(), text)).ok);
}

test "auth scopes: restricted Responses requests must be stateless and full keys keep storage" {
    const infer: scope.Scopes = .{ .inference = true };
    try t.expectEqual(@as(?scope.Denial, null), try denial(infer, "{\"input\":\"x\",\"store\":false}"));
    try t.expectEqual(@as(?scope.Denial, null), try denial(infer, "{\"input\":\"x\",\"store\":false,\"previous_response_id\":null}"));
    try t.expectEqual(@as(?scope.Denial, null), try denial(infer, "[1]"));
    for ([_][]const u8{ "{\"input\":\"x\"}", "{\"store\":true}", "{\"store\":null}", "{\"store\":0}", "{\"store\":\"false\"}" }) |body|
        try t.expectEqual(@as(?scope.Denial, .response_storage), try denial(infer, body));
    for ([_][]const u8{ "{\"store\":false,\"previous_response_id\":\"resp_1\"}", "{\"previous_response_id\":1}" }) |body|
        try t.expectEqual(@as(?scope.Denial, .response_history), try denial(.{ .inference = true, .read = true }, body));
    for ([_][]const u8{ "{\"input\":\"x\"}", "{\"previous_response_id\":\"resp_1\"}" }) |body|
        try t.expectEqual(@as(?scope.Denial, null), try denial(scope.full, body));
}

test "auth scopes: flags, environment and plain file lines keep every route; the first configured key wins" {
    var file = try KeyFile.init("ops: file-secret\nplain-secret\nworker[inference]: app-secret\n");
    defer file.deinit();
    var problem: []const u8 = "";
    var store = try auth.Store.init(t.allocator, &.{"cli-secret"}, "env-secret", file.path, false, &problem);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "cli-secret", "env-secret", "file-secret", "plain-secret" }) |key| {
        try t.expect(store.matchIdentity(a, null, key).?.scopes.eql(scope.full));
        try t.expect(store.matchIdentity(a, try std.fmt.allocPrint(a, "Bearer {s}", .{key}), null).?.scopes.eql(scope.full));
    }
    try t.expectEqualStrings("file-2", store.matchIdentity(a, null, "plain-secret").?.label);
    const worker = store.matchIdentity(a, "Bearer app-secret", null).?;
    try t.expectEqualStrings("worker", worker.label);
    try t.expect(worker.scopes.eql(.{ .inference = true }));
    try t.expectEqualStrings("cli-1", store.matchIdentity(a, "Bearer app-secret", "cli-secret").?.label);
    try t.expect(store.matchIdentity(a, "Bearer app-secret", "cli-secret").?.scopes.admin);
    try t.expectEqualStrings("worker", store.matchIdentity(a, "Bearer app-secret", "unknown").?.label);
    try t.expect(store.matchIdentity(a, "Bearer unknown", "also-unknown") == null);
    try t.expectEqualStrings("ops", store.match(a, null, "file-secret").?);
}

test "auth scopes: a key's scopes follow the file across reloads and an invalid file closes every key" {
    var file = try KeyFile.init("worker[inference]: app-secret\nmonitor[read]: read-secret\n");
    defer file.deinit();
    var problem: []const u8 = "";
    var store = try auth.Store.init(t.allocator, &.{"cli-secret"}, "", file.path, false, &problem);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const first = store.matchIdentity(a, "Bearer app-secret", null).?;
    try t.expect(first.scopes.eql(.{ .inference = true }));
    try t.expect(store.matchIdentity(a, null, "read-secret").?.scopes.eql(.{ .read = true }));
    try file.write("worker[read]: app-secret\n");
    auth.reload_requested.store(true, .release);
    try t.expect(store.matchIdentity(a, null, "app-secret").?.scopes.eql(.{ .read = true }));
    try t.expect(first.scopes.eql(.{ .inference = true })); // a request keeps what it matched
    try t.expectEqualStrings("worker", first.label);
    try file.write("worker[all]: app-secret\n");
    auth.reload_requested.store(true, .release);
    try t.expect(store.matchIdentity(a, null, "app-secret") == null);
    try t.expect(store.matchIdentity(a, null, "cli-secret") == null);
    try file.write("worker[inference,read]: new-secret\n");
    auth.reload_requested.store(true, .release);
    try t.expect(store.matchIdentity(a, null, "new-secret").?.scopes.eql(.{ .inference = true, .read = true }));
    try t.expect(store.matchIdentity(a, null, "app-secret") == null);
    try t.expect(store.matchIdentity(a, null, "cli-secret").?.scopes.admin);
}

test "auth scopes: one key cannot carry two scope sets, so no full copy shadows a restricted line" {
    var file = try KeyFile.init("one[inference]: same-secret\ntwo[read]: same-secret\n");
    defer file.deinit();
    var problem: []const u8 = "";
    try t.expectError(error.KeyFile, auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem));
    try t.expectEqualStrings("the same API key cannot have different scopes", problem);
    try file.write("one[inference]: same-secret\nsame-secret\n");
    try t.expectError(error.KeyFile, auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem));
    try file.write("one[inference]: same-secret\n");
    try t.expectError(error.KeyFile, auth.Store.init(t.allocator, &.{"same-secret"}, "", file.path, false, &problem));
    try t.expectError(error.KeyFile, auth.Store.init(t.allocator, &.{}, "same-secret", file.path, false, &problem));
    try file.write("one[inference]: same-secret\ntwo[inference]: same-secret\nops: other\nops-again: other\n");
    var store = try auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const matched = store.matchIdentity(arena.allocator(), null, "same-secret").?;
    try t.expectEqualStrings("one", matched.label);
    try t.expect(matched.scopes.eql(.{ .inference = true }));
    try t.expectEqualStrings("ops", store.match(arena.allocator(), null, "other").?);
    for ([_][]const u8{ "x[read: key\n", "x[]: key\n", "bad label[read]: key\n" }) |text| {
        try file.write(text);
        try t.expectError(error.KeyFile, auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem));
    }
}
