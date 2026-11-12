//! The reply binding over real sockets: asked for only with the flag, known only when admission and end agree.
const std = @import("std");
const json = @import("json");
const witness = @import("witness.zig");
const fixture = @import("witness_fixture.zig");
const tokens = @import("tokens.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const t = std.testing;
const Allocator = std.mem.Allocator;
const expectStatus = fixture.expectStatus;
const steady = @import("witness_http_test.zig").steady;

const open_body = "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":1,\"temperature\":0,\"execution_witness\":true";
const ids_prompt = open_body ++ "}";
const chat_body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"execution_witness\":true}";

/// A server with the flag on, hashed and ready (an API key is not needed with --unix-socket's flag set).
fn ready(h: *fixture.Harness, engine: *fixture.Engine, c: *fixture.Checkpoint, args: @import("cli.zig").Args) !void {
    var with = args;
    with.execution_witness = true;
    with.unix_socket = "test";
    try h.start(engine, null, c, with);
    h.state.?.start();
    _ = fixture.settled(h.state.?, h.srv);
}

fn post(h: *fixture.Harness, a: Allocator, path: []const u8, data: []const u8) !json.Value {
    const reply = try h.ask(a, "POST", path, "", "localhost", data);
    try expectStatus(reply, 200);
    return fixture.payload(a, reply);
}

/// The test's own walk of the documented encoding, from the reply's JSON as a client parses it.
fn requestSha(v: json.Value) [64]u8 {
    var h = Sha256.init(.{});
    lengthed(&h, "tensorfold resolved request v1");
    walk(&h, v);
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

fn walk(h: *Sha256, v: json.Value) void {
    switch (v) {
        .null => lengthed(h, "n"),
        .bool => |b| pair(h, "b", if (b) "1" else "0"),
        .int => |text| pair(h, "i", text),
        .float => |f| pair(h, "f", &std.mem.toBytes(std.mem.nativeToLittle(u64, @bitCast(f)))),
        .string => |s| pair(h, "s", s),
        .array => |items| {
            pair(h, "a", &std.mem.toBytes(std.mem.nativeToLittle(u64, items.len)));
            for (items) |item| walk(h, item);
        },
        .object => |o| {
            pair(h, "o", &std.mem.toBytes(std.mem.nativeToLittle(u64, o.count())));
            for (o.keys(), o.values()) |k, item| {
                lengthed(h, k);
                walk(h, item);
            }
        },
    }
}

fn pair(h: *Sha256, tag: []const u8, bytes: []const u8) void {
    lengthed(h, tag);
    lengthed(h, bytes);
}

fn lengthed(h: *Sha256, bytes: []const u8) void {
    h.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, bytes.len)));
    h.update(bytes);
}

/// Every digest a binding names recomputes from the fields beside it.
fn expectConsistent(b: json.Value) !void {
    try t.expectEqualStrings(&requestSha(b.get("request").?), b.strField("request_sha256").?);
    if (b.strField("binding")) |binding| {
        var h = Sha256.init(.{});
        lengthed(&h, "tensorfold execution binding v1");
        for ([_][]const u8{ "witness", "prompt_sha256", "request_sha256", "token_sha256", "finish_reason" }) |k| lengthed(&h, b.strField(k).?);
        try t.expectEqualStrings(&std.fmt.bytesToHex(h.finalResult(), .lower), binding);
    }
}

test "execution witness reply: off, a true field is refused; on, an absent or false field leaves the reply as it was" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain_engine: fixture.Engine = .{};
    var plain: fixture.Harness = .{};
    try plain.start(&plain_engine, null, null, .{});
    defer plain.deinit();
    const refused = try plain.ask(a, "POST", "/v1/completions", "", "localhost", ids_prompt);
    try expectStatus(refused, 400);
    try t.expect(std.mem.indexOf(u8, refused, "execution_witness needs a server started with --execution-witness") != null);
    const typed = try plain.ask(a, "POST", "/v1/completions", "", "localhost", "{\"prompt\":\"hi\",\"max_tokens\":1,\"execution_witness\":1}");
    try expectStatus(typed, 400);
    try t.expect(std.mem.indexOf(u8, typed, "execution_witness must be a boolean") != null);
    try t.expectEqual(@as(u32, 0), plain_engine.calls.load(.acquire));
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    const absent = "{\"model\":\"m\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0}";
    const off = "{\"model\":\"m\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0,\"execution_witness\":false}";
    const replies = [_][]const u8{
        try plain.ask(a, "POST", "/v1/completions", "", "localhost", absent),
        try h.ask(a, "POST", "/v1/completions", "", "localhost", absent),
        try h.ask(a, "POST", "/v1/completions", "", "localhost", off),
    };
    for (replies) |r| {
        try expectStatus(r, 200);
        try t.expect(std.mem.indexOf(u8, r, "execution_witness") == null);
    }
    try t.expectEqualStrings(try steady(a, replies[0]), try steady(a, replies[1]));
    try t.expectEqualStrings(try steady(a, replies[0]), try steady(a, replies[2]));
}

test "execution witness reply: known binds the read route's witness, the prompt ids, the resolved request and the tokens" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const read = try fixture.payload(a, try h.ask(a, "GET", witness.path, "", "localhost", ""));
    const reply = try post(&h, a, "/v1/completions", ids_prompt);
    const b = reply.get("execution_witness").?;
    try t.expectEqualStrings("known", b.strField("state").?);
    try t.expect(b.get("reason") == null and b.get("admitted_epoch") == null);
    for ([_][]const u8{ "witness", "instance", "path" }) |k| try t.expectEqualStrings(read.strField(k).?, b.strField(k).?);
    try t.expectEqual(@as(?i64, 1), b.get("epoch").?.int64());
    try t.expectEqualStrings(&tokens.tokenSha256(&.{ 104, 105 }), b.strField("prompt_sha256").?);
    try t.expectEqualStrings(&tokens.tokenSha256(&.{'x'}), b.strField("token_sha256").?);
    try t.expectEqualStrings(reply.get("tensorfold").?.strField("token_sha256").?, b.strField("token_sha256").?);
    try t.expectEqualStrings(reply.get("choices").?.array[0].strField("finish_reason").?, b.strField("finish_reason").?);
    const request = b.get("request").?;
    try t.expectEqual(@as(?i64, 1), request.get("max_tokens").?.int64());
    try t.expect(request.get("sampling").? == .null);
    try expectConsistent(b);
    const again = (try post(&h, a, "/v1/completions", ids_prompt)).get("execution_witness").?;
    try t.expectEqualStrings(b.strField("binding").?, again.strField("binding").?);
    const longer = (try post(&h, a, "/v1/completions", "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":2,\"temperature\":0,\"execution_witness\":true}")).get("execution_witness").?;
    try t.expect(!std.mem.eql(u8, b.strField("request_sha256").?, longer.strField("request_sha256").?));
    try t.expect(!std.mem.eql(u8, b.strField("binding").?, longer.strField("binding").?));
    try t.expectEqualStrings(b.strField("prompt_sha256").?, longer.strField("prompt_sha256").?);
}

test "execution witness reply: an unsupported path or an excluded flag binds unknown with the digests" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { engine: fixture.Engine, args: @import("cli.zig").Args, reason: []const u8, feature: ?[]const u8 = null }{
        .{ .engine = .{ .hooked = false }, .args = .{}, .reason = "unsupported_path" },
        .{ .engine = .{}, .args = .{ .slide = true }, .reason = "feature_excluded", .feature = "--slide" },
        .{ .engine = .{ .facts = .{ .unknown = .files_changed } }, .args = .{}, .reason = "files_changed" },
    };
    for (cases) |case| {
        var engine = case.engine;
        var h: fixture.Harness = .{};
        try ready(&h, &engine, &c, case.args);
        defer h.deinit();
        const b = (try post(&h, a, "/v1/completions", ids_prompt)).get("execution_witness").?;
        try t.expectEqualStrings("unknown", b.strField("state").?);
        try t.expectEqualStrings(case.reason, b.strField("reason").?);
        if (case.feature) |f| try t.expectEqualStrings(f, b.strField("feature").?) else try t.expect(b.get("feature") == null);
        try t.expect(b.get("binding") == null and b.get("witness") == null);
        try t.expectEqual(@as(?i64, 1), b.get("admitted_epoch").?.int64());
        try t.expectEqualStrings(&tokens.tokenSha256(&.{'x'}), b.strField("token_sha256").?);
        try expectConsistent(b);
    }
}

test "execution witness reply: an epoch that moves between admission and the end binds unknown" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    engine.bump = h.state.?;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const b = (try post(&h, a, "/v1/completions", ids_prompt)).get("execution_witness").?;
    try t.expectEqualStrings("unknown", b.strField("state").?);
    try t.expectEqualStrings("epoch_changed", b.strField("reason").?);
    try t.expectEqual(@as(?i64, 1), b.get("admitted_epoch").?.int64());
    try t.expectEqual(@as(?i64, 2), b.get("epoch").?.int64());
    try t.expect(b.get("binding") == null);
}

test "execution witness reply: a stream binds as the whole reply does, on its finish chunk only" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = (try post(&h, a, "/v1/completions", open_body ++ "}")).get("execution_witness").?;
    try t.expectEqualStrings("known", whole.strField("state").?);
    try expectConsistent(whole);
    const streamed = try h.ask(a, "POST", "/v1/completions", "", "localhost", open_body ++ ",\"stream\":true,\"stream_options\":{\"include_usage\":true}}");
    try expectStatus(streamed, 200);
    var found: usize = 0;
    var frames = std.mem.splitSequence(u8, try fixture.body(streamed), "\n\n");
    while (frames.next()) |frame| {
        if (!std.mem.startsWith(u8, frame, "data: {")) continue;
        const v = switch (try json.parse(a, frame["data: ".len..])) {
            .ok => |x| x,
            .err => return error.InvalidJson,
        };
        const b = v.get("execution_witness") orelse continue;
        found += 1;
        try t.expect(v.get("choices").?.array[0].get("finish_reason").? != .null);
        try t.expectEqualStrings(whole.strField("binding").?, b.strField("binding").?);
    }
    try t.expectEqual(@as(usize, 1), found);
}

test "execution witness reply: a sampled reply names the seed it drew, or the one it was given" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const drawn = (try post(&h, a, "/v1/completions", "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":1,\"temperature\":0.7,\"execution_witness\":true}")).get("execution_witness").?;
    const sampling = drawn.get("request").?.get("sampling").?;
    try t.expect(sampling.get("seed").? == .int);
    try t.expectEqual(@as(f64, 0.7), sampling.get("temperature").?.float64().?);
    try expectConsistent(drawn);
    const given = (try post(&h, a, "/v1/completions", "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":1,\"temperature\":0.7,\"seed\":5,\"execution_witness\":true}")).get("execution_witness").?;
    try t.expectEqual(@as(?i64, 5), given.get("request").?.get("sampling").?.get("seed").?.int64());
    try expectConsistent(given);
    const same = (try post(&h, a, "/v1/completions", "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":1,\"temperature\":0.7,\"seed\":5,\"execution_witness\":true}")).get("execution_witness").?;
    try t.expectEqualStrings(given.strField("binding").?, same.strField("binding").?);
}

const Concurrent = struct {
    h: *fixture.Harness,
    witness_hex: []const u8,
    failed: std.atomic.Value(u32) = .init(0),

    fn run(c: *Concurrent, n: usize) void {
        c.one(n) catch {
            _ = c.failed.fetchAdd(1, .acq_rel);
        };
    }

    fn one(c: *Concurrent, n: usize) !void {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const data = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[104,{d}],\"max_tokens\":1,\"temperature\":0,\"execution_witness\":true}}", .{100 + n});
        const b = (try post(c.h, a, "/v1/completions", data)).get("execution_witness").?;
        if (!std.mem.eql(u8, "known", b.strField("state").?)) return error.Unknown;
        if (!std.mem.eql(u8, c.witness_hex, b.strField("witness").?)) return error.OtherWitness;
        const ids = [_]u32{ 104, @intCast(100 + n) };
        if (!std.mem.eql(u8, &tokens.tokenSha256(&ids), b.strField("prompt_sha256").?)) return error.OtherPrompt;
        try expectConsistent(b);
    }
};

test "execution witness reply: concurrent replies each bind their own prompt to the one witness" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const read = try fixture.payload(a, try h.ask(a, "GET", witness.path, "", "localhost", ""));
    var shared: Concurrent = .{ .h = &h, .witness_hex = read.strField("witness").? };
    var threads: [8]std.Thread = undefined;
    for (&threads, 0..) |*th, n| th.* = try std.Thread.spawn(.{}, Concurrent.run, .{ &shared, n });
    for (threads) |th| th.join();
    try t.expectEqual(@as(u32, 0), shared.failed.load(.acquire));
    try t.expectEqual(@as(u32, 8), engine.calls.load(.acquire));
}

test "execution witness reply: a key without read is refused before the engine; read beside inference, admin or full binds" {
    var file = try @import("auth_scope_test.zig").KeyFile.init("app[inference]: infer-secret\nboth[inference,read]: both-secret\nops[admin]: admin-secret\nfull-secret\n");
    defer file.deinit();
    var problem: []const u8 = "";
    var keys = try @import("auth.zig").Store.init(t.allocator, &.{}, "", file.path, false, &problem);
    defer keys.deinit();
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, &keys, &c, .{ .execution_witness = true });
    defer h.deinit();
    h.state.?.start();
    _ = fixture.settled(h.state.?, h.srv);
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const routes = [_][2][]const u8{ .{ "/v1/completions", ids_prompt }, .{ "/v1/chat/completions", chat_body } };
    for (routes) |route| {
        const denied = try h.ask(a, "POST", route[0], "infer-secret", "localhost", route[1]);
        try expectStatus(denied, 403);
        try t.expectEqualStrings("insufficient_scope", (try fixture.payload(a, denied)).get("error").?.strField("code").?);
        try t.expectEqualStrings("execution_witness", (try fixture.payload(a, denied)).get("error").?.strField("param").?);
    }
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire));
    const off = "{\"model\":\"m\",\"prompt\":[104,105],\"max_tokens\":1,\"temperature\":0,\"execution_witness\":false}";
    const plain = try h.ask(a, "POST", "/v1/completions", "infer-secret", "localhost", off);
    try expectStatus(plain, 200);
    try t.expect(std.mem.indexOf(u8, plain, "execution_witness") == null);
    for ([_][]const u8{ "both-secret", "admin-secret", "full-secret" }) |key| for (routes) |route| {
        const reply = try h.ask(a, "POST", route[0], key, "localhost", route[1]);
        try expectStatus(reply, 200);
        const b = (try fixture.payload(a, reply)).get("execution_witness").?;
        try t.expectEqualStrings("known", b.strField("state").?);
        try expectConsistent(b);
    };
    try t.expectEqual(@as(u32, 7), engine.calls.load(.acquire));
}

test "execution witness reply: Responses and Messages refuse a true field by name, before the engine" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try ready(&h, &engine, &c, .{});
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const responses = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":16,\"execution_witness\":true}";
    const routes = [_][2][]const u8{ .{ "/v1/responses", responses }, .{ "/v1/messages", chat_body }, .{ "/v1/messages/count_tokens", chat_body } };
    for (routes) |route| {
        const refused = try h.ask(a, "POST", route[0], "", "localhost", route[1]);
        try expectStatus(refused, 400);
        try t.expect(std.mem.indexOf(u8, refused, "execution_witness is supported on /v1/chat/completions and /v1/completions only") != null);
    }
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire));
    for (routes) |route| {
        const without = try std.mem.replaceOwned(u8, a, route[1], ",\"execution_witness\":true", "");
        try expectStatus(try h.ask(a, "POST", route[0], "", "localhost", without), 200);
    }
    try t.expectEqual(@as(u32, 2), engine.calls.load(.acquire));
}
