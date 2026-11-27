//! The execution witness's read route over real sockets: off by default, keyed by scope, and known only when complete.
const std = @import("std");
const json = @import("json");
const auth = @import("auth.zig");
const witness = @import("witness.zig");
const fixture = @import("witness_fixture.zig");
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const t = std.testing;
const Allocator = std.mem.Allocator;
const expectStatus = fixture.expectStatus;

const key_text = "app[inference]: infer-secret\nmonitor[read]: read-secret\nops[admin]: admin-secret\n";
const completion = "{\"model\":\"m\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0}";

fn keyed(file: *KeyFile) !auth.Store {
    var problem: []const u8 = "";
    return auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem);
}

/// A reply with its per-run values (ids, times and rates) zeroed, so two runs of one request compare whole.
pub fn steady(a: Allocator, reply: []const u8) ![]const u8 {
    var v = try fixture.payload(a, reply);
    zero(&v);
    return json.stringify(a, v, .{});
}

fn zero(v: *json.Value) void {
    switch (v.*) {
        .float => v.* = .{ .float = 0 },
        .array => |items| for (items) |*item| zero(item),
        .object => |o| for (o.keys(), o.values()) |k, *item| {
            if (std.mem.eql(u8, k, "id") or std.mem.eql(u8, k, "created")) item.* = .null else zero(item);
        },
        else => {},
    }
}

test "execution witness HTTP: off answers 404 for every key and leaves replies as they were" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys = try keyed(&file);
    defer keys.deinit();
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain_engine: fixture.Engine = .{};
    var plain: fixture.Harness = .{};
    try plain.start(&plain_engine, &keys, null, .{});
    defer plain.deinit();
    for ([_][]const u8{ "admin-secret", "read-secret" }) |key| try expectStatus(try plain.ask(a, "GET", witness.path, key, "localhost", ""), 404);
    var on_engine: fixture.Engine = .{};
    var on: fixture.Harness = .{};
    try on.start(&on_engine, &keys, &c, .{ .execution_witness = true });
    defer on.deinit();
    on.state.?.start(on.srv);
    _ = fixture.settled(on.state.?, on.srv);
    const before = try plain.ask(a, "POST", "/v1/completions", "admin-secret", "localhost", completion);
    const after = try on.ask(a, "POST", "/v1/completions", "admin-secret", "localhost", completion);
    try expectStatus(before, 200);
    try expectStatus(after, 200);
    try t.expectEqualStrings(try steady(a, before), try steady(a, after));
    try t.expect(std.mem.indexOf(u8, after, "witness") == null);
}

test "execution witness HTTP: read and admin keys read it; no key, an inference key or a foreign Host are refused" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys = try keyed(&file);
    defer keys.deinit();
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try h.start(&engine, &keys, &c, .{ .execution_witness = true });
    defer h.deinit();
    h.state.?.start(h.srv);
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "read-secret", "admin-secret" }) |key| try expectStatus(try h.ask(a, "GET", witness.path, key, "localhost", ""), 200);
    try expectStatus(try h.ask(a, "GET", witness.path, "", "localhost", ""), 401);
    try expectStatus(try h.ask(a, "GET", witness.path, "unknown-secret", "localhost", ""), 401);
    const denied = try h.ask(a, "GET", witness.path, "infer-secret", "localhost", "");
    try expectStatus(denied, 403);
    try t.expect(std.mem.indexOf(u8, denied, "insufficient_scope") != null);
    const foreign = try h.ask(a, "GET", witness.path, "read-secret", "example.com", "");
    try expectStatus(foreign, 403);
    try t.expect(std.mem.indexOf(u8, foreign, "Host header is not allowed") != null);
    try expectStatus(try h.ask(a, "POST", witness.path, "admin-secret", "localhost", "{}"), 404);
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire));
}

test "execution witness HTTP: pending until hashed, then known; the same server answers the same witness" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var engine: fixture.Engine = .{};
    var h: fixture.Harness = .{};
    try h.start(&engine, null, &c, .{ .execution_witness = true, .unix_socket = "test" });
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pending = try fixture.payload(a, try h.ask(a, "GET", witness.path, "", "localhost", ""));
    try t.expectEqualStrings("unknown", pending.strField("state").?);
    try t.expectEqualStrings("pending", pending.strField("reason").?);
    h.state.?.start(h.srv);
    try t.expect(fixture.settled(h.state.?, h.srv).result == .known);
    const reply = try h.ask(a, "GET", witness.path, "", "localhost", "");
    try expectStatus(reply, 200);
    const known = try fixture.payload(a, reply);
    try t.expectEqualStrings("known", known.strField("state").?);
    try t.expectEqualStrings("test/fake", known.strField("path").?);
    try t.expectEqual(@as(?i64, 1), known.get("epoch").?.int64());
    for ([_][]const u8{ "witness", "content", "instance" }) |k| try t.expectEqual(@as(usize, if (std.mem.eql(u8, k, "instance")) 32 else 64), known.strField(k).?.len);
    const named = known.get("components").?;
    for (witness.components) |name| try t.expectEqual(@as(usize, 64), named.strField(name).?.len);
    try t.expectEqualStrings(&std.fmt.bytesToHex(@as([32]u8, @splat(7)), .lower), named.strField("engine").?);
    try t.expectEqualStrings(try fixture.body(reply), try fixture.body(try h.ask(a, "GET", witness.path, "", "localhost", "")));
}

test "execution witness HTTP: an engine without a hook is unsupported_path; an excluded flag or a pending engine says so" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { engine: fixture.Engine, args: @import("cli.zig").Args, reason: []const u8, feature: ?[]const u8 = null }{
        .{ .engine = .{ .hooked = false }, .args = .{ .execution_witness = true }, .reason = "unsupported_path" },
        .{ .engine = .{}, .args = .{ .execution_witness = true, .slide = true }, .reason = "feature_excluded", .feature = "--slide" },
        .{ .engine = .{}, .args = .{ .execution_witness = true, .cortheon = true }, .reason = "feature_excluded", .feature = "--cortheon" },
        .{ .engine = .{ .facts = .pending }, .args = .{ .execution_witness = true }, .reason = "pending" },
        .{ .engine = .{ .facts = .{ .unknown = .files_changed } }, .args = .{ .execution_witness = true }, .reason = "files_changed" },
    };
    for (cases) |case| {
        var engine = case.engine;
        var h: fixture.Harness = .{};
        try h.start(&engine, null, &c, case.args);
        defer h.deinit();
        h.state.?.start(h.srv);
        _ = fixture.settled(h.state.?, h.srv);
        const v = try fixture.payload(a, try h.ask(a, "GET", witness.path, "", "localhost", ""));
        try t.expectEqualStrings("unknown", v.strField("state").?);
        try t.expectEqualStrings(case.reason, v.strField("reason").?);
        if (case.feature) |f| try t.expectEqualStrings(f, v.strField("feature").?) else try t.expect(v.get("feature") == null);
        try t.expect(v.get("witness") == null and v.get("content") == null);
    }
}

test "execution witness HTTP: the engine's hashing begins at ready and at each reload, never on an excluded server" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    for ([_]struct { args: @import("cli.zig").Args, begins: [2]u32 }{
        .{ .args = .{ .execution_witness = true, .unix_socket = "test" }, .begins = .{ 1, 2 } },
        .{ .args = .{ .execution_witness = true, .unix_socket = "test", .cortheon = true }, .begins = .{ 0, 0 } },
    }) |case| {
        var engine: fixture.Engine = .{};
        var h: fixture.Harness = .{};
        try h.start(&engine, null, &c, case.args);
        defer h.deinit();
        try t.expectEqual(@as(u32, 0), engine.begins.load(.acquire)); // nothing before ready
        h.state.?.start(h.srv);
        try t.expectEqual(case.begins[0], engine.begins.load(.acquire));
        h.state.?.bump(h.srv.info); // a reload publishes a new engine
        try t.expectEqual(case.begins[1], engine.begins.load(.acquire));
    }
}

test "execution witness HTTP: --confidential is named in the snapshot and the reply binding, and moves only preparation" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var known: [2]json.Value = undefined;
    for ([_]bool{ false, true }, &known) |on, *out| {
        var engine: fixture.Engine = .{};
        var h: fixture.Harness = .{};
        try h.start(&engine, null, &c, .{ .execution_witness = true, .unix_socket = "test", .confidential = on });
        defer h.deinit();
        h.state.?.start(h.srv);
        try t.expect(fixture.settled(h.state.?, h.srv).result == .known);
        out.* = try fixture.payload(a, try h.ask(a, "GET", witness.path, "", "localhost", ""));
        try t.expectEqual(on, out.get("confidential") != null and out.get("confidential").?.bool);
        const asked = "{\"model\":\"m\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0,\"stop\":[\"caller-stop\"],\"execution_witness\":true}";
        const reply = try h.ask(a, "POST", "/v1/completions", "", "localhost", asked);
        const bound = (try fixture.payload(a, reply)).get("execution_witness").?;
        try t.expectEqualStrings("known", bound.strField("state").?);
        try t.expectEqual(on, bound.get("confidential") != null and bound.get("confidential").?.bool);
        // under --confidential the resolved request, with its stop strings and tool text, is bound only by its digest
        try t.expectEqual(!on, bound.get("request") != null);
        try t.expectEqual(!on, std.mem.indexOf(u8, reply, "caller-stop") != null);
        try t.expectEqual(@as(usize, 64), bound.strField("request_sha256").?.len);
    }
    for (witness.components) |name| {
        const same = std.mem.eql(u8, known[0].get("components").?.strField(name).?, known[1].get("components").?.strField(name).?);
        try t.expectEqual(!std.mem.eql(u8, name, "preparation"), same);
    }
    try t.expect(!std.mem.eql(u8, known[0].strField("content").?, known[1].strField("content").?));
}
