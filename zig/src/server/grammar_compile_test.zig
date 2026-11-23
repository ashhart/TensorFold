//! A compile past its time limit is the request's 400 with the limit named, for every format, before submission.
const std = @import("std");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const fixture = @import("stream_preflight_test.zig");
const t = std.testing;

const Out = struct {
    status: u16 = 0,
    payload: json.Value = .null,
    fn out(o: *@This()) openai.Out {
        return .{ .ctx = o, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {
        return error.Closed;
    }
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {
        return error.Closed;
    }
    fn reply(ctx: *anyopaque, code: u16, payload: json.Value) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.status = code;
        o.payload = payload;
    }
};
fn refuses(_: *anyopaque, _: c_int, _: [*]const u8, _: usize, err: [*]u8, _: usize) callconv(.c) ?*anyopaque {
    const message = "grammar compilation exceeds the 2000 ms time limit";
    @memcpy(err[0..message.len], message);
    err[message.len] = 0;
    return null;
}
fn gone(_: *anyopaque) bool {
    return false;
}

fn run(srv: *Server, a: std.mem.Allocator, field: []const u8, value: json.Value) !Out {
    const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8}")).ok;
    try body.object.put(a, field, value);
    var conn: @import("http_conn.zig").Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .gone_hook = .{ .ctx = srv, .call = gone } };
    var out: Out = .{};
    openai.run(srv, a, out.out(), .{ .conn = &conn }, true, body);
    return out;
}

test "grammar compile server: every format names the deadline in its ordinary 400 reply" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: fixture.TestText = .{};
    var backend: fixture.CallEngine = .{ .tokens = &.{} };
    const srv = try Server.init(t.allocator, t.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var compiler = @import("lanes").grammar_fake.compiler();
    compiler.f.compile = refuses;
    srv.grammars = &compiler;
    srv.info.structures = true;
    for ([_][2][]const u8{
        .{ "response_format", "{\"type\":\"json_object\"}" },
        .{ "guided_json", "{\"type\":\"object\"}" },
        .{ "guided_regex", "\"[a-z]+\"" },
        .{ "guided_choice", "[\"yes\",\"no\"]" },
        .{ "guided_grammar", "\"root ::= \\\"yes\\\"\"" },
    }) |entry| {
        const out = try run(srv, a, entry[0], (try json.parse(a, entry[1])).ok);
        try t.expectEqual(@as(u16, 400), out.status);
        const message = out.payload.get("error").?.strField("message").?;
        try t.expect(std.mem.indexOf(u8, message, "2000 ms time limit") != null);
        try t.expect(std.mem.indexOf(u8, message, entry[0]) != null or std.mem.eql(u8, entry[0], "response_format"));
    }
    var cx: @import("errors.zig").Cx = .{ .a = a };
    try t.expectError(error.Refused, srv.compiledStructure(&cx, .structural_tag, "{}", "tool_choice", false));
    try t.expectEqual(@as(u16, 400), cx.status());
    try t.expect(std.mem.indexOf(u8, cx.message, "2000 ms time limit") != null);
}
