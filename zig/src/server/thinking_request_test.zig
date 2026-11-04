//! Explicit thinking-off requests are either represented in the prompt or refused before submission.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("hf_eos_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Allocator = std.mem.Allocator;

const Script = struct {
    seen: bool = false,
    prompt: [16]u32 = undefined,
    count: usize = 0,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        _ = ctx;
        return .{ .context_window = 262144 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen = true;
        s.count = request.prompt.len;
        @memcpy(s.prompt[0..s.count], request.prompt);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{2} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    whole: ?json.Value = null,
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        _ = ctx;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        _ = ctx;
        _ = value;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
        r.status = status_code;
    }
};

const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

fn capability(srv: *Server, a: Allocator) !bool {
    var wire: Wire = .{ .a = a };
    var c: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    try @import("status_routes.zig").models(srv, &c, a);
    const at = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.MissingHttpHead;
    const body = (try json.parse(a, wire.bytes.items[at + 4 ..])).ok;
    var found: ?bool = null;
    for (body.get("data").?.array) |entry| {
        const caps = entry.get("capabilities") orelse return error.MissingThinkingCapability;
        const applicable = caps.get("thinking_applicable") orelse return error.MissingThinkingApplicability;
        try std.testing.expectEqual(srv.close_id != null, applicable.bool);
        const bit = caps.get("thinking_switch") orelse return error.MissingThinkingCapability;
        try std.testing.expect(bit == .bool);
        if (found) |previous| try std.testing.expectEqual(previous, bit.bool);
        found = bit.bool;
    }
    return found orelse error.MissingModel;
}

const plain = "H";
const dangling = "H{% if add_generation_prompt %}<think>{% endif %}";
const closed = "H{% if add_generation_prompt %}<think></think>{% endif %}";
const switched = "H{% if add_generation_prompt %}{% if enable_thinking %}<think>{% else %}</think>{% endif %}{% endif %}";
const off = ",\"chat_template_kwargs\":{\"enable_thinking\":false}";

const Source = enum { client, ask, ask_in };

fn check(template: []const u8, close_token: bool, server_thinking: bool, extra: []const u8, code: u16, ids: []const u32) !void {
    return checkSource(template, close_token, server_thinking, extra, code, ids, .client);
}

fn checkSource(template: []const u8, close_token: bool, server_thinking: bool, extra: []const u8, code: u16, ids: []const u32, source: Source) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    if (close_token) try f.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0,"O":1,"K":2,"<think>":300,"</think>":301,"<|im_end|>":127906},"merges":[]},"decoder":{"type":"Fuse"},"added_tokens":[{"id":300,"content":"<think>","special":true},{"id":301,"content":"</think>","special":true},{"id":127906,"content":"<|im_end|>","special":true}]}
    });
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = template });
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer text.deinit();
    var script: Script = .{};
    const srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{ "m", "alias" },
        .enable_thinking = server_thinking,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const can_off = try capability(srv, a);
    try t.expect(!script.seen);
    if (code == 400) try t.expect(!can_off) else if (std.mem.eql(u8, extra, off)) try t.expect(can_off);
    if (source != .client) {
        var cx: @import("errors.zig").Cx = .{ .a = a };
        const Stay = struct {
            pub fn check(_: @This()) bool {
                return false;
            }
        };
        const reply = if (source == .ask)
            try @import("slide_ask.zig").ask(srv, &cx, null, "H", 2, Stay{})
        else
            try @import("slide_ask.zig").askIn(srv, &cx, &.{.{ "H", "K" }}, "H", 2, true, Stay{});
        try t.expectEqualStrings("K", reply.content);
        try t.expect(script.seen);
        try t.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
        return;
    }
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"H\"}}],\"max_tokens\":2,\"return_token_ids\":true{s}}}", .{extra}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response: Reply = .{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, body);
    try t.expectEqual(@as(?u16, code), response.status);
    const whole = response.whole.?;
    if (code == 400) {
        try t.expect(!script.seen);
        const message = whole.get("error").?.strField("message").?;
        try t.expect(std.mem.indexOf(u8, message, "thinking") != null);
    } else {
        try t.expect(script.seen);
        try t.expectEqualSlices(u32, ids, script.prompt[0..script.count]);
        const tokens = whole.get("tensorfold").?.get("token_ids").?.array;
        try t.expectEqual(@as(usize, 1), tokens.len);
        try t.expectEqual(@as(?i64, 2), tokens[0].int64());
    }
}

test "thinking request: off without a template switch or closable prefix refuses by name" {
    for ([_][]const u8{ off, ",\"thinking\":false", ",\"thinking\":{\"type\":\"disabled\"}", ",\"reasoning_effort\":\"none\"", ",\"chat_template_kwargs\":{\"thinking\":false}", ",\"chat_template_kwargs\":{\"reasoning_effort\":\"none\"}" }) |extra|
        try check(plain, true, true, extra, 400, &.{});
    try check("H<think></think>H", true, true, off, 400, &.{});
}

test "thinking request: omission, enabled requests and nonreasoning templates keep working" {
    try check(plain, true, true, "", 200, &.{0});
    try check(plain, true, false, "", 200, &.{0});
    try check(plain, true, true, ",\"thinking\":true", 200, &.{0});
    try check(plain, false, true, off, 200, &.{0});
}

test "thinking request: switched, already closed and closable prefixes keep their exact ids" {
    for ([_]bool{ false, true }) |default| {
        try check(switched, true, default, off, 200, &.{ 0, 301 });
        try check(closed, true, default, off, 200, &.{ 0, 300, 301 });
        try check(dangling, true, default, off, 200, &.{ 0, 300, 301 });
    }
}

test "thinking request: internal lesson ask keeps main behavior without an off switch" {
    try checkSource(plain, true, true, "", 200, &.{0}, .ask);
}

test "thinking request: internal lesson askIn keeps main behavior without an off switch" {
    try checkSource(plain, true, true, "", 200, &.{0}, .ask_in);
}
