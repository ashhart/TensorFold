//! POST /v1/responses/input_tokens by the dispatcher: the prompt POST /v1/responses prefills, its refusals, no run.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const server = @import("server.zig");
const http = @import("http_conn.zig");
const scope = @import("auth_scope.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

/// Bytes as tokens; a render is the messages' and tools' JSON, so every field that shapes a prompt changes the count.
const JsonText = struct {
    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, opts: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const said = try json.stringify(a, messages, .{});
        const tools = if (opts.tools) |v| try json.stringify(a, v, .{}) else "";
        return .{ .text = try std.mem.concat(a, u8, &.{ said, tools }) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Each submitted request's prompt length; every reply is one token.
const Engine = struct {
    window: u32 = 4096,
    mutex: std.Io.Mutex = .init,
    prompts: [16]usize = undefined,
    count: usize = 0,

    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .context_window = cast(ctx).window };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = cast(ctx);
        e.mutex.lockUncancelable(t.io);
        e.prompts[e.count] = req.prompt.len;
        e.count += 1;
        e.mutex.unlock(t.io);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn last(e: *Engine) usize {
        return e.prompts[e.count - 1];
    }
};

fn serve(engine: *Engine, config: server.Config) !*server.Server {
    return server.Server.init(t.allocator, t.io, engine.engine(), JsonText.text(), config, null);
}

const plain: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false };

/// The bytes a route writes, in place of a socket.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Answer = struct { status: u16, body: []const u8 };

/// One POST through the dispatcher, as the connection thread hands it over after the key gate gave it `scopes`.
fn post(srv: *server.Server, a: Allocator, path: []const u8, body: []const u8, scopes: scope.Scopes) !Answer {
    var wire: Wire = .{ .a = a };
    const headers = try a.dupe(http.Header, &.{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }});
    var c: http.Conn = .{ .fd = -1, .peer = "127.0.0.1", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = headers, .key_scopes = scopes, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    routes.dispatch(srv, &c, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    return .{ .status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10), .body = out[at + 4 ..] };
}

fn count(srv: *server.Server, a: Allocator, body: []const u8) !usize {
    const got = try post(srv, a, "/v1/responses/input_tokens", body, scope.full);
    if (got.status != 200) std.debug.print("input_tokens {d}: {s}\n", .{ got.status, got.body });
    try t.expectEqual(@as(u16, 200), got.status);
    const v = (try json.parse(a, got.body)).ok;
    try t.expectEqual(@as(usize, 2), v.object.count());
    try t.expectEqualStrings("response.input_tokens", v.strField("object").?);
    return @intCast(v.get("input_tokens").?.int64().?);
}

test "responses input tokens: the count is the prompt POST /v1/responses prefills for the same body, and nothing runs" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    const srv = try serve(&engine, plain);
    defer srv.deinit();
    const tool = "{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}";
    const bodies = [_][]const u8{
        "{\"model\":\"m\",\"input\":\"Hi\"}",
        "{\"input\":\"Hi\",\"instructions\":\"Be brief\",\"max_output_tokens\":4}",
        "{\"input\":\"Hi\",\"stream\":true,\"store\":false}",
        "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"find\"}]},{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"lookup\",\"arguments\":\"{}\"},{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"done\"}],\"tools\":[" ++ tool ++ "]}",
        "{\"input\":\"Hi\",\"tools\":[" ++ tool ++ "],\"tool_choice\":\"auto\"}",
    };
    var seen: [bodies.len]usize = undefined;
    for (bodies, &seen) |body, *n| {
        const before = engine.count;
        n.* = try count(srv, a, body);
        try t.expectEqual(before, engine.count); // counted, never submitted
        try t.expectEqual(@as(u16, 200), (try post(srv, a, "/v1/responses", body, scope.full)).status);
        try t.expectEqual(before + 1, engine.count);
        try t.expectEqual(n.*, engine.last());
    }
    // instructions, items and tools each change the prompt, so a count that left one out would differ; stream does not
    for ([_]usize{ 1, 3, 4 }) |i| try t.expect(seen[i] != seen[0]);
    try t.expectEqual(seen[0], seen[2]);
    // a stored parent's conversation is part of the prompt the continuation runs
    const parent = try post(srv, a, "/v1/responses", "{\"input\":\"one\"}", scope.full);
    const id = (try json.parse(a, parent.body)).ok.strField("id").?;
    const next = try std.fmt.allocPrint(a, "{{\"input\":\"two\",\"previous_response_id\":\"{s}\"}}", .{id});
    const continued = try count(srv, a, next);
    try t.expect(continued > try count(srv, a, "{\"input\":\"two\"}"));
    try t.expectEqual(@as(u16, 200), (try post(srv, a, "/v1/responses", next, scope.full)).status);
    try t.expectEqual(continued, engine.last());
}

test "responses input tokens: a body POST /v1/responses refuses gets the same status and error, before any run" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    const srv = try serve(&engine, plain);
    defer srv.deinit();
    const refused = [_][]const u8{
        "{",                                                        "[1]",
        "{\"input\":[]}",                                           "{\"input\":\"x\",\"tools\":[{\"type\":\"web_search\"}]}",
        "{\"input\":\"x\",\"background\":true}",                    "{\"input\":\"x\",\"truncation\":\"auto\"}",
        "{\"input\":\"x\",\"previous_response_id\":\"resp_gone\"}", "{\"input\":\"x\",\"logprobs\":true}",
        "{\"input\":\"x\",\"temperature\":\"hot\"}",                "{\"input\":\"x\",\"text\":{\"format\":{\"type\":\"bogus\"}}}",
        "{\"input\":[{\"role\":\"robot\",\"content\":\"x\"}]}",     "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:x\"}]}]}",
    };
    for (refused) |body| {
        const counted = try post(srv, a, "/v1/responses/input_tokens", body, scope.full);
        const run = try post(srv, a, "/v1/responses", body, scope.full);
        try t.expect(counted.status >= 400);
        try t.expectEqual(run.status, counted.status);
        try t.expectEqualStrings(run.body, counted.body);
    }
    // a restricted key is held to the same stateless rule: store false, no stored parent
    const infer: scope.Scopes = .{ .inference = true };
    for ([_][]const u8{ "{\"input\":\"x\"}", "{\"input\":\"x\",\"store\":false,\"previous_response_id\":\"resp_x\"}" }) |body| {
        const counted = try post(srv, a, "/v1/responses/input_tokens", body, infer);
        try t.expectEqual(@as(u16, 403), counted.status);
        try t.expectEqualStrings((try post(srv, a, "/v1/responses", body, infer)).body, counted.body);
    }
    try t.expectEqual(@as(u16, 200), (try post(srv, a, "/v1/responses/input_tokens", "{\"input\":\"x\",\"store\":false}", infer)).status);
    try t.expectEqual(@as(usize, 0), engine.count);
    try t.expect(scope.allows(infer, "POST", "/v1/responses/input_tokens") and scope.allows(infer, "POST", "/responses/input_tokens/"));
    try t.expect(!scope.allows(.{ .read = true }, "POST", "/v1/responses/input_tokens"));
}

test "responses input tokens: a request --compact-at would compact first is refused by name, a shorter one counted" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{ .window = 200 };
    var config = plain;
    config.compact_at = .{ .fraction = 0.5 };
    const srv = try serve(&engine, config);
    defer srv.deinit();
    try t.expect(try count(srv, a, "{\"input\":\"Hi\",\"max_output_tokens\":1}") < 100);
    const long: [120]u8 = @splat('w');
    // past the threshold, then past the window itself: the run compacts both rather than refusing the second
    for ([_]usize{ 1, 3 }) |times| {
        var input: std.ArrayList(u8) = .empty;
        for (0..times) |_| try input.appendSlice(a, &long);
        const body = try std.fmt.allocPrint(a, "{{\"input\":\"{s}\",\"max_output_tokens\":1}}", .{input.items});
        const got = try post(srv, a, "/v1/responses/input_tokens", body, scope.full);
        try t.expectEqual(@as(u16, 400), got.status);
        try t.expect(std.mem.indexOf(u8, got.body, "--compact-at compacts its conversation first") != null);
    }
    try t.expectEqual(@as(usize, 0), engine.count);
}

test "responses input tokens: TENSORFOLD_REQUEST_LOG keeps the bodies POST /v1/responses runs, not the counts" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var engine: Engine = .{};
    var config = plain;
    config.request_log = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const srv = try serve(&engine, config);
    defer srv.deinit();
    _ = try count(srv, a, "{\"input\":\"counted\"}");
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, "requests.jsonl", .{}));
    try t.expectEqual(@as(u16, 200), (try post(srv, a, "/v1/responses", "{\"input\":\"run\"}", scope.full)).status);
    const logged = try tmp.dir.readFileAlloc(t.io, "requests.jsonl", a, .limited(1 << 16));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, logged, "\n"));
    try t.expect(std.mem.indexOf(u8, logged, "counted") == null);
}

test "responses input tokens: a turn --cortheon would run is refused by name, one it stays out of counted" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    const srv = try serve(&engine, plain);
    defer srv.deinit();
    var web: @import("cortheon/cortheon_test.zig").Web = .{};
    srv.cortheon = try @import("cortheon/cortheon.zig").State.initWith(t.allocator, t.io, web.web());
    const got = try post(srv, a, "/v1/responses/input_tokens", "{\"input\":\"Hi\"}", scope.full);
    try t.expectEqual(@as(u16, 400), got.status);
    try t.expect(std.mem.indexOf(u8, got.body, "--cortheon runs it as a Cortheon turn") != null);
    // tool_choice none keeps Cortheon out of the turn, so the run's prompt is the plain render this counts
    const tool = "{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}";
    _ = try count(srv, a, "{\"input\":\"Hi\",\"tools\":[" ++ tool ++ "],\"tool_choice\":\"none\"}");
    try t.expectEqual(@as(usize, 0), engine.count);
    try t.expectEqual(@as(usize, 0), web.searches);
}
