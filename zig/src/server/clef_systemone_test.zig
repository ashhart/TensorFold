//! /v1/systemone on a readout engine: 422 locations, each 400, the answers and `model` echoed, text routes refused.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const readout = @import("lanes").readout;
const clef = @import("clef_systemone.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;

/// A readout engine whose option j's logit is j (BF16), or NaN, or a run that ends without its answer.
const Provider = struct {
    mode: enum { normal, nan, fail, refuse } = .normal,
    refusal: api.SubmitError = error.Busy,
    window: u32 = 8192,
    calls: usize = 0,
    seen_options: usize = 0,
    seen_untaught: bool = false,

    fn engine(p: *Provider) api.Engine {
        return .{ .ctx = p, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Provider {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        return .{ .text_generation = false, .context_window = cast(ptr).window, .readouts = .{ .max_rows = 16384, .max_questions = 64, .max_options = 1024 } };
    }
    fn submit(ptr: *anyopaque, id: api.Id, ask: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const p = cast(ptr);
        p.calls += 1;
        const spec = ask.readout orelse return error.InvalidRequest;
        if (p.mode == .refuse) return p.refusal;
        p.seen_options = spec.option_spans.len;
        p.seen_untaught = ask.untaught;
        for (spec.logits, 0..) |*l, j| l.* = if (p.mode == .nan) 0x7fc0 else @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(j)))) >> 16);
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (p.mode == .fail) .stop else .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Present = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

fn server(p: *Provider) !*Server {
    return Server.init(t.allocator, t.io, p.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

/// One token a piece, so a record of a thousand options fits the window.
const OnePiece = struct {
    fn model() @import("model_text.zig").Text {
        var model_text = ByteText.model();
        model_text.vtable = &vtable;
        return model_text;
    }
    const vtable = blk: {
        var v = ByteText.model().vtable.*;
        v.encode = encode;
        break :blk v;
    };
    fn encode(_: *anyopaque, a: std.mem.Allocator, _: []const u8, _: bool) @import("model_text.zig").Error![]u32 {
        return a.dupe(u32, &.{1});
    }
};

fn post(a: std.mem.Allocator, srv: *Server, body: []const u8) !clef.Reply {
    return (try clef.reply(srv, a, (try json.parse(a, body)).ok, Present{})).?;
}

fn text(a: std.mem.Allocator, v: json.Value) ![]const u8 {
    return json.stringify(a, v, .{ .ascii = false, .compact = true });
}

test "Clef System One: a request missing or mistyping a field is a 422 naming its location" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Provider = .{};
    const srv = try server(&p);
    defer srv.deinit();
    const cases = [_][2][]const u8{
        .{ "[]", "{\"type\":\"dict_type\",\"loc\":[\"body\"]" },
        .{ "{}", "\"loc\":[\"body\",\"model\"],\"msg\":\"Field required\"},{\"type\":\"missing\",\"loc\":[\"body\",\"state\"]" },
        .{ "{\"model\":1,\"state\":null,\"questions\":{}}", "{\"type\":\"string_type\",\"loc\":[\"body\",\"model\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":[]}", "{\"type\":\"dict_type\",\"loc\":[\"body\",\"questions\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":1}}", "\"loc\":[\"body\",\"questions\",\"q\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{}}}", "{\"type\":\"missing\",\"loc\":[\"body\",\"questions\",\"q\",\"type\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":3}}}", "{\"type\":\"string_type\",\"loc\":[\"body\",\"questions\",\"q\",\"type\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":[1]}}}", "{\"type\":\"dict_type\",\"loc\":[\"body\",\"questions\",\"q\",\"criteria\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":{\"a\":1}}}}", "{\"type\":\"list_type\",\"loc\":[\"body\",\"questions\",\"q\",\"criteria\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"noul\",\"criteria\":[1]}}}", "{\"type\":\"dict_type\",\"loc\":[\"body\",\"questions\",\"q\",\"criteria\"]" },
        .{ "{\"model\":\"x\",\"state\":1,\"raw\":\"yes\",\"questions\":{\"q\":{\"type\":\"noul\"}}}", "{\"type\":\"bool_type\",\"loc\":[\"body\",\"raw\"]" },
    };
    for (cases) |c| {
        const r = try post(a, srv, c[0]);
        const got = try text(a, r.body);
        if (r.status != 422 or std.mem.indexOf(u8, got, c[1]) == null) {
            std.debug.print("{s}: {d} {s}\n", .{ c[0], r.status, got });
            return error.TestUnexpectedResult;
        }
    }
    try t.expectEqual(@as(usize, 0), p.calls);
}

test "Clef System One: unknown types, empty criteria, the bounds, media and a too-long schema are 400s" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Provider = .{ .window = 300 };
    const srv = try server(&p);
    defer srv.deinit();
    var many: std.ArrayList(u8) = .empty;
    try many.appendSlice(a, "{\"model\":\"x\",\"state\":1,\"questions\":{");
    for (0..65) |i| try many.print(a, "{s}\"q{d}\":{{\"type\":\"noul\"}}", .{ if (i == 0) "" else ",", i });
    try many.appendSlice(a, "}}");
    var wide: std.ArrayList(u8) = .empty;
    try wide.appendSlice(a, "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{");
    for (0..257) |i| try wide.print(a, "{s}\"o{d}\":null", .{ if (i == 0) "" else ",", i });
    try wide.appendSlice(a, "}}}}");
    var long: std.ArrayList(u8) = .empty;
    try long.appendSlice(a, "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"");
    try long.appendNTimes(a, 'w', 300);
    try long.appendSlice(a, "\"}}}");
    const cases = [_][2][]const u8{
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"maybe\"}}}", "q: type must be noul, choice, or score" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{}}}}", "q: criteria must not be empty" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[\"a\"]}}}", "q: a score has 2 to 10 levels" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[1,2,3,4,5,6,7,8,9,10,11]}}}", "a score has 2 to 10 levels" },
        .{ "{\"model\":\"x\",\"state\":1,\"questions\":{}}", "at least one question is required" },
        .{ many.items, "at most 64 questions" },
        .{ wide.items, "q: at most 256 options" },
        .{ "{\"model\":\"x\",\"state\":1,\"images\":[\"a\"],\"questions\":{\"q\":{\"type\":\"noul\"}}}", "images: this engine reads text records only" },
        .{ "{\"model\":\"x\",\"state\":1,\"videos\":[\"a\"],\"questions\":{\"q\":{\"type\":\"noul\"}}}", "videos: this engine reads text records only" },
        .{ long.items, "the schema needs more than 300 tokens" },
    };
    for (cases) |c| {
        const r = try post(a, srv, c[0]);
        const got = try text(a, r.body);
        if (r.status != 400 or std.mem.indexOf(u8, got, c[1]) == null) {
            std.debug.print("{d} {s}\n", .{ r.status, got });
            return error.TestUnexpectedResult;
        }
    }
    try t.expectEqual(@as(usize, 0), p.calls);
}

test "Clef System One: the head's logits answer in request order, `model` echoed, a state of null and `files` taken" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Provider = .{};
    const srv = try server(&p);
    defer srv.deinit();
    const body = "{\"model\":\"anything\",\"state\":null,\"files\":[\"ignored\"],\"raw\":true,\"questions\":{\"c\":{\"type\":\"choice\",\"criteria\":{\"z\":\"Z\",\"a\":\"A\"}},\"n\":{\"type\":\"noul\"}}}";
    const r = try post(a, srv, body);
    try t.expectEqual(@as(u16, 200), r.status);
    const got = try text(a, r.body);
    // encoding order a, z gets logits 0, 1: z wins; noul's true and false get 2 and 3
    try t.expect(std.mem.startsWith(u8, got, "{\"model\":\"anything\",\"answers\":{\"c\":{\"type\":\"choice\",\"choice\":\"z\",\"confidence\":0.7311,\"probabilities\":{\"z\":0.7311,\"a\":0.2689}},\"n\":{\"type\":\"noul\",\"noul\":0.2689}}"));
    try t.expect(std.mem.indexOf(u8, got, "\"usage\":{\"input_tokens\":") != null and std.mem.indexOf(u8, got, "\"raw\":{\"c\":") != null);
    try t.expectEqual(@as(usize, 4), p.seen_options);
    try t.expect(p.seen_untaught);
}

test "Clef System One: a non-finite logit or a run without its answer is a 500" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]@TypeOf(@as(Provider, .{}).mode){ .nan, .fail }) |mode| {
        var p: Provider = .{ .mode = mode };
        const srv = try server(&p);
        defer srv.deinit();
        const r = try post(a, srv, "{\"model\":\"x\",\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"noul\"}}}");
        try t.expectEqual(@as(u16, 500), r.status);
        try t.expect(std.mem.indexOf(u8, try text(a, r.body), "joint schema head") != null);
    }
}

test "Clef System One: chat and decisions on a readout engine refuse with 400 naming /v1/systemone; System One answers" {
    const http = @import("http_conn.zig");
    const Writer = struct {
        bytes: std.ArrayList(u8) = .empty,
        fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
            const w: *@This() = @ptrCast(@alignCast(ptr));
            w.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
        }
    };
    var p: Provider = .{};
    const srv = try server(&p);
    defer srv.deinit();
    const body = "{\"model\":\"m\",\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"noul\"}}}";
    for ([_][]const u8{ "/v1/chat/completions", "/v1/decisions", "/v1/systemone" }) |path| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var c = try http.Conn.init(t.allocator, -1, "fixture");
        defer c.deinit();
        var writer: Writer = .{};
        defer writer.bytes.deinit(t.allocator);
        c.write_hooks = .{ .ctx = &writer, .write = Writer.write };
        c.method = "POST";
        c.path = path;
        c.version = "HTTP/1.1";
        c.close = false;
        c.quiet_log = true;
        var len_buf: [8]u8 = undefined;
        var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.bufPrint(&len_buf, "{d}", .{body.len}) }};
        c.headers = &headers;
        @memcpy(c.buf[0..body.len], body);
        c.end = body.len;
        srv.dispatch(&c, arena.allocator());
        const out = writer.bytes.items;
        if (std.mem.eql(u8, path, "/v1/systemone")) {
            try t.expect(std.mem.indexOf(u8, out, "200 OK") != null and std.mem.indexOf(u8, out, "\"noul\": 0.2689") != null);
        } else try t.expect(std.mem.indexOf(u8, out, "400 Bad Request") != null and std.mem.indexOf(u8, out, "use POST /v1/systemone") != null);
    }
    try t.expectEqual(@as(usize, 1), p.calls);
}

test "Clef System One: 1,024 options in all pass validation, 1,025 are a 400" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Provider = .{};
    const srv = try Server.init(t.allocator, t.io, p.engine(), OnePiece.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    for ([_][2]usize{ .{ 4, 256 }, .{ 5, 205 } }) |shape| {
        var body: std.ArrayList(u8) = .empty;
        try body.appendSlice(a, "{\"model\":\"x\",\"state\":1,\"questions\":{");
        for (0..shape[0]) |q| {
            try body.print(a, "{s}\"q{d}\":{{\"type\":\"choice\",\"criteria\":{{", .{ if (q == 0) "" else ",", q });
            for (0..shape[1]) |o| try body.print(a, "{s}\"o{d}\":null", .{ if (o == 0) "" else ",", o });
            try body.appendSlice(a, "}}");
        }
        try body.appendSlice(a, "}}");
        const r = try post(a, srv, body.items);
        if (shape[0] * shape[1] == 1024) {
            try t.expectEqual(@as(u16, 200), r.status);
            try t.expectEqual(@as(usize, 1024), p.seen_options);
        } else {
            try t.expectEqual(@as(u16, 400), r.status);
            try t.expect(std.mem.indexOf(u8, try text(a, r.body), "at most 1024 options in all") != null);
        }
    }
}

test "Clef System One: the engine's submit refusals are 400s for the request's fault, 503s for the engine's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { api.SubmitError, u16 }{
        .{ error.InvalidRequest, 400 }, .{ error.InvalidSpans, 400 }, .{ error.Busy, 503 },
        .{ error.Closed, 503 },         .{ error.Stale, 503 },        .{ error.MemoryPressure, 503 },
    };
    for (cases) |c| {
        var p: Provider = .{ .mode = .refuse, .refusal = c[0] };
        const srv = try server(&p);
        defer srv.deinit();
        const r = try post(a, srv, "{\"model\":\"x\",\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"noul\"}}}");
        if (r.status != c[1]) {
            std.debug.print("{t}: {d}\n", .{ c[0], r.status });
            return error.TestUnexpectedResult;
        }
    }
}
