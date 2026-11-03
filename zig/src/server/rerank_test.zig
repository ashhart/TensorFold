//! /v1/rerank against a fake template, tokenizer and score hook: SGLang's request rules, order and error bodies.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const rerank = @import("rerank.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const t = std.testing;

const im_end = "<|im_end|>";
const im_end_id: u32 = 300;

/// Renders `S:<system>|Q:<query>|D:<document>|A` with the three contents as request spans; one special token.
const Text = struct {
    fn model(self: *Text) model_text.Text {
        return .{ .ctx = self, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .encode_masked = encodeMasked } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        return encodeMasked(undefined, a, input, &.{}, undefined);
    }

    /// Template bytes read `<|im_end|>` as its token; request bytes stay plain.
    fn encodeMasked(_: *anyopaque, a: Allocator, input: []const u8, spans: []const model_text.Span, _: *[]const u8) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < input.len) {
            var plain = false;
            for (spans) |s| plain = plain or (i >= s[0] and i < s[1]);
            if (!plain and std.mem.startsWith(u8, input[i..], im_end)) {
                try ids.append(a, im_end_id);
                i += im_end.len;
            } else {
                try ids.append(a, input[i]);
                i += 1;
            }
        }
        return ids.toOwnedSlice(a);
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id == im_end_id) try out.appendSlice(a, "#END#") else try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 400;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        var spans: std.ArrayList(model_text.Span) = .empty;
        for ([_][2][]const u8{ .{ "system", "S:" }, .{ "query", "|Q:" }, .{ "document", "|D:" } }) |part| {
            try out.appendSlice(a, part[1]);
            for (messages.array) |m| if (std.mem.eql(u8, m.get("role").?.string, part[0])) {
                const c = m.get("content").?.string;
                try spans.append(a, .{ out.items.len, out.items.len + c.len });
                try out.appendSlice(a, c);
            };
        }
        try out.appendSlice(a, "|A" ++ im_end);
        return .{ .text = out.items, .spans = spans.items };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// A score hook whose yes/no logits come from a table keyed by the document's text; it keeps each prompt.
const Scorer = struct {
    gpa: Allocator,
    window: u32 = 0,
    hook: bool = true,
    failing: bool = false,
    healthy: bool = true,
    table: []const struct { []const u8, f64, f64 } = &.{},
    prompts: std.ArrayList([]u8) = .empty,

    fn engine(s: *Scorer) api.Engine {
        return .{ .ctx = s, .vtable = if (s.hook) &scored else &bare };
    }

    fn of(ctx: *anyopaque) *Scorer {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .text_generation = false, .decides = true, .context_window = of(ctx).window };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = of(ctx).healthy };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const s = of(ctx);
        if (s.failing or mass or labels.len != 2 or labels[0] != 7 or labels[1] != 9) return error.Failed;
        const text = Text.decode(undefined, s.gpa, prompt) catch return error.Failed;
        s.prompts.append(s.gpa, text) catch return error.Failed;
        const at = std.mem.indexOf(u8, text, "|D:").? + 3;
        const doc = text[at..std.mem.lastIndexOf(u8, text, "|A").?];
        for (s.table) |row| if (std.mem.eql(u8, row[0], doc)) {
            logits[0] = row[1];
            logits[1] = row[2];
            return 0;
        };
        logits[0] = 0;
        logits[1] = 0;
        return 0;
    }

    fn deinit(s: *Scorer) void {
        for (s.prompts.items) |p| s.gpa.free(p);
        s.prompts.deinit(s.gpa);
    }

    const bare: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
    const scored: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score };
};

const Present = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

fn serve(text: *Text, s: *Scorer) !*Server {
    const srv = try Server.init(t.allocator, t.io, s.engine(), text.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    srv.info.rerank = .{ .yes = 7, .no = 9 };
    return srv;
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

const table = .{ .{ "low", -2.0, 1.0 }, .{ "high", 3.0, -1.0 }, .{ "mid", 0.5, 0.0 }, .{ "tie", 0.5, 0.0 } };

fn ask(srv: *Server, a: Allocator, body: []const u8) !json.Value {
    var cx: errors.Cx = .{ .a = a };
    return rerank.reply(srv, &cx, (try json.parse(a, body)).ok, Present{});
}

/// The response's ranked `results`.
fn results(out: json.Value) []json.Value {
    return out.get("results").?.array;
}

test "rerank: results come back best first in #1895's shape, scored sigmoid(yes - no), with usage" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "{\"model\": \"anything\", \"query\": \"q\", \"documents\": [\"low\", \"high\", \"mid\"], \"extra\": 1}");
    try t.expectEqual(@as(usize, 3), results(out).len);
    try t.expectEqualStrings("results", out.object.keys()[0]);
    try t.expectEqualStrings("usage", out.object.keys()[1]);
    // the three prompts' tokens: "S:|Q:q|D:" (9) + the document + "|A" (2) + the end token: 15 + 16 + 15
    try t.expectEqual(@as(i64, 46), out.get("usage").?.get("total_tokens").?.int64().?);
    const want = [_]struct { usize, f64, []const u8 }{ .{ 1, sigmoid(4), "high" }, .{ 2, sigmoid(0.5), "mid" }, .{ 0, sigmoid(-3), "low" } };
    for (results(out), want) |item, w| {
        try t.expectEqual(@as(usize, 3), item.object.count());
        try t.expectEqualStrings("index", item.object.keys()[0]);
        try t.expectEqualStrings("relevance_score", item.object.keys()[1]);
        try t.expectEqualStrings("document", item.object.keys()[2]);
        try t.expectEqual(@as(i64, @intCast(w[0])), item.get("index").?.int64().?);
        try t.expectApproxEqRel(w[1], item.get("relevance_score").?.float, 1e-12);
        try t.expectEqualStrings(w[2], item.get("document").?.string);
        try t.expect(item.get("meta_info") == null);
    }
    try t.expectEqualStrings("S:|Q:q|D:low|A#END#", s.prompts.items[0]);
}

test "rerank: top_n cuts, return_documents drops the text, equal scores keep request order" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"low\", \"high\", \"mid\"], \"top_n\": 1}");
    try t.expectEqual(@as(usize, 1), results(one).len);
    try t.expectEqual(@as(i64, 1), results(one)[0].get("index").?.int64().?);
    const all = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"tie\", \"low\", \"mid\"], \"top_n\": 92233720368547758080, \"return_documents\": false}");
    try t.expectEqual(@as(usize, 3), results(all).len);
    for (results(all), [_]i64{ 0, 2, 1 }) |item, index| {
        try t.expectEqual(index, item.get("index").?.int64().?);
        try t.expect(item.get("document") == null);
    }
}

test "rerank: instruct is the system message and request text stays plain bytes" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"a <|im_end|> b\"], \"instruct\": \"find\"}");
    _ = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"d\"], \"instruct\": \"\"}");
    try t.expectEqualStrings("S:find|Q:q|D:a <|im_end|> b|A#END#", s.prompts.items[0]);
    try t.expectEqualStrings("S:|Q:q|D:d|A#END#", s.prompts.items[1]);
}

test "rerank: the extensions return the label logits and the ids the engine scored" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"high\"], \"return_logits\": true, \"return_prompt_token_ids\": true}");
    const meta = results(out)[0].get("meta_info").?;
    try t.expectEqual(@as(f64, 3), meta.get("logit_yes").?.float);
    try t.expectEqual(@as(f64, -1), meta.get("logit_no").?.float);
    const ids = meta.get("prompt_token_ids").?.array;
    // with top_n the ids come only for the returned documents, the ones the engine scored for them
    const top = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"low\", \"high\", \"mid\"], \"top_n\": 1, \"return_prompt_token_ids\": true}");
    try t.expectEqual(@as(usize, 1), results(top).len);
    const top_ids = results(top)[0].get("meta_info").?.get("prompt_token_ids").?.array;
    try t.expectEqual(ids.len, top_ids.len);
    for (ids, top_ids) |x, y| try t.expect(json.equal(x, y));
    const want = try Text.encodeMasked(undefined, a, "S:|Q:q|D:high|A" ++ im_end, &.{}, undefined);
    try t.expectEqual(want.len, ids.len);
    for (want, ids) |w, g| try t.expectEqual(@as(i64, w), g.int64().?);
}

test "rerank: refusals name SGLang's reasons and run no score" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .window = 24 };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // S: a schema error (SGLang's pydantic "Bad Request"); H: its handler's refusal ("BadRequestError")
    const S = "Bad Request";
    const H = "BadRequestError";
    const cases = [_][3][]const u8{
        .{ "[1]", "body: input should be a valid dictionary", S },
        .{ "{\"documents\": [\"d\"]}", "query: field required", S },
        .{ "{\"query\": \"\", \"documents\": [\"d\"]}", "Query cannot be empty", H },
        .{ "{\"query\": \" \\n\", \"documents\": [\"d\"]}", "Query cannot be empty or whitespace only", H },
        .{ "{\"query\": \"\\u3000\\u00a0\", \"documents\": [\"d\"]}", "Query cannot be empty or whitespace only", H },
        .{ "{\"query\": \"q\", \"documents\": [\"\\u001c\\u2028\"]}", "Each document cannot be empty or whitespace only", H },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": -3}", "Value error, parameter top_n should be larger than 0.", S },
        .{ "{\"query\": [], \"documents\": [\"d\"]}", "Query cannot be empty", H },
        .{ "{\"query\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"x\"}}], \"documents\": [\"d\"]}", "this reranker reads text: image_url parts are not supported", H },
        .{ "{\"query\": [{\"text\": \"q\"}], \"documents\": [\"d\"]}", "query: each content part needs a type", S },
        .{ "{\"query\": [{\"type\": \"text\", \"text\": 3}], \"documents\": [\"d\"]}", "query: a text part needs its text as a string", S },
        .{ "{\"query\": [{\"type\": \"bogus\"}], \"documents\": [\"d\"]}", "query: bogus is not a content part type", S },
        .{ "{\"query\": \"q\", \"documents\": [[{\"type\": \"video_url\"}]]}", "documents: a video_url part needs its video_url object", S },
        .{ "{\"query\": [{\"type\": \"image_url\", \"image_url\": {}}], \"documents\": [\"d\"]}", "query: a image_url part needs a url string", S },
        .{ "{\"query\": \"\", \"documents\": \"d\"}", "documents: input should be a valid list", S },
        .{ "{\"query\": [{\"type\": \"text\", \"text\": \" \"}], \"documents\": [\"d\"]}", "Query cannot be empty or whitespace only", H },
        .{ "{\"query\": 3, \"documents\": [\"d\"]}", "query: input should be a string, a {\"text\"} object or a list of content parts", S },
        .{ "{\"query\": \"q\"}", "documents: field required", S },
        .{ "{\"query\": \"q\", \"documents\": []}", "Documents cannot be empty", H },
        .{ "{\"query\": \"q\", \"documents\": \"d\"}", "documents: input should be a valid list", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\", \"\"]}", "Each document must be a non-empty string", H },
        .{ "{\"query\": \"q\", \"documents\": [\"\\t\"]}", "Each document cannot be empty or whitespace only", H },
        .{ "{\"query\": \"q\", \"documents\": [[{\"type\": \"video_url\", \"video_url\": {\"url\": \"v\"}}]]}", "this reranker reads text: video_url parts are not supported", H },
        .{ "{\"query\": \"q\", \"documents\": [1]}", "documents: input should be a string, a {\"text\"} object or a list of content parts", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": 0}", "Value error, parameter top_n should be larger than 0.", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"two\"}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": 1.5}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": 1e19}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"1__0\"}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"_1\"}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"2.5\"}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"2.\"}", "top_n: input should be a valid integer", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"top_n\": \"0\"}", "Value error, parameter top_n should be larger than 0.", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"return_documents\": \"maybe\"}", "return_documents: input should be a valid boolean", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"return_documents\": null}", "return_documents: input should be a valid boolean", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"return_logits\": 2}", "return_logits: input should be a valid boolean", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"instruct\": 2}", "instruction: input should be a valid string", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\", \"a document past the window\"]}", "document 1: the prompt has 38 tokens, which does not fit the context length of 24 tokens", H },
        .{ "{\"query\": \"q\", \"documents\": [\"d\", \"dd\"], \"max_length\": 13}", "document 1: the prompt has 14 tokens, past max_length 13; this server does not truncate", H },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"max_length\": 0}", "max_length: input should be at least 1", S },
        .{ "{\"query\": \"q\", \"documents\": [\"d\"], \"max_length\": \"8\"}", "max_length: input should be a valid integer", S },
    };
    for (cases) |case| {
        var cx: errors.Cx = .{ .a = a };
        try t.expectError(error.Refused, rerank.reply(srv, &cx, (try json.parse(a, case[0])).ok, Present{}));
        try t.expectEqualStrings(case[1], cx.message);
        try t.expectEqual(@as(u16, 400), cx.status());
        try t.expectEqualStrings(case[2], (try rerank.errorBody(&cx)).get("type").?.string);
    }
    try t.expectEqual(@as(usize, 0), s.prompts.items.len);
}

test "rerank: no labels, no hook, a failed pass and a stopping engine each fail in their own class" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"d\"]}")).ok;
    const Want = struct { []const u8, u16 };
    const cases = [_]struct { ?api.Rerank, bool, bool, bool, Want }{
        .{ null, true, false, true, .{ "this model is not a reranker", 400 } },
        .{ .{ .yes = 7, .no = 9 }, false, false, true, .{ "this engine does not score labels", 400 } },
        .{ .{ .yes = 7, .no = 9 }, true, true, true, .{ "rerank inference failed", 500 } },
        .{ .{ .yes = 7, .no = 9 }, true, true, false, .{ "the rerank engine is unavailable", 503 } },
    };
    for (cases) |case| {
        srv.info.rerank = case[0];
        s.hook = case[1];
        srv.engine = s.engine();
        s.failing = case[2];
        s.healthy = case[3];
        var cx: errors.Cx = .{ .a = a };
        try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
        try t.expectEqualStrings(case[4][0], cx.message);
        try t.expectEqual(case[4][1], cx.status());
        const err = try rerank.encode(a, try rerank.errorBody(&cx));
        const want = try std.fmt.allocPrint(a, "{{\"object\":\"error\",\"message\":\"{s}\",\"type\":\"{s}\",\"param\":null,\"code\":{d}}}", .{ case[4][0], switch (case[4][1]) {
            500 => "InternalServerError",
            503 => "ServiceUnavailableError",
            else => "BadRequestError",
        }, case[4][1] });
        try t.expectEqualStrings(want, err);
    }
}

test "rerank: a client gone before a document stops before its pass" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const After = struct {
        scorer: *Scorer,
        pub fn check(g: @This()) bool {
            return g.scorer.prompts.items.len >= 1;
        }
    };
    var cx: errors.Cx = .{ .a = a };
    const body = (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"a\", \"b\", \"c\"]}")).ok;
    try t.expectError(error.Cancelled, rerank.reply(srv, &cx, body, After{ .scorer = &s }));
    try t.expectEqual(@as(usize, 1), s.prompts.items.len);
}

/// The bytes a dispatched request wrote back.
const Wire = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(ptr));
        w.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
    }
};

/// POSTs `body` to `path` through the server's dispatcher; the reply's bytes land in `wire`.
fn dispatch(srv: *Server, path: []const u8, body: []const u8, wire: *Wire) !void {
    return dispatchAs(srv, "POST", path, body, wire);
}

/// `dispatch` by `method`.
fn dispatchAs(srv: *Server, method: []const u8, path: []const u8, body: []const u8, wire: *Wire) !void {
    const http = @import("http_conn.zig");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try http.Conn.init(t.allocator, -1, "fixture");
    defer c.deinit();
    c.write_hooks = .{ .ctx = wire, .write = Wire.write };
    c.method = method;
    c.path = path;
    c.version = "HTTP/1.1";
    c.close = false;
    c.quiet_log = true;
    var len: [16]u8 = undefined;
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.bufPrint(&len, "{d}", .{body.len}) }};
    c.headers = &headers;
    @memcpy(c.buf[0..body.len], body);
    @memcpy(c.buf[body.len..][0..4], "NEXT");
    c.end = body.len + 4;
    srv.dispatch(&c, arena.allocator());
    try t.expect(c.response_complete);
    try t.expectEqualStrings("NEXT", c.buf[c.start..c.end]);
}

test "rerank: a reranker server sends every other engine route to /v1/rerank and serves the route itself" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    for ([_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/audio/speech", "/v1/decisions", "/v1/systemone", "/v1/embeddings" }) |path| {
        var wire: Wire = .{};
        defer wire.bytes.deinit(t.allocator);
        try dispatch(srv, path, "{}", &wire);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "400 Bad Request") != null);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "this model is a reranker: use POST /v1/rerank") != null);
    }
    var ok: Wire = .{};
    defer ok.bytes.deinit(t.allocator);
    try dispatch(srv, "/rerank", "{\"query\": \"q\", \"documents\": [\"low\", \"high\"]}", &ok);
    try t.expect(std.mem.indexOf(u8, ok.bytes.items, "200 OK") != null);
    try t.expect(std.mem.indexOf(u8, ok.bytes.items, "\"index\":1") != null);
    var bad: Wire = .{};
    defer bad.bytes.deinit(t.allocator);
    try dispatch(srv, "/v1/rerank", "{not js", &bad);
    try t.expect(std.mem.indexOf(u8, bad.bytes.items, "400 Bad Request") != null);
    try t.expect(std.mem.indexOf(u8, bad.bytes.items, "{\"object\":\"error\",") != null);
    try t.expect(std.mem.indexOf(u8, bad.bytes.items, "\"type\":\"Bad Request\",\"param\":null,\"code\":400}") != null); // SGLang: json_invalid
    try t.expectEqual(@as(usize, 2), s.prompts.items.len);
    try t.expectEqual(@as(u32, 0), srv.generating.load(.seq_cst));
    const scope = @import("auth_scope.zig");
    for ([_][]const u8{ "/v1/rerank", "/rerank" }) |path| {
        try t.expect(scope.allows(.{ .inference = true }, "POST", path));
        try t.expect(!scope.allows(.{ .read = true }, "POST", path));
    }
}

test "rerank: the engine declares a reranker; a 1_LogitScore file in the served folder alone makes none" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.Io.Dir.cwd().createDirPath(t.io, try std.fs.path.join(a, &.{ dir, "1_LogitScore" }));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "1_LogitScore/config.json", .data = "{\"true_token_id\": 7, \"false_token_id\": 9}" });
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator };
    defer s.deinit();
    const srv = try Server.init(t.allocator, t.io, s.engine(), text.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .model_dir = dir }, null);
    defer srv.deinit();
    try t.expectEqual(@as(?api.Rerank, null), srv.info.rerank);
    const body = (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"d\"]}")).ok;
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqualStrings("this model is not a reranker", cx.message);
    try t.expectEqual(@as(usize, 0), s.prompts.items.len);
}

test "rerank: a NaN or an infinite logit is a failed pass, never a 200 with invalid JSON" {
    var text: Text = .{};
    const bad = .{ .{ "nan", std.math.nan(f64), 0.0 }, .{ "inf", std.math.inf(f64), 0.0 }, .{ "ninf", 0.0, -std.math.inf(f64) } };
    var s: Scorer = .{ .gpa = t.allocator, .table = &bad };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "nan", "inf", "ninf" }) |doc| {
        var cx: errors.Cx = .{ .a = a };
        const body = try std.fmt.allocPrint(a, "{{\"query\": \"q\", \"documents\": [\"{s}\"], \"return_logits\": true}}", .{doc});
        try t.expectError(error.Refused, rerank.reply(srv, &cx, (try json.parse(a, body)).ok, Present{}));
        try t.expectEqualStrings("rerank inference failed", cx.message);
        try t.expectEqual(@as(u16, 500), cx.status());
    }
}

test "rerank: text parts join with a space, as SGLang extracts them, and the document returns as the request sent it" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "{\"query\": [{\"type\": \"text\", \"text\": \"a\"}, {\"type\": \"text\", \"text\": \"b\"}], \"documents\": [[{\"type\": \"text\", \"text\": \"high\"}], \"mid\"]}");
    try t.expectEqualStrings("S:|Q:a b|D:high|A#END#", s.prompts.items[0]);
    // the document comes back as the request sent it: here its content-part list
    try t.expectEqualStrings("high", results(out)[0].get("document").?.array[0].get("text").?.string);
    try t.expectEqual(@as(i64, 0), results(out)[0].get("index").?.int64().?);
}

test "rerank: PUT reaches the route as POST does, under the same key scope; other routes keep refusing PUT" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    for ([_][]const u8{ "/v1/rerank", "/rerank" }) |path| {
        var wire: Wire = .{};
        defer wire.bytes.deinit(t.allocator);
        try dispatchAs(srv, "PUT", path, "{\"query\": \"q\", \"documents\": [\"low\", \"high\"]}", &wire);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "200 OK") != null);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "\"index\":1") != null);
    }
    var other: Wire = .{};
    defer other.bytes.deinit(t.allocator);
    try dispatchAs(srv, "PUT", "/v1/completions", "{}", &other);
    try t.expect(std.mem.indexOf(u8, other.bytes.items, "501") != null);
    const scope = @import("auth_scope.zig");
    try t.expect(scope.allows(.{ .inference = true }, "PUT", "/v1/rerank"));
    try t.expect(!scope.allows(.{ .read = true }, "PUT", "/v1/rerank"));
    try t.expect(!scope.allows(.{ .inference = true }, "PUT", "/v1/completions"));
}

test "rerank: top_n and the flags take what pydantic's lax mode takes, as SGLang's schema does" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const docs = "\"documents\": [\"low\", \"high\", \"mid\"]";
    for ([_][]const u8{ "\"2\"", "2.0", "\" 2 \"", "\"+2\"", "\"2.0\"", "\"\\u00a02\\u3000\"", "\"0_2\"" }) |n| {
        const out = try ask(srv, a, try std.fmt.allocPrint(a, "{{\"query\": \"q\", {s}, \"top_n\": {s}}}", .{ docs, n }));
        try t.expectEqual(@as(usize, 2), results(out).len);
    }
    for ([_][]const u8{ "\"false\"", "0", "\"no\"", "\"OFF\"", "0.0" }) |f| {
        const out = try ask(srv, a, try std.fmt.allocPrint(a, "{{\"query\": \"q\", {s}, \"return_documents\": {s}}}", .{ docs, f }));
        try t.expect(results(out)[0].get("document") == null);
    }
    for ([_][]const u8{ "\"true\"", "1", "\"Yes\"", "\"t\"" }) |f| {
        const out = try ask(srv, a, try std.fmt.allocPrint(a, "{{\"query\": \"q\", {s}, \"return_documents\": {s}}}", .{ docs, f }));
        try t.expect(results(out)[0].get("document") != null);
    }
}

test "rerank: #1895's request fields: {text} documents and instruction, echoed and used" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const out = try ask(srv, a, "{\"query\": \"q\", \"instruction\": \"find\", \"documents\": [{\"text\": \"low\"}, {\"text\": \"high\"}]}");
    try t.expectEqualStrings("S:find|Q:q|D:low|A#END#", s.prompts.items[0]);
    try t.expectEqual(@as(i64, 1), results(out)[0].get("index").?.int64().?);
    try t.expectEqualStrings("high", results(out)[0].get("document").?.get("text").?.string);
}
