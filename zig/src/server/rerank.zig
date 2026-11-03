//! POST /v1/rerank as SGLang serves a Qwen3 reranker: each document's yes/no logits through the engine's score hook.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const reply_text = @import("reply_text.zig");
const model_text = @import("model_text.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Failure = errors.Refused || error{Cancelled};

/// The request's fields once checked; `documents` keeps the request's strings, in order.
const Ask = struct {
    query: []const u8,
    documents: []const []const u8,
    instruct: ?[]const u8,
    top_n: ?usize,
    return_documents: bool,
    return_logits: bool,
    return_ids: bool,
};

/// One document's score and the label logits it came from.
const Scored = struct { score: f64, yes: f64, no: f64 };

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch |err| {
        if (err == error.OutOfMemory) outOfMemory(&cx);
        return sendErr(conn, &cx);
    };
    const Gone = struct {
        conn: *Conn,
        fn check(g: @This()) bool {
            return g.conn.peerGone();
        }
    };
    const value = reply(srv, &cx, raw, Gone{ .conn = conn }) catch |err| {
        if (err == error.Cancelled) {
            conn.close = true;
            return;
        }
        if (err == error.OutOfMemory) outOfMemory(&cx);
        return sendErr(conn, &cx);
    };
    send(conn, a, 200, value);
}

/// SGLang's bytes: compact separators and UTF-8 as it is (FastAPI's ORJSONResponse).
pub fn encode(a: Allocator, v: Value) Allocator.Error![]u8 {
    return json.stringify(a, v, .{ .compact = true, .ascii = false }) catch return error.OutOfMemory;
}

fn send(conn: *Conn, a: Allocator, code: u16, v: Value) void {
    conn.sendJson(code, encode(a, v) catch return);
}

fn outOfMemory(cx: *Cx) void {
    cx.kind = .server;
    cx.message = "out of memory";
}

/// SGLang's error body; type "Bad Request" for a schema error (`other`), "BadRequestError" for a handler refusal.
pub fn errorBody(cx: *const Cx) Allocator.Error!Value {
    const status = cx.status();
    const o = try json.newObject(cx.a);
    try o.put(cx.a, "object", .{ .string = "error" });
    try o.put(cx.a, "message", .{ .string = cx.message });
    try o.put(cx.a, "type", .{ .string = switch (status) {
        500 => "InternalServerError",
        503 => "ServiceUnavailableError",
        else => if (cx.kind == .other) "Bad Request" else "BadRequestError",
    } });
    try o.put(cx.a, "param", .null);
    try o.put(cx.a, "code", try json.intValue(cx.a, status));
    return .{ .object = o };
}

fn sendErr(conn: *Conn, cx: *Cx) void {
    const body = errorBody(cx) catch return;
    send(conn, cx.a, cx.status(), body);
}

/// The ranked documents, or the refusal; `gone` says the client left (checked before each document's pass).
pub fn reply(srv: *Server, cx: *Cx, body: Value, gone: anytype) Failure!Value {
    const labels = srv.rerank orelse return cx.refuse("this model is not a reranker");
    const ask = try parse(cx, body);
    // each prompt is built in a scratch arena and dropped: a small body of many documents keeps no ids alive
    var scratch = std.heap.ArenaAllocator.init(srv.gpa);
    defer scratch.deinit();
    const window = srv.info.context_window;
    // every document fits the window before the first pass, so a refusal costs no GPU work
    for (ask.documents, 0..) |doc, i| {
        if (gone.check()) return error.Cancelled;
        _ = scratch.reset(.retain_capacity);
        const ids = try prompt(srv, cx, scratch.allocator(), ask, doc);
        if (window > 0 and ids.len > window)
            return cx.fail(.request, "document {d}: the prompt has {d} tokens, which does not fit the context length of {d} tokens", .{ i, ids.len, window });
    }
    const scored = try cx.a.alloc(Scored, ask.documents.len);
    for (ask.documents, scored) |doc, *row| {
        if (gone.check()) return error.Cancelled;
        _ = scratch.reset(.retain_capacity);
        const ids = try prompt(srv, cx, scratch.allocator(), ask, doc);
        var logits: [2]f64 = undefined;
        _ = srv.engine.score(ids, &.{ labels.yes, labels.no }, &logits, false) catch |err| switch (err) {
            error.Unsupported => return cx.refuse("this engine does not score labels"),
            error.Failed => return if (healthy(srv))
                cx.fail(.server, "rerank inference failed", .{})
            else
                cx.fail(.capacity, "the rerank engine is unavailable", .{}),
        };
        // a NaN or an infinity is no score, and would not even be JSON
        if (!std.math.isFinite(logits[0]) or !std.math.isFinite(logits[1])) return cx.fail(.server, "rerank inference failed", .{});
        row.* = .{ .score = 1.0 / (1.0 + @exp(logits[1] - logits[0])), .yes = logits[0], .no = logits[1] };
    }
    return build(srv, cx, ask, scored);
}

fn healthy(srv: *Server) bool {
    var st: api.Status = .{};
    srv.engine.status(&st, &.{});
    return st.healthy;
}

/// The request as SGLang reads it: its schema whole first (pydantic: a "Bad Request"), then its handler's checks.
fn parse(cx: *Cx, body: Value) errors.Refused!Ask {
    if (body != .object) return cx.other("body: input should be a valid dictionary");
    const query = try content(cx, body.get("query") orelse return cx.other("query: field required"), "query");
    const docs = body.get("documents") orelse return cx.other("documents: field required");
    if (docs != .array) return cx.other("documents: input should be a valid list");
    const parts = try cx.a.alloc(Content, docs.array.len);
    for (docs.array, parts) |d, *out| out.* = try content(cx, d, "documents");
    const instruct: ?[]const u8 = switch (body.get("instruct") orelse Value.null) {
        .null => null,
        .string => |str| if (str.len == 0) null else str,
        else => return cx.other("instruct: input should be a valid string"),
    };
    const top_n: ?usize = switch (body.get("top_n") orelse Value.null) {
        .null => null,
        else => |v| try topN(cx, v),
    };
    const documents = try cx.a.alloc([]const u8, parts.len);
    const ask: Ask = .{
        .query = query.text,
        .documents = documents,
        .instruct = instruct,
        .top_n = top_n,
        .return_documents = try flag(cx, body, "return_documents", true),
        .return_logits = try flag(cx, body, "return_logits", false),
        .return_ids = try flag(cx, body, "return_prompt_token_ids", false),
    };
    // the handler's checks, in its order (serving_rerank.py _validate_request), then this reranker's own
    if (query.empty) return cx.refuse("Query cannot be empty");
    if (query.string and reply_text.pyStrip(query.text).len == 0) return cx.refuse("Query cannot be empty or whitespace only");
    if (parts.len == 0) return cx.refuse("Documents cannot be empty");
    for (parts) |d| {
        if (d.empty) return cx.refuse("Each document must be a non-empty string");
        if (d.string and reply_text.pyStrip(d.text).len == 0) return cx.refuse("Each document cannot be empty or whitespace only");
    }
    if (query.media) |kind| return mediaRefused(cx, kind);
    for (parts) |d| if (d.media) |kind| return mediaRefused(cx, kind);
    // whitespace text parts: SGLang leaves them to the model; this reranker refuses a prompt with nothing to read
    if (reply_text.pyStrip(query.text).len == 0) return cx.refuse("Query cannot be empty or whitespace only");
    for (parts, documents) |d, *out| {
        if (reply_text.pyStrip(d.text).len == 0) return cx.refuse("Each document cannot be empty or whitespace only");
        out.* = d.text;
    }
    return ask;
}

/// A query or document as the schema reads it; `media` names a well-formed image or video part (unreadable here).
const Content = struct { text: []const u8, string: bool, empty: bool, media: ?[]const u8 = null };

fn content(cx: *Cx, v: Value, comptime field: []const u8) errors.Refused!Content {
    switch (v) {
        .string => |str| return .{ .text = str, .string = true, .empty = str.len == 0 },
        .array => |items| {
            var out: std.ArrayList(u8) = .empty;
            var media: ?[]const u8 = null;
            var texts: usize = 0;
            for (items) |part| {
                const kind = part.strField("type") orelse return cx.other(field ++ ": each content part needs a type");
                if (std.mem.eql(u8, kind, "text")) {
                    const t = part.strField("text") orelse return cx.other(field ++ ": a text part needs its text as a string");
                    if (texts > 0) try out.append(cx.a, ' ');
                    try out.appendSlice(cx.a, t);
                    texts += 1;
                } else if (std.mem.eql(u8, kind, "image_url") or std.mem.eql(u8, kind, "video_url")) {
                    // the part's payload, as the schema requires it: {"<type>": {"url": "..."}}
                    const payload = part.get(kind) orelse return cx.fail(.other, field ++ ": a {s} part needs its {s} object", .{ kind, kind });
                    if (payload.strField("url") == null) return cx.fail(.other, field ++ ": a {s} part needs a url string", .{kind});
                    media = media orelse kind;
                } else return cx.fail(.other, field ++ ": {s} is not a content part type", .{kind});
            }
            return .{ .text = out.items, .string = false, .empty = items.len == 0, .media = media };
        },
        else => return cx.other(field ++ ": input should be a valid string or a list of content parts"),
    }
}

fn mediaRefused(cx: *Cx, kind: []const u8) errors.Refused {
    return cx.fail(.request, "this reranker reads text: {s} parts are not supported", .{kind});
}

/// top_n as pydantic's lax int takes it: an integer, an integral float or string, or a bool; at least 1.
fn topN(cx: *Cx, v: Value) errors.Refused!usize {
    const bad = "top_n: input should be a valid integer";
    const digits: []const u8, const negative: bool = switch (v) {
        .int => |d| if (d[0] == '-') .{ d[1..], true } else .{ d, false },
        .bool => |b| .{ if (b) "1" else "0", false },
        .float => |x| blk: {
            // a float's integer within i64, as pydantic converts it; anything past it or fractional is refused
            if (!std.math.isFinite(x) or x != @trunc(x) or @abs(x) >= 0x1p63) return cx.other(bad);
            if (x < 1) return cx.other("Value error, parameter top_n should be larger than 0.");
            break :blk .{ std.fmt.allocPrint(cx.a, "{d}", .{@as(u64, @intFromFloat(x))}) catch return error.OutOfMemory, false };
        },
        .string => |raw| try intText(cx, reply_text.pyStrip(raw), bad),
        else => return cx.other(bad),
    };
    const n = std.fmt.parseInt(usize, digits, 10) catch |err| switch (err) {
        error.Overflow => std.math.maxInt(usize), // any positive integer: more than the documents returns them all
        error.InvalidCharacter => return cx.other(bad),
    };
    if (negative or n == 0) return cx.other("Value error, parameter top_n should be larger than 0.");
    return n;
}

/// A string's integer as pydantic reads it (sign, single underscores, an all-zero fraction): its digits and sign.
fn intText(cx: *Cx, str: []const u8, comptime bad: []const u8) errors.Refused!struct { []const u8, bool } {
    var rest = str;
    var negative = false;
    if (rest.len > 0 and (rest[0] == '-' or rest[0] == '+')) {
        negative = rest[0] == '-';
        rest = rest[1..];
    }
    const whole = if (std.mem.indexOfScalar(u8, rest, '.')) |dot| blk: {
        if (dot + 1 == rest.len) return cx.other(bad); // "2." has no fraction to be zero
        for (rest[dot + 1 ..]) |c| if (c != '0') return cx.other(bad);
        break :blk rest[0..dot];
    } else rest;
    if (whole.len == 0 or whole[0] == '_' or whole[whole.len - 1] == '_') return cx.other(bad);
    var out: std.ArrayList(u8) = .empty;
    for (whole, 0..) |c, i| {
        if (c == '_') {
            if (whole[i - 1] == '_') return cx.other(bad);
            continue;
        }
        if (c < '0' or c > '9') return cx.other(bad);
        try out.append(cx.a, c);
    }
    return .{ out.items, negative };
}

/// An optional boolean as pydantic's lax bool takes it (true/false, 0/1, and its yes/no words); null is refused.
fn flag(cx: *Cx, body: Value, comptime key: []const u8, default: bool) errors.Refused!bool {
    const v = body.get(key) orelse return default;
    const bad = key ++ ": input should be a valid boolean";
    return switch (v) {
        .bool => |b| b,
        .int => |d| if (std.mem.eql(u8, d, "0")) false else if (std.mem.eql(u8, d, "1")) true else cx.other(bad),
        .float => |x| if (x == 0) false else if (x == 1) true else cx.other(bad),
        .string => |str| blk: {
            var buf: [8]u8 = undefined;
            if (str.len > buf.len) break :blk cx.other(bad);
            const low = std.ascii.lowerString(&buf, str);
            for ([_][]const u8{ "0", "off", "f", "false", "n", "no" }) |w| if (std.mem.eql(u8, low, w)) break :blk false;
            for ([_][]const u8{ "1", "on", "t", "true", "y", "yes" }) |w| if (std.mem.eql(u8, low, w)) break :blk true;
            break :blk cx.other(bad);
        },
        else => cx.other(bad),
    };
}

/// The checkpoint template over [system?, query, document] in `a`; the request's text is encoded as plain text.
fn prompt(srv: *Server, cx: *Cx, a: Allocator, ask: Ask, document: []const u8) errors.Refused![]const u32 {
    var messages: std.ArrayList(Value) = .empty;
    if (ask.instruct) |i| try messages.append(a, try message(a, "system", i));
    try messages.append(a, try message(a, "query", ask.query));
    try messages.append(a, try message(a, "document", document));
    var problem: []const u8 = "";
    const rendered = srv.text.render(a, .{ .array = messages.items }, .{ .add_generation_prompt = true, .enable_thinking = false }, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{problem}),
    };
    return rendered.encode(srv.text, a, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{"the tokenizer cannot encode this prompt"}),
    };
}

fn message(a: Allocator, role: []const u8, words: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = words });
    return .{ .object = o };
}

/// The array SGLang sends: best first, equal scores in request order, cut to top_n.
fn build(srv: *Server, cx: *Cx, ask: Ask, scored: []const Scored) errors.Refused!Value {
    const order = try cx.a.alloc(usize, scored.len);
    for (order, 0..) |*o, i| o.* = i;
    std.sort.insertion(usize, order, scored, struct {
        fn before(s: []const Scored, x: usize, y: usize) bool {
            return s[x].score > s[y].score;
        }
    }.before);
    const n = @min(order.len, ask.top_n orelse order.len);
    const out = try cx.a.alloc(Value, n);
    for (order[0..n], out) |i, *item| {
        const o = try json.newObject(cx.a);
        try o.put(cx.a, "score", .{ .float = scored[i].score });
        if (ask.return_documents) try o.put(cx.a, "document", .{ .string = ask.documents[i] });
        try o.put(cx.a, "index", try json.intValue(cx.a, i));
        // the returned documents' ids rendered again (the same bytes give the same ids): none kept while scoring
        const ids = if (ask.return_ids) try prompt(srv, cx, cx.a, ask, ask.documents[i]) else &.{};
        try o.put(cx.a, "meta_info", try meta(cx.a, ask, scored[i], ids));
        item.* = .{ .object = o };
    }
    return .{ .array = out };
}

/// Null unless the request asked for the logits or the ids (TensorFold's extensions; SGLang clients send neither).
fn meta(a: Allocator, ask: Ask, s: Scored, ids: []const u32) Allocator.Error!Value {
    if (!ask.return_logits and !ask.return_ids) return .null;
    const o = try json.newObject(a);
    if (ask.return_logits) {
        try o.put(a, "logit_yes", .{ .float = s.yes });
        try o.put(a, "logit_no", .{ .float = s.no });
    }
    if (ask.return_ids) {
        const list = try a.alloc(Value, ids.len);
        for (ids, list) |id, *v| v.* = try json.intValue(a, id);
        try o.put(a, "prompt_token_ids", .{ .array = list });
    }
    return .{ .object = o };
}
