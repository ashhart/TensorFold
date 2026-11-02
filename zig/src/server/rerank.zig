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

/// SGLang's error body: top-level fields, a status-derived type, the status again as `code`.
pub fn errorBody(cx: *const Cx) Allocator.Error!Value {
    const status = cx.status();
    const o = try json.newObject(cx.a);
    try o.put(cx.a, "object", .{ .string = "error" });
    try o.put(cx.a, "message", .{ .string = cx.message });
    try o.put(cx.a, "type", .{ .string = switch (status) {
        500 => "InternalServerError",
        503 => "ServiceUnavailableError",
        else => "BadRequestError",
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

fn parse(cx: *Cx, body: Value) errors.Refused!Ask {
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    const query = body.get("query") orelse return cx.refuse("Query cannot be empty");
    const q = try text(cx, query, "Query");
    const docs = body.get("documents") orelse return cx.refuse("Documents cannot be empty");
    if (docs == .string) return cx.refuse(strings_only);
    if (docs != .array) return cx.refuse("documents must be an array of strings");
    if (docs.array.len == 0) return cx.refuse("Documents cannot be empty");
    const documents = try cx.a.alloc([]const u8, docs.array.len);
    for (docs.array, documents) |d, *out| out.* = try text(cx, d, "Each document");
    const instruct: ?[]const u8 = switch (body.get("instruct") orelse Value.null) {
        .null => null,
        .string => |s| if (s.len == 0) null else s,
        else => return cx.refuse("instruct must be a string"),
    };
    const top_n_value = body.get("top_n") orelse Value.null;
    const top_n: ?usize = switch (top_n_value) {
        .null => null,
        // any positive integer, however large: more than the documents returns them all
        .int => |digits| blk: {
            if (digits[0] == '-' or std.mem.eql(u8, digits, "0")) return cx.refuse("Value error, parameter top_n should be larger than 0.");
            break :blk std.fmt.parseInt(usize, digits, 10) catch std.math.maxInt(usize);
        },
        else => return cx.refuse("top_n must be an integer"),
    };
    return .{
        .query = q,
        .documents = documents,
        .instruct = instruct,
        .top_n = top_n,
        .return_documents = try flag(cx, body, "return_documents", true),
        .return_logits = try flag(cx, body, "return_logits", false),
        .return_ids = try flag(cx, body, "return_prompt_token_ids", false),
    };
}

const strings_only = "this reranker reads text: query and documents must be strings";

/// A query or document: a string with something past whitespace, in SGLang's words.
fn text(cx: *Cx, v: Value, comptime what: []const u8) errors.Refused![]const u8 {
    const s = switch (v) {
        .string => |s| s,
        .array => return cx.refuse(strings_only),
        else => return cx.refuse(what ++ " must be a string"),
    };
    if (s.len == 0) return cx.refuse(if (comptime std.mem.eql(u8, what, "Query")) "Query cannot be empty" else "Each document must be a non-empty string");
    if (reply_text.pyStrip(s).len == 0) return cx.refuse(what ++ " cannot be empty or whitespace only");
    return s;
}

/// An optional boolean field; null is refused, as SGLang's fields are not nullable.
fn flag(cx: *Cx, body: Value, comptime key: []const u8, default: bool) errors.Refused!bool {
    const v = body.get(key) orelse return default;
    if (v != .bool) return cx.refuse(key ++ " must be a boolean");
    return v.bool;
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

fn message(a: Allocator, role: []const u8, content: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
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
