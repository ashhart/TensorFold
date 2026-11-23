//! Score and rerank wire formats translated onto the existing reranker prompt and score hooks.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const rerank = @import("rerank.zig");
const strip = @import("reply_text.zig").pyStrip;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;

/// The routes this file answers: the pair-score aliases, both rerank aliases and Cohere's v2 path.
pub fn route(path: []const u8) bool {
    for ([_][]const u8{ "/score", "/v1/score", "/rerank", "/v1/rerank", "/v2/rerank" }) |p| if (std.mem.eql(u8, path, p)) return true;
    return false;
}

/// The HTTP reply for every `route`, the legacy /rerank body included: one reader, one envelope.
pub fn post(srv: *Server, conn: *Conn, a: std.mem.Allocator) void {
    var cx: Cx = .{ .a = a };
    const body = @import("http_body.zig").readJson(conn, &cx) catch |err| {
        if (err == error.OutOfMemory) oom(&cx);
        return sendError(conn, &cx);
    };
    const Gone = struct {
        conn: *Conn,
        pub fn check(g: @This()) bool {
            return g.conn.peerGone();
        }
    };
    const out = reply(srv, &cx, @import("auth.zig").routePath(conn.path), body, Gone{ .conn = conn }) catch |err| {
        if (err == error.Cancelled) {
            conn.close = true;
            return;
        }
        if (err == error.OutOfMemory) oom(&cx);
        return sendError(conn, &cx);
    };
    conn.sendJson(200, rerank.encode(a, out) catch return);
}
fn oom(cx: *Cx) void {
    cx.kind = .server;
    cx.message = "out of memory";
}
fn sendError(conn: *Conn, cx: *Cx) void {
    const body = rerank.errorBody(cx) catch return;
    conn.sendJson(cx.status(), rerank.encode(cx.a, body) catch return);
}

/// `path`'s shape as rerank.zig's request, its scores read back unchanged into that shape's response.
pub fn reply(srv: *Server, cx: *Cx, path: []const u8, body: Value, gone: anytype) rerank.Failure!Value {
    const score = std.mem.eql(u8, path, "/score") or std.mem.eql(u8, path, "/v1/score");
    const cohere = std.mem.eql(u8, path, "/v2/rerank");
    const tei = !score and !cohere and body.has("texts") and !body.has("documents");
    if (!score and !cohere and !tei) return rerank.reply(srv, cx, body, gone);
    if (srv.info.rerank == null) return cx.refuse("this model is not a reranker");
    if (srv.engine.vtable.score == null and srv.engine.vtable.decide == null) return cx.refuse("this engine does not score labels");
    if (body != .object) return cx.refuse("body must be a JSON object");
    if (body.field("model")) |model| {
        if (model != .string) return cx.refuse("model must be a string");
        var found = false;
        for (srv.config.model_ids) |id| found = found or std.mem.eql(u8, id, model.string);
        if (!found) return cx.refuse("model is not served here");
    }
    for ([_][]const u8{ "rank_fields", "max_tokens_per_doc", "priority" }) |name| if (body.has(name))
        return cx.fail(.request, "{s} is not supported", .{name});
    if (score) return scores(srv, cx, body, gone);
    const key: []const u8 = if (tei) "texts" else "documents";
    const docs = body.get(key) orelse return cx.fail(.request, "{s} is required", .{key});
    if (docs != .array or docs.array.len == 0) return cx.fail(.request, "{s} must be a non-empty list", .{key});
    const query = body.get("query") orelse return cx.refuse("query is required");
    _ = try string(cx, query, "query");
    for (docs.array) |doc| {
        if (tei or doc != .object) {
            _ = try string(cx, doc, key);
        } else {
            _ = try string(cx, doc.get("text") orelse return cx.refuse("documents: text is required"), "documents.text");
        }
    }
    const count = try topN(cx, body, docs.array.len);
    const returned = if (tei and body.field("return_text") != null) try flag(cx, body, "return_text", false) else try flag(cx, body, "return_documents", false);
    const raw = if (tei) try flag(cx, body, "raw_scores", false) else false;
    if (cohere and body.has("raw_scores")) return cx.refuse("raw_scores is only supported by the TEI texts shape");
    const truncate = if (tei) try flag(cx, body, "truncate", false) else true;
    if (body.field("truncation_direction")) |direction| {
        if (direction != .string or !(std.mem.eql(u8, direction.string, "Right") or std.mem.eql(u8, direction.string, "right")))
            return cx.refuse("truncation_direction: only Right is supported");
    }
    const normalized = try request(cx, body, query, docs, raw);
    const ranked = try rerank.replyOptions(srv, cx, normalized, gone, truncate);
    const rows = ranked.get("results").?.array;
    const translated = try cx.a.alloc(Value, rows.len);
    for (rows, translated) |row, *item| {
        const out = try json.newObject(cx.a);
        try out.put(cx.a, "index", row.get("index").?);
        const probability = row.get("relevance_score").?;
        const value: Value = if (raw) blk: {
            const meta = row.get("meta_info").?;
            const difference = number(meta.get("logit_yes").?) - number(meta.get("logit_no").?);
            if (!std.math.isFinite(difference)) return cx.fail(.server, "rerank inference failed", .{});
            break :blk .{ .float = difference };
        } else probability;
        try out.put(cx.a, if (tei) "score" else "relevance_score", value);
        if (returned) {
            const document = docs.array[@intCast(row.get("index").?.int64().?)];
            if (tei) try out.put(cx.a, "text", document) else {
                const wrapped = if (document == .object) document else blk: {
                    const d = try json.newObject(cx.a);
                    try d.put(cx.a, "text", document);
                    break :blk Value{ .object = d };
                };
                try out.put(cx.a, "document", wrapped);
            }
        }
        item.* = .{ .object = out };
    }
    if (raw) std.sort.insertion(Value, translated, {}, struct {
        fn before(_: void, x: Value, y: Value) bool {
            return number(x.get("score").?) > number(y.get("score").?);
        }
    }.before);
    if (tei) return .{ .array = translated[0..count] };
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "id", .{ .string = try responseId(srv, cx, "rerank") });
    try out.put(cx.a, "results", .{ .array = translated[0..count] });
    const version = try json.newObject(cx.a);
    try version.put(cx.a, "version", .{ .string = "2" });
    const meta = try json.newObject(cx.a);
    try meta.put(cx.a, "api_version", .{ .object = version });
    try out.put(cx.a, "meta", .{ .object = meta });
    return .{ .object = out };
}

fn string(cx: *Cx, value: Value, name: []const u8) errors.Refused![]const u8 {
    if (value != .string or strip(value.string).len == 0) return cx.fail(.request, "{s} must be a non-empty string", .{name});
    return value.string;
}
fn flag(cx: *Cx, body: Value, name: []const u8, default: bool) errors.Refused!bool {
    const value = body.field(name) orelse return default; // TEI's truncate is nullable: null keeps the default
    return if (value == .bool) value.bool else cx.fail(.request, "{s} must be a boolean", .{name});
}
fn topN(cx: *Cx, body: Value, maximum: usize) errors.Refused!usize {
    const value = body.field("top_n") orelse return maximum;
    if (value != .int or value.int[0] == '-') return cx.refuse("top_n must be a positive integer");
    // a positive integer past usize still asks for every result, as rerank.zig reads it
    const n = std.fmt.parseInt(usize, value.int, 10) catch maximum;
    if (n == 0) return cx.refuse("top_n must be a positive integer");
    return @min(maximum, n);
}
fn number(v: Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.int64().?);
}
fn responseId(srv: *Server, cx: *Cx, prefix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(cx.a, "{s}-{d}", .{ prefix, srv.next_id.fetchAdd(1, .monotonic) });
}
/// rerank.zig's body for one query; document text is returned from the request itself, so rerank.zig sends none.
fn request(cx: *Cx, body: Value, query: Value, documents: Value, raw: bool) errors.Refused!Value {
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "query", query);
    try out.put(cx.a, "documents", documents);
    try out.put(cx.a, "return_documents", .{ .bool = false });
    try out.put(cx.a, "return_logits", .{ .bool = raw });
    for ([_][]const u8{ "instruction", "instruct", "max_length" }) |key| if (body.get(key)) |value| try out.put(cx.a, key, value);
    if (body.field("truncate_prompt_tokens")) |value| {
        if (body.field("max_length") != null) return cx.refuse("max_length and truncate_prompt_tokens cannot both be set");
        if (value != .int or value.int64() == null or value.int64().? <= 0) return cx.refuse("truncate_prompt_tokens must be a positive integer");
        try out.put(cx.a, "max_length", value);
    }
    return .{ .object = out };
}
fn strings(cx: *Cx, body: Value, key: []const u8) errors.Refused![]const Value {
    const value = body.get(key) orelse return cx.fail(.request, "{s} is required", .{key});
    if (value == .string) {
        _ = try string(cx, value, key);
        return cx.a.dupe(Value, &.{value});
    }
    if (value != .array or value.array.len == 0) return cx.fail(.request, "{s} must be a string or non-empty list", .{key});
    for (value.array) |v| _ = try string(cx, v, key);
    return value.array;
}
/// vLLM's pair scores: one query's documents as one rerank.zig request, else one a pair, every pair checked first.
fn scores(srv: *Server, cx: *Cx, body: Value, gone: anytype) rerank.Failure!Value {
    for ([_][]const u8{ "top_n", "return_documents", "return_text", "raw_scores" }) |key| if (body.has(key))
        return cx.fail(.request, "{s} is only supported by rerank requests", .{key});
    const left = try strings(cx, body, "text_1");
    const right = try strings(cx, body, "text_2");
    if (left.len != right.len and left.len != 1 and right.len != 1) return cx.refuse("text_1 and text_2 lists must have equal lengths or one item to broadcast");
    const n = @max(left.len, right.len);
    const truncate = body.field("truncate_prompt_tokens") != null;
    const data = try cx.a.alloc(Value, n);
    var tokens: i64 = 0;
    if (left.len == 1) {
        const documents = try cx.a.dupe(Value, right);
        const normalized = try request(cx, body, left[0], .{ .array = documents }, false);
        const scored = try rerank.replyOptions(srv, cx, normalized, gone, truncate);
        tokens = scored.get("usage").?.get("total_tokens").?.int64().?;
        for (scored.get("results").?.array) |row| {
            const i: usize = @intCast(row.get("index").?.int64().?);
            data[i] = try scoreItem(cx, i, row.get("relevance_score").?);
        }
    } else {
        const asks = try cx.a.alloc(Value, n);
        for (asks, 0..) |*ask, i| {
            if (gone.check()) return error.Cancelled;
            const doc = right[if (right.len == 1) 0 else i];
            ask.* = try request(cx, body, left[i], .{ .array = try cx.a.dupe(Value, &.{doc}) }, false);
            try rerank.validate(srv, cx, ask.*, truncate);
        }
        for (asks, data, 0..) |ask, *item, i| {
            const scored = try rerank.replyOptions(srv, cx, ask, gone, truncate);
            tokens += scored.get("usage").?.get("total_tokens").?.int64().?;
            item.* = try scoreItem(cx, i, scored.get("results").?.array[0].get("relevance_score").?);
        }
    }
    const usage = try json.newObject(cx.a);
    try usage.put(cx.a, "prompt_tokens", try json.intValue(cx.a, tokens));
    try usage.put(cx.a, "completion_tokens", try json.intValue(cx.a, 0));
    try usage.put(cx.a, "total_tokens", try json.intValue(cx.a, tokens));
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "id", .{ .string = try responseId(srv, cx, "score") });
    try out.put(cx.a, "object", .{ .string = "list" });
    try out.put(cx.a, "created", try json.intValue(cx.a, std.Io.Clock.real.now(srv.io).toSeconds()));
    try out.put(cx.a, "model", .{ .string = srv.config.served_name });
    try out.put(cx.a, "data", .{ .array = data });
    try out.put(cx.a, "usage", .{ .object = usage });
    return .{ .object = out };
}

fn scoreItem(cx: *Cx, index: usize, score: Value) errors.Refused!Value {
    const row = try json.newObject(cx.a);
    try row.put(cx.a, "index", try json.intValue(cx.a, index));
    try row.put(cx.a, "object", .{ .string = "score" });
    try row.put(cx.a, "score", score);
    return .{ .object = row };
}
