//! POST /v1/rerank in #1895's shape, SGLang's request fields too, each document scored by the engine's score hook.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const openai = @import("openai.zig");
const reply_text = @import("reply_text.zig");
const model_text = @import("model_text.zig");
const wait = @import("wait_reason.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Failure = errors.Refused || error{Cancelled};

/// The request's fields once checked; `documents` holds each document's text, `given` the item as the request sent it.
const Ask = struct {
    query: []const u8,
    documents: []const []const u8,
    given: []const Value,
    instruct: ?[]const u8,
    top_n: ?usize,
    /// #1895's cap on a prompt's tokens: a longer prompt's content is cut from its end to fit.
    max_length: usize,
    return_documents: bool,
    return_logits: bool,
    return_ids: bool,
    /// False refuses a prompt past max_length instead of cutting it (the TEI and pair-score shapes' default).
    truncate: bool = true,
};

/// #1895's default cap on a prompt's tokens: the model card's max_length.
pub const max_length_default = 8192;

/// One document's score and the label logits it came from.
const Scored = struct { score: f64, yes: f64, no: f64 };

/// Compact separators and UTF-8 as it is.
pub fn encode(a: Allocator, v: Value) Allocator.Error![]u8 {
    return json.stringify(a, v, .{ .compact = true, .ascii = false }) catch return error.OutOfMemory;
}

/// The server's error envelope, as every route sends it: `{"error": {"message", "type", ...}}`.
pub fn errorBody(cx: *const Cx) Allocator.Error!Value {
    return openai.wrapError(cx.a, try openai.errorBody(cx.a, cx, "documents"));
}

/// The ranked documents, or the refusal; `gone` says the client left (checked before each document's pass).
pub fn reply(srv: *Server, cx: *Cx, body: Value, gone: anytype) Failure!Value {
    return replyOptions(srv, cx, body, gone, true);
}

/// `reply`, but a prompt past max_length is refused when `truncate` is false (score_routes.zig's shapes).
pub fn replyOptions(srv: *Server, cx: *Cx, body: Value, gone: anytype, truncate: bool) Failure!Value {
    const labels = srv.info.rerank orelse return cx.refuse("this model is not a reranker");
    var ask = try parse(cx, body);
    ask.truncate = truncate;
    // each prompt is built in a scratch arena and dropped: a small body of many documents keeps no ids alive
    var scratch = std.heap.ArenaAllocator.init(srv.gpa);
    defer scratch.deinit();
    var frame: ?Frame = null;
    // every prompt checked before any GPU work; the trunk is the prefix all share, each keeping its last token
    var first: []const u32 = &.{};
    var trunk: usize = 0;
    var tokens: usize = 0; // the prompts as scored: a template may render differently on another pass
    for (ask.documents, 0..) |doc, i| {
        if (gone.check()) return error.Cancelled;
        _ = scratch.reset(.retain_capacity);
        const ids = try ready(srv, cx, scratch.allocator(), ask, &frame, doc, i);
        if (i == 0) {
            first = try cx.a.dupe(u32, ids);
            trunk = ids.len - 1;
        } else trunk = @min(trunk, ids.len - 1, common(first, ids));
    }
    const scored = try cx.a.alloc(Scored, ask.documents.len);
    const logits = try cx.a.alloc(f64, 2 * ask.documents.len);
    if (srv.engine.vtable.decide != null) {
        // one decision: the trunk once, each prompt's rest a branch (the request holds no prompt twice)
        const branches = try cx.a.alloc([]const u32, ask.documents.len);
        for (ask.documents, branches, 0..) |doc, *b, i| {
            _ = scratch.reset(.retain_capacity);
            const ids = try ready(srv, cx, scratch.allocator(), ask, &frame, doc, i);
            // a template that reads the clock may render another prompt now: the trunk must still be its prefix
            if (ids.len <= trunk or !std.mem.eql(u32, ids[0..trunk], first[0..trunk]))
                return cx.fail(.request, "document {d}: the chat template rendered another prompt on its second pass", .{i});
            tokens += ids.len;
            b.* = try cx.a.dupe(u32, ids[trunk..]);
        }
        const counts = try cx.a.alloc(usize, ask.documents.len);
        @memset(counts, 2);
        const pairs = try cx.a.alloc(u32, 2 * ask.documents.len);
        for (0..ask.documents.len) |i| pairs[2 * i ..][0..2].* = .{ labels.yes, labels.no };
        if (gone.check()) return error.Cancelled;
        var d: api.Decision = .{ .trunk = first[0..trunk], .branches = branches, .counts = counts, .labels = pairs, .logits = logits, .arena = cx.a };
        srv.engine.decide(&d) catch |err| switch (err) {
            error.Unsupported => return cx.refuse("this engine does not score labels"),
            error.Refused => return cx.refuse(d.problem),
            error.Failed => return failed(srv, cx),
        };
    } else for (ask.documents, 0..) |doc, i| {
        if (gone.check()) return error.Cancelled;
        _ = scratch.reset(.retain_capacity);
        const ids = try ready(srv, cx, scratch.allocator(), ask, &frame, doc, i);
        tokens += ids.len;
        _ = srv.engine.score(ids, &.{ labels.yes, labels.no }, logits[2 * i ..][0..2], false) catch |err| switch (err) {
            error.Unsupported => return cx.refuse("this engine does not score labels"),
            error.Failed => return failed(srv, cx),
        };
    }
    for (scored, 0..) |*row, i| {
        const yes = logits[2 * i];
        const no = logits[2 * i + 1];
        // a NaN or an infinity is no score, and would not even be JSON
        if (!std.math.isFinite(yes) or !std.math.isFinite(no)) return cx.fail(.server, "rerank inference failed", .{});
        row.* = .{ .score = 1.0 / (1.0 + @exp(no - yes)), .yes = yes, .no = no };
    }
    return build(srv, cx, ask, &frame, scored, tokens);
}

/// The prompts `replyOptions` would score, built and checked with no engine work: a request of pairs checks all first.
pub fn validate(srv: *Server, cx: *Cx, body: Value, truncate: bool) errors.Refused!void {
    var ask = try parse(cx, body);
    ask.truncate = truncate;
    var arena: std.heap.ArenaAllocator = .init(srv.gpa);
    defer arena.deinit();
    var frame: ?Frame = null;
    for (ask.documents, 0..) |doc, i| {
        _ = arena.reset(.retain_capacity);
        _ = try ready(srv, cx, arena.allocator(), ask, &frame, doc, i);
    }
}

/// A failed pass: a server error while the engine is healthy, else the engine is unavailable.
fn failed(srv: *Server, cx: *Cx) errors.Refused {
    return if (healthy(srv))
        cx.fail(.server, "rerank inference failed", .{})
    else
        cx.wait(wait.engine_unavailable, "the rerank engine is unavailable", .{});
}

fn healthy(srv: *Server) bool {
    var st: api.Status = .{};
    srv.engine.status(&st, &.{});
    return st.healthy;
}

/// The request (#1895's fields and SGLang's): its schema whole first ("Bad Request"), then the handler's checks.
fn parse(cx: *Cx, body: Value) errors.Refused!Ask {
    if (body != .object) return cx.other("body: input should be a valid dictionary");
    const query = try content(cx, body.get("query") orelse return cx.other("query: field required"), "query");
    const docs = body.get("documents") orelse return cx.other("documents: field required");
    if (docs != .array) return cx.other("documents: input should be a valid list");
    const parts = try cx.a.alloc(Content, docs.array.len);
    for (docs.array, parts) |d, *out| out.* = try content(cx, d, "documents");
    // `instruction` as #1895 names it; SGLang's clients send `instruct`
    const instruct: ?[]const u8 = switch (body.get("instruction") orelse body.get("instruct") orelse Value.null) {
        .null => null,
        .string => |str| if (str.len == 0) null else str,
        else => return cx.other("instruction: input should be a valid string"),
    };
    const top_n: ?usize = switch (body.get("top_n") orelse Value.null) {
        .null => null,
        else => |v| try topN(cx, v),
    };
    const max_length: usize = switch (body.get("max_length") orelse Value.null) {
        .null => max_length_default,
        .int => |d| if (d[0] == '-' or std.mem.eql(u8, d, "0")) return cx.other("max_length: input should be at least 1") else std.fmt.parseInt(usize, d, 10) catch std.math.maxInt(usize),
        else => return cx.other("max_length: input should be a valid integer"),
    };
    const documents = try cx.a.alloc([]const u8, parts.len);
    const ask: Ask = .{
        .query = query.text,
        .documents = documents,
        .given = docs.array,
        .instruct = instruct,
        .top_n = top_n,
        .max_length = max_length,
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
        // a `{"text": ...}` document, as rerank clients send it
        .object => {
            const str = v.strField("text") orelse return cx.other(field ++ ": an object needs its text as a string");
            return .{ .text = str, .string = true, .empty = str.len == 0 };
        },
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
        else => return cx.other(field ++ ": input should be a string, a {\"text\"} object or a list of content parts"),
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

/// The template's tail after the document, from probe documents "0" and "1": a cut keeps it whole (see `ready`).
const Frame = struct { p0: []const u32, p1: []const u32, suffix: usize, tail: []const u8, fixed: bool };

fn frameOf(srv: *Server, cx: *Cx, ask: Ask) errors.Refused!Frame {
    const p0 = try prompt(srv, cx, cx.a, ask, "0");
    const p1 = try prompt(srv, cx, cx.a, ask, "1");
    const both = @min(p0.ids.len, p1.ids.len);
    var suffix: usize = 0;
    while (suffix < both and p0.ids[p0.ids.len - 1 - suffix] == p1.ids[p1.ids.len - 1 - suffix]) suffix += 1;
    // the whole rendered tail, encoded alone, must be exactly the shared ids: no template byte merged into the document
    const alone = srv.text.encode(cx.a, p0.tail, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.fail(.request, "the chat template failed: {s}", .{"the tokenizer cannot encode this prompt"}),
    };
    // a cut needs a tail that starts at an added token (the masked encoder splits only there) and renders alike
    const fixed = suffix > 0 and suffix < both and std.mem.eql(u8, p0.tail, p1.tail) and
        srv.text.isAdded(p0.ids[p0.ids.len - suffix]) and std.mem.eql(u32, alone, p0.ids[p0.ids.len - suffix ..]);
    return .{ .p0 = p0.ids, .p1 = p1.ids, .suffix = suffix, .tail = p0.tail, .fixed = fixed };
}

/// The length of the ids `x` and `y` begin with alike.
fn common(x: []const u32, y: []const u32) usize {
    var n: usize = 0;
    while (n < @min(x.len, y.len) and x[n] == y[n]) n += 1;
    return n;
}

/// Document `i`'s prompt as scored: past max_length its content loses its last tokens, as the card's tokenizer cuts.
fn ready(srv: *Server, cx: *Cx, a: Allocator, ask: Ask, frame: *?Frame, document: []const u8, i: usize) errors.Refused![]const u32 {
    const built = try prompt(srv, cx, a, ask, document);
    if (built.ids.len == 0) return cx.fail(.request, "document {d}: the chat template renders no tokens", .{i});
    var ids = built.ids;
    const cap = ask.max_length;
    if (ids.len > cap) {
        if (!ask.truncate) return cx.fail(.request, "document {d}: the prompt has {d} tokens, past max_length {d}, and truncation is disabled", .{ i, ids.len, cap });
        const f = frame.* orelse try frameOf(srv, cx, ask);
        frame.* = f;
        // a tail this template renders otherwise for this document, a cut leaving no document token: refused
        if (!f.fixed or !std.mem.eql(u8, built.tail, f.tail) or !std.mem.endsWith(u32, ids, f.p0[f.p0.len - f.suffix ..]))
            return cx.fail(.request, "document {d}: the prompt has {d} tokens, past max_length {d}, and this chat template has no fixed tail after the document to cut its content before", .{ i, ids.len, cap });
        // where this document starts: before the first token either probe's document changes
        const start = @min(common(ids, f.p0), common(ids, f.p1));
        if (cap <= start + f.suffix)
            return cx.fail(.request, "document {d}: max_length {d} leaves no room for the document; the prompt without it takes {d} tokens", .{ i, cap, start + f.suffix });
        const cut = try a.alloc(u32, cap);
        @memcpy(cut[0 .. cap - f.suffix], ids[0 .. cap - f.suffix]);
        @memcpy(cut[cap - f.suffix ..], ids[ids.len - f.suffix ..]);
        ids = cut;
    }
    const window = srv.info.context_window;
    if (window > 0 and ids.len > window)
        return cx.fail(.context_length, "document {d}: the prompt has {d} tokens, which does not fit the context length of {d} tokens", .{ i, ids.len, window });
    return ids;
}

/// A prompt's ids and the template text after its last request span (the document's).
const Built = struct { ids: []const u32, tail: []const u8 };

/// The checkpoint template over [system?, query, document] in `a`; the request's text is encoded as plain text.
fn prompt(srv: *Server, cx: *Cx, a: Allocator, ask: Ask, document: []const u8) errors.Refused!Built {
    var messages: std.ArrayList(Value) = .empty;
    if (ask.instruct) |ins| try messages.append(a, try message(a, "system", ins));
    try messages.append(a, try message(a, "query", ask.query));
    try messages.append(a, try message(a, "document", document));
    var problem: []const u8 = "";
    const rendered = srv.text.render(a, .{ .array = messages.items }, .{ .add_generation_prompt = true, .enable_thinking = false }, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{problem}),
    };
    const ids = rendered.encode(srv.text, a, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template failed: {s}", .{"the tokenizer cannot encode this prompt"}),
    };
    const spans = rendered.spans;
    return .{ .ids = ids, .tail = if (spans.len > 0) rendered.text[spans[spans.len - 1][1]..] else rendered.text };
}

fn message(a: Allocator, role: []const u8, words: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = words });
    return .{ .object = o };
}

/// #1895's shape: `results` best first (equal scores in request order, cut to top_n), then `usage`.
fn build(srv: *Server, cx: *Cx, ask: Ask, frame: *?Frame, scored: []const Scored, tokens: usize) errors.Refused!Value {
    const order = try cx.a.alloc(usize, scored.len);
    for (order, 0..) |*o, i| o.* = i;
    std.sort.insertion(usize, order, scored, struct {
        fn before(s: []const Scored, x: usize, y: usize) bool {
            return s[x].score > s[y].score;
        }
    }.before);
    const n = @min(order.len, ask.top_n orelse order.len);
    const results = try cx.a.alloc(Value, n);
    for (order[0..n], results) |i, *item| {
        const o = try json.newObject(cx.a);
        try o.put(cx.a, "index", try json.intValue(cx.a, i));
        try o.put(cx.a, "relevance_score", .{ .float = scored[i].score });
        if (ask.return_documents) try o.put(cx.a, "document", ask.given[i]);
        // the returned documents' ids built again, cut as scored (the same bytes give the same ids): none kept
        const ids = if (ask.return_ids) try ready(srv, cx, cx.a, ask, frame, ask.documents[i], i) else &.{};
        if (try meta(cx.a, ask, scored[i], ids)) |m| try o.put(cx.a, "meta_info", m);
        item.* = .{ .object = o };
    }
    const usage = try json.newObject(cx.a);
    try usage.put(cx.a, "total_tokens", try json.intValue(cx.a, tokens));
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "results", .{ .array = results });
    try out.put(cx.a, "usage", .{ .object = usage });
    return .{ .object = out };
}

/// The opt-in checks' fields (raw label logits, scored ids), or null when the request asked for neither.
fn meta(a: Allocator, ask: Ask, s: Scored, ids: []const u32) Allocator.Error!?Value {
    if (!ask.return_logits and !ask.return_ids) return null;
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
