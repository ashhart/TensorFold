//! OpenAI's Responses API: each request runs as its chat completion, translated item by item and event by event.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const pyrepr = @import("pyrepr.zig");
const ids = @import("ids.zig");
const translate_mod = @import("responses_translate.zig");
const store_mod = @import("responses_store.zig");
const cancel_ids = @import("cancel_ids.zig");
const Sink = @import("responses_sink.zig").Sink;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// A ``/v1/responses`` path's response id ("" for the collection), else null.
pub fn route(path: []const u8) ?[]const u8 {
    for ([_][]const u8{ "/v1/responses", "/responses" }) |prefix| {
        if (std.mem.eql(u8, path, prefix)) return "";
        if (path.len > prefix.len + 1 and std.mem.startsWith(u8, path, prefix) and path[prefix.len] == '/') {
            const rest = path[prefix.len + 1 ..];
            if (std.mem.indexOfScalar(u8, rest, '/') == null) return rest;
        }
    }
    return null;
}

/// The store lives in its own module; the server keeps it by this name.
pub const Store = store_mod.Store;

fn refuse(conn: *Conn, a: Allocator, message: []const u8, status: u16, param: ?[]const u8) void {
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"invalid_request_error\", \"param\": {s}, \"code\": null}}}}", .{ json.quote(a, message, .{}) catch return, if (param) |p| json.quote(a, p, .{}) catch return else "null" }) catch return;
    conn.sendJson(status, body);
}

pub fn get(srv: *Server, conn: *Conn, a: Allocator, rid: []const u8) void {
    const found = srv.store.get(srv.io, a, rid) catch null;
    if (found) |v| return routes.sendValue(conn, a, 200, v);
    refuse(conn, a, std.fmt.allocPrint(a, "no stored response has id {s}", .{pyrepr.repr(a, .{ .string = rid }) catch rid}) catch return, 404, "response_id");
}

pub fn delete(srv: *Server, conn: *Conn, a: Allocator, rid: ?[]const u8) void {
    const id = rid orelse "";
    if (rid == null or id.len == 0 or !srv.store.delete(srv.io, id)) {
        const shown = if (rid) |r| pyrepr.repr(a, .{ .string = r }) catch r else "None";
        return refuse(conn, a, std.fmt.allocPrint(a, "no stored response has id {s}", .{shown}) catch return, 404, "response_id");
    }
    const body = std.fmt.allocPrint(a, "{{\"id\": {s}, \"object\": \"response\", \"deleted\": true}}", .{json.quote(a, id, .{}) catch return}) catch return;
    conn.sendJson(200, body);
}

/// POST /v1/responses.
pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |m| return refuse(conn, a, m, 400, null),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => return refuse(conn, a, "the request body is not JSON", 400, null),
    };
    // a restricted key is stateless here: checked before any stored conversation is read
    if (@import("auth_scope.zig").responseDenial(conn.key_scopes, body)) |why|
        return @import("auth_scope_reply.zig").deny(conn, false, why);
    const history: store_mod.History = .{ .store = &srv.store, .io = srv.io };
    const t = translate_mod.translate(&cx, body, history) catch return refuse(conn, a, cx.message, 400, null);
    switch (turn(srv, a, conn, &cx, body, t, @import("responses_sink.zig").sse(conn), .{})) {
        .streamed => {},
        .refused => |r| routes.sendValue(conn, a, r.status, r.payload),
        .whole => |final| routes.sendValue(conn, a, 200, final),
    }
}

/// How a turn ended, for its transport to answer.
pub const Outcome = union(enum) {
    /// The final response object sent as events (null: the client left first, or nothing could be sent).
    streamed: ?Value,
    /// The chat route's whole error reply, before any event.
    refused: struct { status: u16, payload: Value },
    /// A final response that was not streamed (HTTP only).
    whole: Value,
};

/// What a transport adds to a turn beyond its translated request.
pub const TurnOptions = struct {
    /// The parent's whole conversation when it came from outside the store; the store then keeps the turn whole.
    whole_history: ?[]Value = null,
};

/// One translated Responses request run as its chat completion, its events sent through ``sink``.
pub fn turn(srv: *Server, a: Allocator, conn: *Conn, cx: *Cx, body: Value, t: translate_mod.Request, sink: Sink, opts: TurnOptions) Outcome {
    const base = newBase(srv, a, body, t) catch return .{ .streamed = null };
    if (t.rows_if_offered and !(@import("logprobs.zig").offered(srv, a) catch false)) _ = t.chat.object.orderedRemove("logprobs");
    // a reply with tools gets its rows once its calls are parsed: each item's done events wait for them
    const late = json.truthyField(t.chat, "logprobs") and json.truthyField(t.chat, "tools");
    var reply: Reply = .{ .srv = srv, .a = a, .sink = sink, .base = base, .streaming = t.stream, .keep = t.store, .added = t.added, .cx = cx, .call_ids = .{ .key = &srv.tool_ids, .body = t.chat }, .whole_history = opts.whole_history, .late_rows = late };
    var out: Wire = .{ .reply = &reply };
    cancel_ids.alias = base.get("id").?.string; // the cancel route takes the response's id too
    defer cancel_ids.alias = null;
    openai.runNamed(srv, a, out.out(), .{ .conn = conn }, true, t.chat, t.format, true);
    if (out.opened or conn.broken) return .{ .streamed = reply.final };
    const status = out.status orelse return .{ .streamed = null };
    if (status != 200) return .{ .refused = .{ .status = status, .payload = out.payload } };
    return .{ .whole = reply.completion(out.payload) catch return .{ .streamed = null } };
}

/// A ``generate: false`` turn: created, in progress, then completed with no output and ``prompt_tokens`` of input.
pub fn warmup(srv: *Server, a: Allocator, cx: *Cx, body: Value, t: translate_mod.Request, sink: Sink, prompt_tokens: usize, opts: TurnOptions) ?Value {
    const base = newBase(srv, a, body, t) catch return null;
    var reply: Reply = .{ .srv = srv, .a = a, .sink = sink, .base = base, .streaming = true, .keep = t.store, .added = t.added, .cx = cx, .call_ids = .{ .key = &srv.tool_ids, .body = t.chat }, .whole_history = opts.whole_history, .warmup = true };
    const usage = json.newObject(a) catch return null;
    usage.put(a, "prompt_tokens", json.intValue(a, prompt_tokens) catch return null) catch return null;
    usage.put(a, "completion_tokens", .{ .int = "0" }) catch return null;
    sink.vt.open(sink.ctx) catch return null;
    reply.start() catch return null;
    return reply.finish("stop", .{ .object = usage }, null) catch null;
}

/// The response object a turn starts from: in progress, with no output yet and the request's echoed fields.
fn newBase(srv: *Server, a: Allocator, body: Value, t: translate_mod.Request) !*json.Object {
    const base = try json.newObject(a);
    try base.put(a, "id", .{ .string = try ids.make(a, "resp_", 32) });
    try base.put(a, "object", .{ .string = "response" });
    try base.put(a, "created_at", try json.intValue(a, std.Io.Clock.real.now(srv.io).toSeconds()));
    try base.put(a, "status", .{ .string = "in_progress" });
    try base.put(a, "error", .null);
    try base.put(a, "incomplete_details", .null);
    try base.put(a, "model", .{ .string = openai.replyModel(srv, body) });
    try base.put(a, "output", .{ .array = &.{} });
    try base.put(a, "usage", .null);
    for (t.echo.keys(), t.echo.values()) |k, v| try base.put(a, k, v);
    return base;
}

/// The chat handler's output, translated as Python's Wire reads it.
const Wire = struct {
    reply: *Reply,
    opened: bool = false,
    status: ?u16 = null,
    payload: Value = .null,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = whole, .keepalive = keepalive } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.opened = true;
        const sink = w.reply.sink;
        try sink.vt.open(sink.ctx);
        try w.reply.start();
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        return w.reply.chunk(payload);
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        const sink = w.reply.sink;
        try sink.vt.keepalive(sink.ctx);
    }

    fn whole(ctx: *anyopaque, status: u16, payload: Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = status;
        w.payload = payload;
    }
};

/// A Response built from chat-completion deltas, in order: reasoning, text, calls; streamed as typed events.
const Reply = struct {
    srv: *Server,
    a: Allocator,
    sink: Sink,
    base: *json.Object,
    streaming: bool,
    keep: bool,
    added: []Value,
    cx: *Cx,
    call_ids: ids.Calls, // the chat run's ids over its translated body, for a call the deltas leave unnamed
    items: std.ArrayList(*json.Object) = .empty,
    current: ?*json.Object = null,
    buffers: std.ArrayList(std.ArrayList(u8)) = .empty,
    calls: std.AutoArrayHashMapUnmanaged(i64, *json.Object) = .empty,
    seq: u64 = 0,
    final: ?Value = null,
    /// The parent's whole conversation from outside the store (null over HTTP): the store keeps this turn whole.
    whole_history: ?[]Value = null,
    /// A warmup turn: no empty message item, so its output stays empty.
    warmup: bool = false,
    /// Logprobs come after the calls are parsed: done events wait until ``finish``, when every item has its rows.
    late_rows: bool = false,
    finishing: bool = false,

    const E = error{Closed};

    fn obj(r: *Reply) E!*json.Object {
        return json.newObject(r.a) catch error.Closed;
    }

    fn put(r: *Reply, o: *json.Object, k: []const u8, v: Value) E!void {
        o.put(r.a, k, v) catch return error.Closed;
    }

    /// One event: ``type``, its sequence number, then its fields, written at once as ``event:`` and ``data:``.
    fn event(r: *Reply, kind: []const u8, fields: []const struct { []const u8, Value }) E!void {
        if (!r.streaming) return;
        const o = try r.obj();
        try r.put(o, "type", .{ .string = kind });
        try r.put(o, "sequence_number", json.intValue(r.a, r.seq) catch return error.Closed);
        r.seq += 1;
        for (fields) |f| try r.put(o, f[0], f[1]);
        return r.sink.vt.event(r.sink.ctx, r.a, kind, o);
    }

    fn int(r: *Reply, n: anytype) E!Value {
        return json.intValue(r.a, n) catch error.Closed;
    }

    fn start(r: *Reply) E!void {
        try r.event("response.created", &.{.{ "response", .{ .object = r.base } }});
        try r.event("response.in_progress", &.{.{ "response", .{ .object = r.base } }});
    }

    fn index(r: *Reply, item: *json.Object) usize {
        for (r.items.items, 0..) |x, i| if (x == item) return i;
        return 0;
    }

    fn open(r: *Reply, kind: []const u8, call_id: []const u8, name: []const u8) E!*json.Object {
        try r.close("completed");
        const item = try r.obj();
        var part: ?*json.Object = null;
        if (std.mem.eql(u8, kind, "message")) {
            try r.put(item, "id", .{ .string = ids.make(r.a, "msg_", 32) catch return error.Closed });
            try r.put(item, "type", .{ .string = "message" });
            try r.put(item, "role", .{ .string = "assistant" });
            try r.put(item, "status", .{ .string = "in_progress" });
            try r.put(item, "content", .{ .array = &.{} });
            part = try r.obj();
            try r.put(part.?, "type", .{ .string = "output_text" });
            try r.put(part.?, "text", .{ .string = "" });
            try r.put(part.?, "annotations", .{ .array = &.{} });
            try r.put(part.?, "logprobs", .{ .array = &.{} });
        } else if (std.mem.eql(u8, kind, "reasoning")) {
            try r.put(item, "id", .{ .string = ids.make(r.a, "rs_", 32) catch return error.Closed });
            try r.put(item, "type", .{ .string = "reasoning" });
            try r.put(item, "summary", .{ .array = &.{} });
            try r.put(item, "content", .{ .array = &.{} });
            try r.put(item, "status", .{ .string = "in_progress" });
            part = try r.obj();
            try r.put(part.?, "type", .{ .string = "reasoning_text" });
            try r.put(part.?, "text", .{ .string = "" });
        } else {
            try r.put(item, "id", .{ .string = ids.derived(r.a, r.call_ids.key, "fc_", 32, call_id) catch return error.Closed });
            try r.put(item, "type", .{ .string = "function_call" });
            try r.put(item, "call_id", .{ .string = call_id });
            try r.put(item, "name", .{ .string = name });
            try r.put(item, "arguments", .{ .string = "" });
            try r.put(item, "status", .{ .string = "in_progress" });
        }
        r.items.append(r.a, item) catch return error.Closed;
        r.buffers.append(r.a, .empty) catch return error.Closed;
        r.current = item;
        try r.event("response.output_item.added", &.{ .{ "output_index", try r.int(r.items.items.len - 1) }, .{ "item", .{ .object = item } } });
        if (part) |p| {
            const list = r.a.alloc(Value, 1) catch return error.Closed;
            list[0] = .{ .object = p };
            try r.put(item, "content", .{ .array = list });
            try r.event("response.content_part.added", &.{ .{ "item_id", item.get("id").? }, .{ "output_index", try r.int(r.index(item)) }, .{ "content_index", .{ .int = "0" } }, .{ "part", .{ .object = p } } });
        }
        return item;
    }

    fn close(r: *Reply, status: []const u8) E!void {
        const item = r.current orelse return;
        r.current = null;
        try r.put(item, "status", .{ .string = status });
        if (r.late_rows and !r.finishing) return;
        const where_id = item.get("id").?;
        const where_index = try r.int(r.index(item));
        if (std.mem.eql(u8, item.get("type").?.string, "function_call")) {
            try r.event("response.function_call_arguments.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "name", item.get("name").? }, .{ "arguments", item.get("arguments").? } });
        } else {
            const part = item.get("content").?.array[0].object;
            if (std.mem.eql(u8, item.get("type").?.string, "message")) {
                try r.event("response.output_text.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "text", part.get("text").? }, .{ "logprobs", part.get("logprobs").? } });
            } else {
                try r.event("response.reasoning_text.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "text", part.get("text").? } });
            }
            try r.event("response.content_part.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "part", .{ .object = part } } });
        }
        try r.event("response.output_item.done", &.{ .{ "output_index", where_index }, .{ "item", .{ .object = item } } });
    }

    /// The open item ends with `status`; with late rows every item sends its deferred done events now, in order.
    fn closeAll(r: *Reply, status: []const u8) E!void {
        if (!r.late_rows) return r.close(status);
        r.finishing = true;
        const current = r.current;
        for (r.items.items) |item| {
            r.current = item;
            try r.close(if (item == current) status else item.get("status").?.string);
        }
    }

    /// Each item's text or arguments grows in one buffer, including calls whose deltas interleave.
    fn append(r: *Reply, item: *json.Object, text: []const u8) E![]const u8 {
        const buffer = &r.buffers.items[r.index(item)];
        buffer.appendSlice(r.a, text) catch return error.Closed;
        return buffer.items;
    }

    /// A text delta, with the entries the chunk carried for it (null: none asked).
    fn write(r: *Reply, kind: []const u8, text: []const u8, rows: ?[]Value) E!void {
        const reuse = if (r.current) |c| std.mem.eql(u8, c.get("type").?.string, kind) else false;
        const item = if (reuse) r.current.? else try r.open(kind, "", "");
        const part = item.get("content").?.array[0].object;
        try r.put(part, "text", .{ .string = try r.append(item, text) });
        const entries: Value = .{ .array = if (rows) |list| try r.carry(part, list) else &.{} };
        const where_id = item.get("id").?;
        const where_index = try r.int(r.index(item));
        if (std.mem.eql(u8, kind, "message")) {
            try r.event("response.output_text.delta", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = text } }, .{ "logprobs", entries } });
        } else if (rows != null) {
            try r.event("response.reasoning_text.delta", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = text } }, .{ "logprobs", entries } });
        } else try r.event("response.reasoning_text.delta", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = text } } });
    }

    /// `list` appended to what `owner` already carries; returns `list`.
    fn carry(r: *Reply, owner: *json.Object, list: []Value) E![]Value {
        const had = if (owner.get("logprobs")) |v| v.array else &.{};
        try r.put(owner, "logprobs", .{ .array = std.mem.concat(r.a, Value, &.{ had, list }) catch return error.Closed });
        return list;
    }

    fn textOf(v: ?Value) ?[]const u8 {
        const x = v orelse return null;
        return if (x == .string and x.string.len > 0) x.string else null;
    }

    /// One chat delta, with its chunk's logprobs (null: none): a channel's entries go with its text, others to items.
    fn delta(r: *Reply, d: Value, rows: ?Value) E!void {
        const thought = if (d == .object) textOf(d.get("reasoning_content")) orelse textOf(d.get("reasoning")) else null;
        const content = if (d == .object) textOf(d.get("content")) else null;
        if (thought) |t| try r.write("reasoning", t, if (rows) |v| listOf(v, "reasoning_content") else null);
        if (content) |c| try r.write("message", c, if (rows) |v| listOf(v, "content") else null);
        if (rows) |v| try r.probabilityItems(v, thought != null, content != null);
        if (d != .object) return;
        const calls = d.get("tool_calls") orelse return;
        if (calls != .array) return;
        for (calls.array) |call| {
            const function = call.get("function") orelse Value.null;
            const at: i64 = if (call.get("index")) |ix| ix.int64() orelse 0 else 0;
            const item = r.calls.get(at) orelse blk: {
                const id = textOf(call.get("id")) orelse (r.call_ids.id(r.a, 0, std.math.cast(u64, at) orelse 0) catch return error.Closed);
                const name = textOf(function.get("name")) orelse "";
                const made = try r.open("function_call", id, name);
                r.calls.put(r.a, at, made) catch return error.Closed;
                break :blk made;
            };
            if (textOf(function.get("arguments"))) |args| {
                try r.put(item, "arguments", .{ .string = try r.append(item, args) });
                try r.event("response.function_call_arguments.delta", &.{ .{ "item_id", item.get("id").? }, .{ "output_index", try r.int(r.index(item)) }, .{ "delta", .{ .string = args } } });
            }
        }
    }

    /// Entries that arrived without their text, and the calls' groups: each to its item, with an empty delta.
    fn probabilityItems(r: *Reply, rows: Value, skip_reasoning: bool, skip_content: bool) E!void {
        const logprobs = @import("response_logprobs.zig");
        var added: std.ArrayList(logprobs.Added) = .empty;
        logprobs.attach(r.a, r.items.items, rows, skip_reasoning, skip_content, &added) catch return error.Closed;
        if (listOf(rows, "tool_calls")) |groups| for (groups) |group| {
            const item = r.calls.get((group.get("index") orelse continue).int64() orelse continue) orelse continue;
            const list = (group.get("content") orelse continue).items() orelse continue;
            added.append(r.a, .{ .item = item, .owner = item, .rows = list }) catch return error.Closed;
        };
        for (added.items) |one| {
            _ = try r.carry(one.owner, one.rows);
            const where = try r.int(r.index(one.item));
            const entries: Value = .{ .array = one.rows };
            if (one.item == one.owner) {
                try r.event("response.function_call_arguments.delta", &.{ .{ "item_id", one.item.get("id").? }, .{ "output_index", where }, .{ "delta", .{ .string = "" } }, .{ "logprobs", entries } });
                continue;
            }
            const name = if (std.mem.eql(u8, one.item.get("type").?.string, "reasoning")) "response.reasoning_text.delta" else "response.output_text.delta";
            try r.event(name, &.{ .{ "item_id", one.item.get("id").? }, .{ "output_index", where }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = "" } }, .{ "logprobs", entries } });
        }
    }

    fn listOf(rows: Value, key: []const u8) ?[]Value {
        return (rows.get(key) orelse return null).items();
    }

    fn usage(r: *Reply, chat: ?Value) E!Value {
        const c = chat orelse return .null;
        if (!c.truthy()) return .null;
        const o = try r.obj();
        const cached = if (c.get("prompt_tokens_details")) |d| (if (d.truthy()) d.get("cached_tokens") orelse Value{ .int = "0" } else Value{ .int = "0" }) else Value{ .int = "0" };
        const thought = if (c.get("completion_tokens_details")) |d| (if (d.truthy()) d.get("reasoning_tokens") orelse Value{ .int = "0" } else Value{ .int = "0" }) else Value{ .int = "0" };
        const prompt = c.get("prompt_tokens") orelse Value{ .int = "0" };
        const generated = c.get("completion_tokens") orelse Value{ .int = "0" };
        try r.put(o, "input_tokens", prompt);
        const in_details = try r.obj();
        try r.put(in_details, "cached_tokens", cached);
        try r.put(o, "input_tokens_details", .{ .object = in_details });
        try r.put(o, "output_tokens", generated);
        const out_details = try r.obj();
        try r.put(out_details, "reasoning_tokens", thought);
        try r.put(o, "output_tokens_details", .{ .object = out_details });
        try r.put(o, "total_tokens", try r.int((prompt.int64() orelse 0) + (generated.int64() orelse 0)));
        return .{ .object = o };
    }

    fn itemsValue(r: *Reply) E!Value {
        const list = r.a.alloc(Value, r.items.items.len) catch return error.Closed;
        for (r.items.items, list) |item, *slot| slot.* = .{ .object = item };
        return .{ .array = list };
    }

    fn finish(r: *Reply, reason: ?[]const u8, chat_usage: ?Value, stats: ?Value) E!Value {
        const length = reason != null and std.mem.eql(u8, reason.?, "length");
        const filtered = reason != null and std.mem.eql(u8, reason.?, @import("loop_stop.zig").finish_reason);
        const cancelled = reason != null and std.mem.eql(u8, reason.?, "cancelled"); // POST /v1/requests/{id}/cancel
        const incomplete = length or filtered;
        const status = if (cancelled) "cancelled" else if (incomplete) "incomplete" else "completed";
        if (!incomplete and !cancelled and !r.warmup) {
            const all_reasoning = for (r.items.items) |item| {
                if (!std.mem.eql(u8, item.get("type").?.string, "reasoning")) break false;
            } else true;
            if (all_reasoning) _ = try r.open("message", "", ""); // an empty answer is still a message
        }
        try r.closeAll(if (cancelled) "incomplete" else status); // an output item is never "cancelled"
        const final = json.copyObject(r.a, r.base) catch return error.Closed;
        try r.put(final, "status", .{ .string = status });
        try r.put(final, "output", try r.itemsValue());
        try r.put(final, "usage", try r.usage(chat_usage));
        if (incomplete) {
            const d = try r.obj();
            try r.put(d, "reason", .{ .string = if (filtered) "content_filter" else "max_output_tokens" });
            try r.put(final, "incomplete_details", .{ .object = d });
        } else try r.put(final, "incomplete_details", .null);
        if (!incomplete and !cancelled) try r.put(final, "completed_at", try r.int(std.Io.Clock.real.now(r.srv.io).toSeconds()));
        if (stats) |s| if (s != .null) try r.put(final, "tensorfold", s);
        r.final = .{ .object = final };
        if (r.keep) r.store(.{ .object = final });
        // a cancelled response ends with the terminal event clients know for a cut reply, its status "cancelled"
        try r.event(if (incomplete or cancelled) "response.incomplete" else "response.completed", &.{.{ "response", .{ .object = final } }});
        return .{ .object = final };
    }

    /// Keeps the final response: chained to its stored parent, or whole when the parent came from elsewhere.
    fn store(r: *Reply, final: Value) void {
        const s = &r.srv.store;
        const earlier = r.whole_history orelse return s.put(r.srv.io, r.cx, final, r.added) catch {};
        const from_output = translate_mod.messages(r.cx, final.get("output").?.array) catch return;
        const all = std.mem.concat(r.cx.a, Value, &.{ earlier, r.added, from_output }) catch return;
        s.putWhole(r.srv.io, r.cx, final, all) catch {};
    }

    fn fail(r: *Reply, problem: Value) E!void {
        try r.closeAll("incomplete");
        const message = if (problem == .object) problem.get("message") else null;
        const kind = if (problem == .object) problem.get("type") else null;
        const is_request = kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "invalid_request_error");
        const e = try r.obj();
        try r.put(e, "code", .{ .string = if (is_request) "invalid_prompt" else "server_error" });
        const text = if (problem == .string) problem.string else if (message != null and message.?.truthy()) (if (message.? == .string) message.?.string else "the reply failed") else "the reply failed";
        try r.put(e, "message", .{ .string = text });
        const final = json.copyObject(r.a, r.base) catch return error.Closed;
        try r.put(final, "status", .{ .string = "failed" });
        try r.put(final, "output", try r.itemsValue());
        try r.put(final, "error", .{ .object = e });
        r.final = .{ .object = final };
        try r.event("response.failed", &.{.{ "response", .{ .object = final } }});
    }

    /// One chat-completion stream event (null: its ``[DONE]``).
    fn chunk(r: *Reply, payload: ?Value) E!void {
        if (r.final != null) return;
        const p = payload orelse return r.fail(.{ .string = "the reply ended early" });
        if (p.has("error")) return r.fail(p.get("error").?);
        const choices = p.get("choices");
        const choice: Value = if (choices != null and choices.? == .array and choices.?.array.len > 0) choices.?.array[0] else .null;
        try r.delta(if (choice.get("delta")) |d| d else .null, choice.get("logprobs"));
        if (choice.get("finish_reason")) |f| if (f.truthy()) {
            _ = try r.finish(if (f == .string) f.string else null, p.get("usage"), p.get("tensorfold"));
        };
    }

    /// A whole chat completion (not streamed).
    fn completion(r: *Reply, data: Value) E!Value {
        const choice = data.get("choices").?.array[0];
        const message = choice.get("message") orelse Value.null;
        const d = try r.obj();
        try r.put(d, "reasoning_content", message.get("reasoning_content") orelse .null);
        try r.put(d, "content", message.get("content") orelse .null);
        var calls: std.ArrayList(Value) = .empty;
        if (message.get("tool_calls")) |tc| if (tc == .array) for (tc.array, 0..) |call, i| {
            const c = try r.obj();
            try r.put(c, "index", try r.int(i));
            if (call == .object) for (call.object.keys(), call.object.values()) |k, v| try r.put(c, k, v);
            calls.append(r.a, .{ .object = c }) catch return error.Closed;
        };
        try r.put(d, "tool_calls", .{ .array = calls.items });
        try r.delta(.{ .object = d }, null);
        if (choice.get("logprobs")) |rows| try r.probabilityItems(rows, false, false);
        const reason = choice.get("finish_reason");
        return r.finish(if (reason != null and reason.? == .string) reason.?.string else null, data.get("usage"), data.get("tensorfold"));
    }
};

test "a call delta that names no id takes the chat run's id at its index, and one that names its id keeps it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const key: ids.Key = .of(ids.test_key_text);
    const chat = (try json.parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}")).ok;
    var cx: Cx = .{ .a = a };
    var reply: Reply = .{ .srv = undefined, .a = a, .sink = undefined, .base = try json.newObject(a), .streaming = false, .keep = false, .added = &.{}, .cx = &cx, .call_ids = .{ .key = &key, .body = chat } };
    try reply.delta((try json.parse(a, "{\"tool_calls\":[{\"index\":1,\"function\":{\"name\":\"f\"}},{\"index\":2,\"id\":\"call_sent\",\"function\":{\"name\":\"f\"}}]}")).ok, null);
    var calls: ids.Calls = .{ .key = &key, .body = chat };
    try std.testing.expectEqualStrings(try calls.id(a, 0, 1), reply.items.items[0].get("call_id").?.string);
    try std.testing.expectEqualStrings("call_sent", reply.items.items[1].get("call_id").?.string);
}

test "Server recovery: Responses grows reasoning, text and interleaved call arguments in bounded buffers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var srv: Server = undefined;
    const key: ids.Key = .of(ids.test_key_text);
    var r: Reply = .{ .srv = &srv, .a = a, .sink = undefined, .base = try json.newObject(a), .streaming = false, .keep = false, .added = &.{}, .cx = &cx, .call_ids = .{ .key = &key, .body = .null } };
    const chunk = "abcdefgh";
    for ([_][]const u8{ "reasoning", "message" }) |kind| {
        for (0..3000) |_| try r.write(kind, chunk, null);
        const text = r.current.?.get("content").?.array[0].get("text").?.string;
        try std.testing.expectEqual(@as(usize, 3000 * chunk.len), text.len);
        for (0..3000) |i| try std.testing.expectEqualStrings(chunk, text[i * chunk.len ..][0..chunk.len]);
    }
    const first = (try json.parse(a, "{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"f\",\"arguments\":\"abcdefgh\"}}]}")).ok;
    const second = (try json.parse(a, "{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"g\",\"arguments\":\"abcdefgh\"}}]}")).ok;
    for (0..3000) |_| {
        try r.delta(first, null);
        try r.delta(second, null);
    }
    for ([_]i64{ 0, 1 }) |i| {
        const args = r.calls.get(i).?.get("arguments").?.string;
        try std.testing.expectEqual(@as(usize, 3000 * chunk.len), args.len);
        for (0..3000) |j| try std.testing.expectEqualStrings(chunk, args[j * chunk.len ..][0..chunk.len]);
    }
    try std.testing.expectEqual(@as(usize, 4), r.items.items.len);
    try std.testing.expect(arena.queryCapacity() < 4 << 20);
}
