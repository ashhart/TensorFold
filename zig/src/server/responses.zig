//! OpenAI's Responses API: each request runs as its chat completion, translated item by item and event by event.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const sse = @import("sse.zig");
const pyrepr = @import("pyrepr.zig");
const ids = @import("ids.zig");
const translate_mod = @import("responses_translate.zig");
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

const Entry = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    response: Value,
    parent: ?[]const u8,
    messages: []Value,
    size: usize,
};

/// The newest stored responses, in memory, for GET, DELETE and ``previous_response_id``.
pub const Store = struct {
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayList(*Entry) = .empty, // oldest first
    bytes: usize = 0,
    limit: usize = 1024,
    max_bytes: usize = 256 * 1024 * 1024,

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| free(s.gpa, e);
        s.entries.deinit(s.gpa);
    }

    fn free(gpa: Allocator, e: *Entry) void {
        var arena = e.arena;
        arena.deinit();
        gpa.destroy(e);
    }

    fn find(s: *Store, id: []const u8) ?usize {
        for (s.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.id, id)) return i;
        return null;
    }

    /// Keeps a finished response with the messages it added; the oldest go past 1,024 entries or 256 MiB.
    pub fn put(s: *Store, io: std.Io, cx: *Cx, response: Value, added: []const Value) !void {
        const e = try s.gpa.create(Entry);
        e.arena = .init(s.gpa);
        const a = e.arena.allocator();
        e.response = try json.deepCopy(a, response);
        e.id = e.response.get("id").?.string;
        const parent = e.response.get("previous_response_id");
        e.parent = if (parent != null and parent.? == .string) parent.?.string else null;
        const output = response.get("output").?.array;
        const from_output = try translate_mod.messages(cx, output);
        const all = try std.mem.concat(cx.a, Value, &.{ added, from_output });
        e.messages = (try json.deepCopy(a, .{ .array = all })).array;
        e.size = (try json.stringify(cx.a, e.response, .{})).len + (try json.stringify(cx.a, .{ .array = e.messages }, .{})).len;
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        if (s.find(e.id)) |i| {
            const old = s.entries.orderedRemove(i);
            s.bytes -= old.size;
            free(s.gpa, old);
        }
        try s.entries.append(s.gpa, e);
        s.bytes += e.size;
        while (s.entries.items.len > 0 and (s.entries.items.len > s.limit or s.bytes > s.max_bytes)) {
            const old = s.entries.orderedRemove(0);
            s.bytes -= old.size;
            free(s.gpa, old);
        }
    }

    pub fn get(s: *Store, io: std.Io, a: Allocator, id: []const u8) !?Value {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return null;
        return try json.deepCopy(a, s.entries.items[i].response);
    }

    pub fn delete(s: *Store, io: std.Io, id: []const u8) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return false;
        const old = s.entries.orderedRemove(i);
        s.bytes -= old.size;
        free(s.gpa, old);
        return true;
    }
};

/// Every message of a stored response's chain, oldest first (the chain stays newest in the store).
const History = struct {
    srv: *Server,

    pub fn conversation(h: History, cx: *Cx, rid: Value) errors.Refused![]Value {
        if (rid != .string) return cx.refuse("previous_response_id must be a string");
        const s = &h.srv.store;
        s.mutex.lockUncancelable(h.srv.io);
        defer s.mutex.unlock(h.srv.io);
        var chain: std.ArrayList([]Value) = .empty;
        var at: ?[]const u8 = rid.string;
        while (at) |id| {
            const i = s.find(id) orelse return cx.fail(.request, "previous response {s} is not stored here (store: false, deleted, or evicted): send the conversation's items instead", .{try pyrepr.repr(cx.a, .{ .string = id })});
            const e = s.entries.orderedRemove(i);
            s.entries.append(s.gpa, e) catch {};
            try chain.append(cx.a, e.messages);
            at = e.parent;
        }
        var out: std.ArrayList(Value) = .empty;
        var i = chain.items.len;
        while (i > 0) {
            i -= 1;
            for (chain.items[i]) |m| try out.append(cx.a, try json.deepCopy(cx.a, m));
        }
        return out.items;
    }
};

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
    const request = blk: {
        const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
            .ok => |b| b,
            .refused => |m| return refuse(conn, a, m, 400, null),
        };
        const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
            .ok => |v| v,
            .err => return refuse(conn, a, "the request body is not JSON", 400, null),
        };
        const r = translate_mod.translate(&cx, body, History{ .srv = srv }) catch return refuse(conn, a, cx.message, 400, null);
        break :blk .{ r, body };
    };
    const t = request[0];
    const base = json.newObject(a) catch return;
    base.put(a, "id", .{ .string = ids.make(a, "resp_", 32) catch return }) catch return;
    base.put(a, "object", .{ .string = "response" }) catch return;
    base.put(a, "created_at", json.intValue(a, std.Io.Clock.real.now(srv.io).toSeconds()) catch return) catch return;
    base.put(a, "status", .{ .string = "in_progress" }) catch return;
    base.put(a, "error", .null) catch return;
    base.put(a, "incomplete_details", .null) catch return;
    base.put(a, "model", .{ .string = openai.replyModel(srv, request[1]) }) catch return;
    base.put(a, "output", .{ .array = &.{} }) catch return;
    base.put(a, "usage", .null) catch return;
    for (t.echo.keys(), t.echo.values()) |k, v| base.put(a, k, v) catch return;
    var reply: Reply = .{ .srv = srv, .a = a, .conn = conn, .base = base, .streaming = t.stream, .keep = t.store, .added = t.added, .cx = &cx };
    var out: Wire = .{ .reply = &reply };
    openai.run(srv, a, out.out(), .{ .conn = conn }, true, t.chat);
    if (out.opened or conn.broken) return;
    const status = out.status orelse return;
    if (status != 200) return routes.sendValue(conn, a, status, out.payload);
    const final = reply.completion(out.payload) catch return;
    routes.sendValue(conn, a, 200, final);
}

/// The chat handler's output, translated as Python's Wire reads it.
const Wire = struct {
    reply: *Reply,
    opened: bool = false,
    status: ?u16 = null,
    payload: Value = .null,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = whole } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.opened = true;
        try sse.open(w.reply.conn);
        try w.reply.start();
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        return w.reply.chunk(payload);
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
    conn: *Conn,
    base: *json.Object,
    streaming: bool,
    keep: bool,
    added: []Value,
    cx: *Cx,
    items: std.ArrayList(*json.Object) = .empty,
    current: ?*json.Object = null,
    calls: std.AutoArrayHashMapUnmanaged(i64, *json.Object) = .empty,
    seq: u64 = 0,
    final: ?Value = null,

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
        return sse.event(r.conn, r.a, kind, .{ .object = o });
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
            try r.put(item, "id", .{ .string = ids.make(r.a, "fc_", 32) catch return error.Closed });
            try r.put(item, "type", .{ .string = "function_call" });
            try r.put(item, "call_id", .{ .string = call_id });
            try r.put(item, "name", .{ .string = name });
            try r.put(item, "arguments", .{ .string = "" });
            try r.put(item, "status", .{ .string = "in_progress" });
        }
        r.items.append(r.a, item) catch return error.Closed;
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
        const where_id = item.get("id").?;
        const where_index = try r.int(r.index(item));
        if (std.mem.eql(u8, item.get("type").?.string, "function_call")) {
            try r.event("response.function_call_arguments.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "name", item.get("name").? }, .{ "arguments", item.get("arguments").? } });
        } else {
            const part = item.get("content").?.array[0].object;
            if (std.mem.eql(u8, item.get("type").?.string, "message")) {
                try r.event("response.output_text.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "text", part.get("text").? }, .{ "logprobs", .{ .array = &.{} } } });
            } else {
                try r.event("response.reasoning_text.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "text", part.get("text").? } });
            }
            try r.event("response.content_part.done", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "part", .{ .object = part } } });
        }
        try r.event("response.output_item.done", &.{ .{ "output_index", where_index }, .{ "item", .{ .object = item } } });
    }

    fn write(r: *Reply, kind: []const u8, text: []const u8) E!void {
        const reuse = if (r.current) |c| std.mem.eql(u8, c.get("type").?.string, kind) else false;
        const item = if (reuse) r.current.? else try r.open(kind, "", "");
        const part = item.get("content").?.array[0].object;
        try r.put(part, "text", .{ .string = std.mem.concat(r.a, u8, &.{ part.get("text").?.string, text }) catch return error.Closed });
        const where_id = item.get("id").?;
        const where_index = try r.int(r.index(item));
        if (std.mem.eql(u8, kind, "message")) {
            try r.event("response.output_text.delta", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = text } }, .{ "logprobs", .{ .array = &.{} } } });
        } else try r.event("response.reasoning_text.delta", &.{ .{ "item_id", where_id }, .{ "output_index", where_index }, .{ "content_index", .{ .int = "0" } }, .{ "delta", .{ .string = text } } });
    }

    fn textOf(v: ?Value) ?[]const u8 {
        const x = v orelse return null;
        return if (x == .string and x.string.len > 0) x.string else null;
    }

    fn delta(r: *Reply, d: Value) E!void {
        if (d != .object) return;
        const thought = textOf(d.get("reasoning_content")) orelse textOf(d.get("reasoning"));
        if (thought) |t| try r.write("reasoning", t);
        if (textOf(d.get("content"))) |c| try r.write("message", c);
        const calls = d.get("tool_calls") orelse return;
        if (calls != .array) return;
        for (calls.array) |call| {
            const function = call.get("function") orelse Value.null;
            const at: i64 = if (call.get("index")) |ix| ix.int64() orelse 0 else 0;
            const item = r.calls.get(at) orelse blk: {
                const id = textOf(call.get("id")) orelse (ids.make(r.a, "call_", 32) catch return error.Closed);
                const name = textOf(function.get("name")) orelse "";
                const made = try r.open("function_call", id, name);
                r.calls.put(r.a, at, made) catch return error.Closed;
                break :blk made;
            };
            if (textOf(function.get("arguments"))) |args| {
                try r.put(item, "arguments", .{ .string = std.mem.concat(r.a, u8, &.{ item.get("arguments").?.string, args }) catch return error.Closed });
                try r.event("response.function_call_arguments.delta", &.{ .{ "item_id", item.get("id").? }, .{ "output_index", try r.int(r.index(item)) }, .{ "delta", .{ .string = args } } });
            }
        }
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
        const incomplete = reason != null and std.mem.eql(u8, reason.?, "length");
        const status = if (incomplete) "incomplete" else "completed";
        if (!incomplete) {
            const all_reasoning = for (r.items.items) |item| {
                if (!std.mem.eql(u8, item.get("type").?.string, "reasoning")) break false;
            } else true;
            if (all_reasoning) _ = try r.open("message", "", ""); // an empty answer is still a message
        }
        try r.close(status);
        const final = json.copyObject(r.a, r.base) catch return error.Closed;
        try r.put(final, "status", .{ .string = status });
        try r.put(final, "output", try r.itemsValue());
        try r.put(final, "usage", try r.usage(chat_usage));
        if (incomplete) {
            const d = try r.obj();
            try r.put(d, "reason", .{ .string = "max_output_tokens" });
            try r.put(final, "incomplete_details", .{ .object = d });
        } else try r.put(final, "incomplete_details", .null);
        if (!incomplete) try r.put(final, "completed_at", try r.int(std.Io.Clock.real.now(r.srv.io).toSeconds()));
        if (stats) |s| if (s != .null) try r.put(final, "tensorfold", s);
        r.final = .{ .object = final };
        if (r.keep) r.srv.store.put(r.srv.io, r.cx, .{ .object = final }, r.added) catch {};
        try r.event(if (incomplete) "response.incomplete" else "response.completed", &.{.{ "response", .{ .object = final } }});
        return .{ .object = final };
    }

    fn fail(r: *Reply, problem: Value) E!void {
        try r.close("incomplete");
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
        try r.delta(if (choice.get("delta")) |d| d else .null);
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
        try r.delta(.{ .object = d });
        const reason = choice.get("finish_reason");
        return r.finish(if (reason != null and reason.? == .string) reason.?.string else null, data.get("usage"), data.get("tensorfold"));
    }
};
