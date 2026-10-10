//! Anthropic's Messages routes over the chat handler: typed SSE events, tool_use blocks, count_tokens.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const fields = @import("fields.zig");
const messages_mod = @import("messages.zig");
const openai = @import("openai.zig");
const prompt = @import("prompt.zig");
const routes = @import("routes.zig");
const sse = @import("sse.zig");
const tool_specs = @import("tool_specs.zig");
const ids = @import("ids.zig");
const auth = @import("auth.zig");
const translate = @import("anthropic_translate.zig").translate;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub fn route(path: []const u8) bool {
    const p = auth.routePath(path);
    for ([_][]const u8{ "/v1/messages", "/messages", "/v1/messages/count_tokens", "/messages/count_tokens" }) |r| if (std.mem.eql(u8, p, r)) return true;
    return false;
}

/// Anthropic's error object for an HTTP status.
fn errorValue(a: Allocator, message: []const u8, status: u16) Allocator.Error!Value {
    const kind = switch (status) {
        400 => "invalid_request_error",
        404 => "not_found_error",
        503 => "overloaded_error",
        else => "api_error",
    };
    const inner = try json.newObject(a);
    try inner.put(a, "type", .{ .string = kind });
    try inner.put(a, "message", .{ .string = message });
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "error" });
    try o.put(a, "error", .{ .object = inner });
    return .{ .object = o };
}

fn sendError(conn: *Conn, a: Allocator, message: []const u8, status: u16) void {
    routes.sendValue(conn, a, status, errorValue(a, message, status) catch return);
}

/// Rendered like the chat route renders, without generating: the prompt's token count.
fn countTokens(srv: *Server, cx: *Cx, chat: Value) errors.Refused!usize {
    const thinking = try fields.thinkingFields(cx, chat, srv.effort_levels);
    const on = thinking.enable orelse srv.config.enable_thinking;
    const tools = try tool_specs.active(cx, chat.get("tools"), chat.get("tool_choice"));
    const msgs = try messages_mod.normalizeWith(cx, chat.get("messages"), "system", srv.needs_user_after_tool, .of(srv.info.vision));
    const ids_ = prompt.renderIds(srv, cx, msgs, tools, on, srv.effortFor(thinking.effort), true) catch |e| switch (e) {
        error.Refused => return cx.other(cx.message),
        else => |x| return x,
    };
    return ids_.len;
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const count = std.mem.endsWith(u8, auth.routePath(conn.path), "/count_tokens");
    var cx: Cx = .{ .a = a };
    const prepared = blk: {
        const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
            .ok => |b| b,
            .refused => |m| return sendError(conn, a, m, 400),
        };
        const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
            .ok => |v| v,
            .err => |m| return sendError(conn, a, m, 400),
        };
        const chat = translate(&cx, body, count) catch return sendError(conn, a, cx.message, cx.status());
        if (count) {
            const n = countTokens(srv, &cx, chat) catch return sendError(conn, a, cx.message, if (cx.kind == .capacity) 503 else 400);
            const o = json.newObject(a) catch return;
            o.put(a, "input_tokens", json.intValue(a, n) catch return) catch return;
            return routes.sendValue(conn, a, 200, .{ .object = o });
        }
        break :blk .{ chat, body };
    };
    var reply: Reply = .{ .a = a, .conn = conn, .model = openai.replyModel(srv, prepared[1]) };
    reply.init() catch return;
    var wire: Wire = .{ .reply = &reply };
    openai.run(srv, a, wire.out(), .{ .conn = conn }, true, prepared[0]);
    if (wire.opened or wire.status == null or conn.broken) return;
    const payload = wire.payload;
    if (wire.status.? != 200) {
        const problem = payload.get("error") orelse Value.null;
        const message = if (problem == .object) (if (problem.get("message")) |m| (if (m == .string) m.string else "request failed") else "request failed") else (if (problem == .string) problem.string else "request failed");
        return sendError(conn, a, message, wire.status.?);
    }
    const whole = reply.completion(payload) catch |e| {
        const message = std.fmt.allocPrint(a, "invalid model response: {s}", .{reply.problem orelse @errorName(e)}) catch return;
        return sendError(conn, a, message, 500);
    };
    routes.sendValue(conn, a, 200, whole);
}

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

const Call = struct { id: []const u8 = "", name: []const u8 = "", arguments: []const u8 = "", sent: usize = 0, done: bool = false };
const Pending = struct { kind: []const u8, text: []const u8 };

/// Typed Messages events from chat chunks, each delta on a block of its own type.
const Reply = struct {
    a: Allocator,
    conn: *Conn,
    model: []const u8,
    base: *json.Object = undefined,
    index: i64 = -1,
    kind: ?[]const u8 = null,
    calls: std.AutoArrayHashMapUnmanaged(i64, Call) = .empty,
    pending: std.ArrayList(Pending) = .empty,
    active: ?i64 = null,
    finished: bool = false,
    finish: []const u8 = "end_turn",
    stop_sequence: Value = .null,
    tokens: Value = .null,
    problem: ?[]const u8 = null,

    const E = error{Closed};

    fn init(r: *Reply) Allocator.Error!void {
        const a = r.a;
        r.base = try json.newObject(a);
        try r.base.put(a, "id", .{ .string = try ids.make(a, "msg_", 32) });
        try r.base.put(a, "type", .{ .string = "message" });
        try r.base.put(a, "role", .{ .string = "assistant" });
        try r.base.put(a, "model", .{ .string = r.model });
        try r.base.put(a, "content", .{ .array = &.{} });
        try r.base.put(a, "stop_reason", .null);
        try r.base.put(a, "stop_sequence", .null);
        r.tokens = try usage(a, .null);
        try r.base.put(a, "usage", r.tokens);
    }

    fn send(r: *Reply, kind: []const u8, payload: *json.Object) E!void {
        return sse.event(r.conn, r.a, kind, .{ .object = payload });
    }

    fn obj(r: *Reply, kind: []const u8) E!*json.Object {
        const o = json.newObject(r.a) catch return error.Closed;
        o.put(r.a, "type", .{ .string = kind }) catch return error.Closed;
        return o;
    }

    fn put(r: *Reply, o: *json.Object, k: []const u8, v: Value) E!void {
        o.put(r.a, k, v) catch return error.Closed;
    }

    fn start(r: *Reply) E!void {
        const o = try r.obj("message_start");
        try r.put(o, "message", .{ .object = r.base });
        try r.send("message_start", o);
    }

    fn close(r: *Reply) E!void {
        const kind = r.kind orelse return;
        if (std.mem.eql(u8, kind, "thinking")) {
            const d = try r.obj("signature_delta");
            try r.put(d, "signature", .{ .string = "" }); // local reasoning is plaintext: no provider signature
            const o = try r.obj("content_block_delta");
            try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
            try r.put(o, "delta", .{ .object = d });
            try r.send("content_block_delta", o);
        }
        const o = try r.obj("content_block_stop");
        try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.send("content_block_stop", o);
        r.kind = null;
    }

    fn text(r: *Reply, kind: []const u8, value: []const u8) E!void {
        if (value.len == 0) return;
        if (r.active != null) {
            r.pending.append(r.a, .{ .kind = kind, .text = value }) catch return error.Closed;
            return;
        }
        const key = if (std.mem.eql(u8, kind, "thinking")) "thinking" else "text";
        if (r.kind == null or !std.mem.eql(u8, r.kind.?, kind)) {
            try r.close();
            r.index += 1;
            r.kind = kind;
            const block = try r.obj(kind);
            try r.put(block, key, .{ .string = "" });
            const o = try r.obj("content_block_start");
            try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
            try r.put(o, "content_block", .{ .object = block });
            try r.send("content_block_start", o);
        }
        const d = try r.obj(std.mem.concat(r.a, u8, &.{ kind, "_delta" }) catch return error.Closed);
        try r.put(d, key, .{ .string = value });
        const o = try r.obj("content_block_delta");
        try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
        try r.put(o, "delta", .{ .object = d });
        try r.send("content_block_delta", o);
    }

    fn chunk(r: *Reply, payload: ?Value) E!void {
        if (r.finished) return;
        const p = payload orelse {
            try r.drain(true);
            try r.close();
            const d = json.newObject(r.a) catch return error.Closed;
            try r.put(d, "stop_reason", .{ .string = r.finish });
            try r.put(d, "stop_sequence", r.stop_sequence);
            const o = try r.obj("message_delta");
            try r.put(o, "delta", .{ .object = d });
            try r.put(o, "usage", r.tokens);
            try r.send("message_delta", o);
            try r.send("message_stop", try r.obj("message_stop"));
            r.finished = true;
            return;
        };
        if (p.has("error")) {
            try r.close();
            const problem = p.get("error").?;
            const message = if (problem.get("message")) |m| (if (m == .string) m.string else "generation failed") else "generation failed";
            const kind = problem.get("type");
            const status: u16 = if (kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "invalid_request_error")) 400 else 500;
            const e = errorValue(r.a, message, status) catch return error.Closed;
            try r.send("error", e.object);
            r.finished = true;
            return;
        }
        if (p.get("usage")) |u| if (u != .null) {
            r.tokens = usage(r.a, u) catch return error.Closed;
        };
        if (p.get("stop_sequence")) |s| if (s != .null) {
            r.stop_sequence = s;
        };
        if (p.get("choices")) |choices| if (choices == .array) for (choices.array) |choice| {
            if (choice.get("finish_reason")) |f| if (f.truthy()) {
                r.finish = finishOf(json.strOr(f));
            };
            if (r.stop_sequence != .null) r.finish = "stop_sequence";
            const delta = choice.get("delta") orelse Value.null;
            try r.text("thinking", json.strOr(delta.get("reasoning_content")));
            try r.text("text", json.strOr(delta.get("content")));
            if (delta.get("tool_calls")) |tc| if (tc == .array) for (tc.array) |call| {
                const at: i64 = if (call.get("index")) |i| i.int64() orelse 0 else 0;
                const got = r.calls.getOrPut(r.a, at) catch return error.Closed;
                if (!got.found_existing) got.value_ptr.* = .{};
                const target = got.value_ptr;
                if (call.get("id")) |id| if (id.truthy() and id == .string) {
                    target.id = id.string;
                };
                const function = call.get("function") orelse Value.null;
                target.name = std.mem.concat(r.a, u8, &.{ target.name, json.strOr(function.get("name")) }) catch return error.Closed;
                target.arguments = std.mem.concat(r.a, u8, &.{ target.arguments, json.strOr(function.get("arguments")) }) catch return error.Closed;
            };
        };
        try r.drain(false);
    }

    fn drain(r: *Reply, final: bool) E!void {
        for (r.calls.keys(), r.calls.values()) |at, *call| {
            if (call.done) continue;
            if (call.id.len == 0 or call.name.len == 0) return;
            if (r.active == null) {
                try r.close();
                r.index += 1;
                r.kind = "tool_use";
                r.active = at;
                const block = try r.obj("tool_use");
                try r.put(block, "id", .{ .string = call.id });
                try r.put(block, "name", .{ .string = call.name });
                try r.put(block, "input", .{ .object = json.newObject(r.a) catch return error.Closed });
                const o = try r.obj("content_block_start");
                try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
                try r.put(o, "content_block", .{ .object = block });
                try r.send("content_block_start", o);
            }
            if (at != r.active.?) return;
            const delta = call.arguments[call.sent..];
            if (delta.len > 0) {
                const d = try r.obj("input_json_delta");
                try r.put(d, "partial_json", .{ .string = delta });
                const o = try r.obj("content_block_delta");
                try r.put(o, "index", json.intValue(r.a, r.index) catch return error.Closed);
                try r.put(o, "delta", .{ .object = d });
                try r.send("content_block_delta", o);
                call.sent = call.arguments.len;
            }
            var complete = false;
            if (std.mem.endsWith(u8, std.mem.trimEnd(u8, call.arguments, " \t\r\n"), "}")) {
                if (json.parseText(r.a, call.arguments) catch null) |res| complete = res == .ok and res.ok == .object;
            }
            if (!complete and !final) return;
            try r.close();
            call.done = true;
            r.active = null;
            const pending = r.pending.items;
            r.pending = .empty;
            for (pending) |p| try r.text(p.kind, p.text);
        }
    }

    /// A whole chat completion as one Messages reply.
    fn completion(r: *Reply, data: Value) !Value {
        const a = r.a;
        const choice = data.get("choices").?.array[0];
        const message = choice.get("message").?;
        var content: std.ArrayList(Value) = .empty;
        if (message.get("reasoning_content")) |t| if (t.truthy()) {
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "thinking" });
            try b.put(a, "thinking", t);
            try b.put(a, "signature", .{ .string = "" });
            try content.append(a, .{ .object = b });
        };
        if (message.get("content")) |t| if (t.truthy()) {
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "text" });
            try b.put(a, "text", t);
            try content.append(a, .{ .object = b });
        };
        if (message.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| {
            const function = call.get("function").?;
            const raw = function.get("arguments").?;
            const args = if (raw == .string) switch (try json.parseText(a, raw.string)) {
                .ok => |v| v,
                .err => |m| {
                    r.problem = m;
                    return error.InvalidResponse;
                },
            } else raw;
            if (args != .object) {
                r.problem = "model tool arguments are not a JSON object";
                return error.InvalidResponse;
            }
            const b = try json.newObject(a);
            try b.put(a, "type", .{ .string = "tool_use" });
            try b.put(a, "id", call.get("id").?);
            try b.put(a, "name", function.get("name").?);
            try b.put(a, "input", args);
            try content.append(a, .{ .object = b });
        };
        const out = try json.copyObject(a, r.base);
        try out.put(a, "content", .{ .array = content.items });
        try out.put(a, "usage", try usage(a, data.get("usage") orelse .null));
        const stop = data.get("stop_sequence") orelse Value.null;
        try out.put(a, "stop_sequence", stop);
        try out.put(a, "stop_reason", .{ .string = if (stop != .null) "stop_sequence" else finishOf(json.strOr(choice.get("finish_reason"))) });
        return .{ .object = out };
    }
};

fn finishOf(reason: []const u8) []const u8 {
    if (std.mem.eql(u8, reason, "tool_calls")) return "tool_use";
    if (std.mem.eql(u8, reason, "length")) return "max_tokens";
    return "end_turn";
}

/// A chat usage as Anthropic's: cached prompt tokens read from the cache, thinking tokens as output detail.
fn usage(a: Allocator, value: Value) Allocator.Error!Value {
    const prompt_tokens = (if (value.get("prompt_tokens")) |p| p.int64() else null) orelse 0;
    const details = value.get("prompt_tokens_details");
    const cached_raw = if (details != null and details.?.truthy()) (if (details.?.get("cached_tokens")) |c| c.int64() orelse 0 else 0) else 0;
    const cached = @min(prompt_tokens, @max(0, cached_raw));
    const o = try json.newObject(a);
    try o.put(a, "input_tokens", try json.intValue(a, prompt_tokens - cached));
    try o.put(a, "output_tokens", value.get("completion_tokens") orelse Value{ .int = "0" });
    try o.put(a, "cache_creation_input_tokens", .{ .int = "0" });
    try o.put(a, "cache_read_input_tokens", try json.intValue(a, cached));
    const out_details = value.get("completion_tokens_details");
    if (out_details != null and out_details.? == .object and out_details.?.has("reasoning_tokens")) {
        const t = try json.newObject(a);
        try t.put(a, "thinking_tokens", out_details.?.get("reasoning_tokens").?);
        try o.put(a, "output_tokens_details", .{ .object = t });
    }
    return .{ .object = o };
}
