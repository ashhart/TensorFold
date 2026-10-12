//! POST /v1/chat/completions and /v1/completions, JSON or SSE, as ``server.http`` answers them.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const chat = @import("chat.zig");
const compact = @import("compact.zig");
const grammar = @import("grammar.zig");
const messages = @import("messages.zig");
const tool_specs = @import("tool_specs.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const request_log = @import("request_log.zig");
const sse = @import("sse.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const warm = @import("warm.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const openai_run = @import("openai_run.zig");
const Run = openai_run.Run;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Where a chat reply goes: the client's socket, or a Responses or Messages translation of it.
pub const Out = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        /// 200 with text/event-stream: the stream's head.
        open: *const fn (ctx: *anyopaque) error{Closed}!void,
        /// One ``data:`` payload; null is ``[DONE]``.
        event: *const fn (ctx: *anyopaque, payload: ?Value) error{Closed}!void,
        /// A whole JSON reply.
        reply: *const fn (ctx: *anyopaque, status: u16, payload: Value) void,
        /// Bytes that keep a silent stream alive (an SSE comment, or the API's ping); null: none.
        keepalive: ?*const fn (ctx: *anyopaque) error{Closed}!void = null,
    };
};

/// POST /v1/chat/completions or /v1/completions, read from the client and answered to it.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, is_chat: bool) void {
    var cx: Cx = .{ .a = a };
    const body = http_body.readJson(conn, &cx) catch {
        const err = errorBody(a, &cx, if (is_chat) "messages" else "prompt") catch return;
        return routes.sendValue(conn, a, cx.status(), wrapError(a, err) catch return);
    };
    var client: ClientOut = .{ .conn = conn, .a = a };
    run(srv, a, client.out(), .{ .conn = conn }, is_chat, body);
}

/// A chat reply written straight to the client.
const ClientOut = struct {
    conn: *Conn,
    a: Allocator,

    fn out(c: *ClientOut) Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.open(c.conn);
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return if (payload) |p| sse.data(c.conn, c.a, p) else sse.done(c.conn);
    }

    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        routes.sendValue(c.conn, c.a, status, payload);
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.comment(c.conn);
    }
};

/// Whether the client has gone, read between rounds (Python's socket_cancellation).
pub const Gone = struct {
    conn: *Conn,
    /// The stream a keepalive goes to once it has been silent `every_ns` (0: never)
    out: ?Out = null,
    every_ns: i96 = 0,
    pub fn check(g: Gone) bool {
        return g.conn.peerGone();
    }

    /// A keepalive if the stream has been silent long enough; a failed write means the client left.
    pub fn beat(g: Gone, silent_ns: i96) error{Closed}!bool {
        const o = g.out orelse return false;
        const send = o.vt.keepalive orelse return false;
        if (g.every_ns <= 0 or silent_ns < g.every_ns) return false;
        try send(o.ctx);
        return true;
    }
};

/// ``error_body``: OpenAI's error object, with param and code where clients key on them.
pub fn errorBody(a: Allocator, cx: *const Cx, param: ?[]const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = cx.message });
    if (cx.kind == .other or cx.kind == .server) return .{ .object = o };
    try o.put(a, "type", .{ .string = "invalid_request_error" });
    if (cx.kind == .context_length) {
        try o.put(a, "param", if (param) |p| .{ .string = p } else .null);
        try o.put(a, "code", .{ .string = "context_length_exceeded" });
    }
    return .{ .object = o };
}

pub fn wrapError(a: Allocator, body: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "error", body);
    return .{ .object = o };
}

/// The id a reply names: the one asked for when this endpoint answers to it, else the served name.
pub fn replyModel(srv: *const Server, body: Value) []const u8 {
    if (body.get("model")) |m| if (m == .string) for (srv.config.model_ids) |known| if (std.mem.eql(u8, known, m.string)) return m.string;
    return srv.config.served_name;
}

pub const Plan = struct {
    input: chat.Input,
    stream: bool,
    separate_usage: bool,
    policy: tool_stream.Policy,
    named: []const u8,
};

/// Everything checked before a reply starts (a refusal here is a 400 with no stream opened).
fn plan(srv: *Server, cx: *Cx, is_chat: bool, raw: Value) errors.Refused!Plan {
    const body = try fields.parseNumbers(cx, raw);
    try fields.validateModalities(cx, body);
    // rows need each token's exact bytes: a tokenizer whose decoding depends on context (null) can't give them
    const exact_bytes = json.truthyField(body, "logprobs") and srv.info.logprobs and (srv.text.tokenBytes(cx.a, 0) catch return error.OutOfMemory) != null;
    const logprobs = try fields.probabilityOptions(cx, body, exact_bytes);
    if (srv.config.request_log) |path| request_log.append(cx.a, path, body);
    var input: chat.Input = .{ .fields = undefined, .body = body };
    if (srv.family) |fam| if (is_chat) {
        // a family renders the client's messages and tools itself (its encoding reads OpenAI's shapes)
        const m = body.get("messages") orelse Value.null;
        if (m != .array or m.array.len == 0) return cx.refuse("messages must be a non-empty list");
        input.messages = m;
        input.tools = try fam.tools(cx, body);
    };
    if (is_chat and srv.family == null) {
        input.messages = try messages.normalize(cx, body.get("messages"), "system", srv.needs_user_after_tool);
        input.tools = try tool_specs.active(cx, body.get("tools"), body.get("tool_choice"));
    } else if (!is_chat) if (body.get("messages")) |m| if (m == .array and m.array.len > 0) {
        input.messages = try messages.normalize(cx, m, "system", srv.needs_user_after_tool);
    };
    if (!is_chat and input.messages.array.len == 0) input.prompt = try legacyPrompt(srv, cx, body.get("prompt"));
    // ``body.get("max_tokens") or body.get("max_completion_tokens")``: both are ints or None by now
    const first = body.get("max_tokens");
    const max = if (first != null and first.?.truthy()) first else body.get("max_completion_tokens");
    input.max_tokens = if (max) |m| switch (m) {
        .int => |t| m.int64() orelse (if (t[0] == '-') @as(i64, -1) else std.math.maxInt(i32)),
        .null => null,
        else => 0,
    } else null;
    if (body.get("temperature")) |t| if (t.truthy()) {
        input.temperature = if (t == .float) t.float else std.fmt.parseFloat(f64, t.int) catch 0;
    };
    const sampling = try json.newObject(cx.a);
    for ([_][]const u8{ "temperature", "top_p", "top_k", "min_p", "seed", "priority", "draft", "thinking_budget", "ignore_eos", "stop" } ++ grammar.fields) |k| if (body.get(k)) |v| try sampling.put(cx.a, k, v);
    try grammar.refusal(cx, body);
    if (srv.family == null and input.tools.len > 0 and try tool_specs.choiceRequiresCall(cx.a, body.get("tool_choice"))) try sampling.put(cx.a, "tool_call_required", .{ .bool = true });
    if (srv.family == null) if (body.field("parallel_tool_calls")) |p| if (p == .bool and !p.bool) try sampling.put(cx.a, "tool_call_single", .{ .bool = true });
    const thinking: fields.Thinking = if (srv.family) |fam| blk: {
        const t = try fam.thinking(cx, body);
        break :blk .{ .enable = t.enable, .effort = t.effort };
    } else try fields.thinkingFields(cx, body, srv.effort_levels);
    if (thinking.effort) |e| try sampling.put(cx.a, "reasoning_effort", .{ .string = e });
    if (thinking.enable) |on| try sampling.put(cx.a, "enable_thinking", .{ .bool = on });
    input.fields = .{ .object = sampling };
    const streamed = if (body.get("stream")) |s| s.truthy() else false;
    if (logprobs != null) {
        // 0.6.6's set: the rows describe the plain reply text, so nothing may reshape or split it
        if (!is_chat) return cx.refuse("logprobs are supported on /v1/chat/completions only");
        var structured = false;
        for (grammar.fields) |k| structured = structured or json.truthyField(body, k);
        if (streamed or input.tools.len > 0 or json.truthyField(body, "stop") or structured or json.truthyField(body, "thinking_budget"))
            return cx.refuse("logprobs support nonstreamed chat without tools, stop strings, structured output or a thinking budget");
    }
    input.logprobs = logprobs;
    var policy: tool_stream.Policy = .{};
    if (body.field("parallel_tool_calls")) |p| {
        if (p != .bool) return cx.refuse("parallel_tool_calls must be a boolean");
        policy.single = !p.bool;
        input.single_call = policy.single;
    }
    const options = body.get("stream_options");
    return .{
        .input = input,
        .stream = streamed,
        .separate_usage = options != null and options.? == .object and json.truthyField(options.?, "include_usage"),
        .policy = policy,
        .named = replyModel(srv, body),
    };
}

/// A completion's prompt as the model reads it: token ids as given, anything else as text.
fn legacyPrompt(srv: *Server, cx: *Cx, prompt: ?Value) errors.Refused!chat.Prompt {
    const p = prompt orelse return .{ .text = "" };
    if (p == .array and p.array.len > 0 and allInts(p.array)) {
        const vocab = srv.text.vocabSize();
        const out = try cx.a.alloc(u32, p.array.len);
        for (p.array, out) |t, *slot| slot.* = try promptId(cx, vocab, t);
        return .{ .ids = out };
    }
    return .{ .text = try promptText(srv, cx, p) };
}

/// A prompt token id, refused unless it is an integer inside the vocabulary.
fn promptId(cx: *Cx, vocab: u32, t: Value) errors.Refused!u32 {
    const n = if (t == .int) t.int64() else null;
    if (n == null or n.? < 0 or n.? >= vocab) return cx.fail(.request, "prompt token ids must be integers in the valid range 0 to {d}", .{@as(i64, vocab) - 1});
    return @intCast(n.?);
}

fn allInts(items: []const Value) bool {
    for (items) |t| if (t != .int and t != .bool) return false;
    return true;
}

fn promptText(srv: *Server, cx: *Cx, p: Value) errors.Refused![]const u8 {
    switch (p) {
        .string => |s| return s,
        .null => return "",
        .array => |items| {
            if (allInts(items)) {
                const vocab = srv.text.vocabSize();
                const toks = try cx.a.alloc(u32, items.len);
                for (items, toks) |t, *slot| slot.* = try promptId(cx, vocab, t);
                return srv.text.decode(cx.a, toks) catch return error.OutOfMemory;
            }
            var parts: std.ArrayList([]const u8) = .empty;
            for (items) |item| try parts.append(cx.a, try promptText(srv, cx, item));
            return std.mem.join(cx.a, "\n", parts.items);
        },
        else => return tool_specs.pyStr(cx.a, p),
    }
}

/// The chat or completion reply to ``raw`` (already decoded), written to ``out``.
pub fn run(srv: *Server, a: Allocator, out: Out, gone: Gone, is_chat: bool, raw: Value) void {
    const field: []const u8 = if (is_chat) "messages" else "prompt";
    const id = ids.make(a, if (is_chat) "chatcmpl-" else "cmpl-", 32) catch return;
    var cx: Cx = .{ .a = a };
    var p = plan(srv, &cx, is_chat, raw) catch |e| {
        if (e == error.OutOfMemory) cx.message = "out of memory";
        logRefused(id, cx.message);
        const body = (if (cx.kind == .other) errorOther(a, cx.message) else errorBody(a, &cx, field)) catch return;
        out.vt.reply(out.ctx, cx.status(), wrapError(a, body) catch return);
        return;
    };
    p.input.id = id;
    var r: Run = .{ .srv = srv, .a = a, .out = out, .is_chat = is_chat, .plan = p, .id = id, .created = std.Io.Clock.real.now(srv.io).toSeconds() };
    if (p.stream) r.stream(gone, field) else r.whole(gone, field);
}

/// Why a request got an error reply, logged before it: its access line shows only the status.
pub fn logRefused(id: []const u8, message: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, message)});
}

pub fn errorOther(a: Allocator, message: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message });
    return .{ .object = o };
}
