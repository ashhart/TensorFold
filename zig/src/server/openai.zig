//! POST /v1/chat/completions and /v1/completions, JSON or SSE, as ``server.http`` answers them.
const std = @import("std");
const json = @import("json.zig");
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
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
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
};

/// Whether the client has gone, read between rounds (Python's socket_cancellation).
pub const Gone = struct {
    conn: *Conn,
    pub fn check(g: Gone) bool {
        return g.conn.peerGone();
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

const Plan = struct {
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
    var input: chat.Input = .{ .fields = undefined };
    if (is_chat) {
        input.messages = try messages.normalizeWith(cx, body.get("messages"), "system", srv.needs_user_after_tool, .of(srv.info.vision));
        input.tools = try tool_specs.active(cx, body.get("tools"), body.get("tool_choice"));
    } else if (body.get("messages")) |m| if (m == .array and m.array.len > 0) {
        input.messages = try messages.normalizeWith(cx, m, "system", srv.needs_user_after_tool, .of(srv.info.vision));
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
    if (input.tools.len > 0 and try tool_specs.choiceRequiresCall(cx.a, body.get("tool_choice"))) try sampling.put(cx.a, "tool_call_required", .{ .bool = true });
    const thinking = try fields.thinkingFields(cx, body, srv.effort_levels);
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
fn logRefused(id: []const u8, message: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, message)});
}

fn errorOther(a: Allocator, message: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message });
    return .{ .object = o };
}

const Run = struct {
    srv: *Server,
    a: Allocator,
    out: Out,
    is_chat: bool,
    plan: Plan,
    id: []const u8,
    created: i64,
    streamed_prose: bool = false,
    prose_sent: std.ArrayList(u8) = .empty, // the content a tool stream sent

    fn chunk(r: *Run, delta: ?Value, finish: ?[]const u8) Allocator.Error!Value {
        const a = r.a;
        const o = try json.newObject(a);
        try o.put(a, "id", .{ .string = r.id });
        try o.put(a, "object", .{ .string = if (r.is_chat) "chat.completion.chunk" else "text_completion" });
        try o.put(a, "created", try json.intValue(a, r.created));
        try o.put(a, "model", .{ .string = r.plan.named });
        const choice = try json.newObject(a);
        try choice.put(a, "index", .{ .int = "0" });
        if (r.is_chat) {
            const d: Value = if (delta) |v| (if (v == .object) v else if (v.string.len > 0) try chat.deltaOf(a, "content", v.string) else .{ .object = try json.newObject(a) }) else .{ .object = try json.newObject(a) };
            try choice.put(a, "delta", d);
            try choice.put(a, "finish_reason", if (finish) |f| .{ .string = f } else .null);
        } else {
            try choice.put(a, "text", delta orelse .{ .string = "" });
            try choice.put(a, "finish_reason", if (finish) |f| .{ .string = f } else .null);
            try choice.put(a, "logprobs", .null);
        }
        const list = try a.alloc(Value, 1);
        list[0] = .{ .object = choice };
        try o.put(a, "choices", .{ .array = list });
        return .{ .object = o };
    }

    fn emit(r: *Run, payload: Value) error{Closed}!void {
        return r.out.vt.event(r.out.ctx, payload);
    }

    fn usage(r: *Run, reply: *const chat.Reply) Allocator.Error!Value {
        const a = r.a;
        const o = try json.newObject(a);
        try o.put(a, "prompt_tokens", try json.intValue(a, reply.prompt_tokens));
        try o.put(a, "completion_tokens", try json.intValue(a, reply.completion_tokens));
        try o.put(a, "total_tokens", try json.intValue(a, reply.prompt_tokens + reply.completion_tokens));
        const cached = try json.newObject(a);
        try cached.put(a, "cached_tokens", try json.intValue(a, reply.cached_tokens));
        try o.put(a, "prompt_tokens_details", .{ .object = cached });
        const thought = try json.newObject(a);
        try thought.put(a, "reasoning_tokens", try json.intValue(a, reply.reasoning_tokens));
        try o.put(a, "completion_tokens_details", .{ .object = thought });
        return .{ .object = o };
    }

    /// ``response_extras``: exact_mode, the matched stop, the runtime and the draft counters.
    fn extras(r: *Run, o: *json.Object, reply: *const chat.Reply) Allocator.Error!void {
        try o.put(r.a, "exact_mode", .{ .string = "exact" });
        if (reply.stop_sequence) |s| try o.put(r.a, "stop_sequence", .{ .string = s });
        try o.put(r.a, "tensorfold", reply.runtime);
        try o.put(r.a, "speculative", reply.speculative);
    }

    /// ``ToolCallPolicy.finish``: the reply's calls parsed out of its content.
    fn attachCalls(r: *Run, reply: *chat.Reply) Allocator.Error!?[]Value {
        const tools = r.plan.input.tools;
        if (tools.len == 0) return null;
        const parsed = try tool_parse.parse(r.a, reply.content, tools, r.plan.policy.maxCalls());
        const calls = parsed.calls orelse {
            if (r.plan.policy.single) reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
            return null;
        };
        reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
        reply.finish_reason = "tool_calls";
        if (r.plan.policy.single) reply.tool_calls_streamed = false;
        return if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls;
    }

    fn fail(r: *Run, cx: *const Cx, field: []const u8) void {
        const a = r.a;
        const kind_ok = cx.kind != .other and cx.kind != .server;
        const body = (if (kind_ok) errorBody(a, cx, field) else errorOther(a, cx.message)) catch return;
        r.out.vt.reply(r.out.ctx, if (kind_ok) 400 else 500, wrapError(a, body) catch return);
    }

    /// A reply that ended before anything was sent: a refusal's 400, a failure's 500, nothing when the client left.
    fn unsent(r: *Run, cx: *Cx, e: chat.Failure, field: []const u8) void {
        switch (e) {
            error.Cancelled => {},
            error.OutOfMemory => r.fail(&.{ .a = r.a, .kind = .server, .message = "out of memory" }, field),
            error.Failed => r.fail(&.{ .a = r.a, .kind = .server, .message = "the reply failed" }, field),
            error.Refused => {
                if (cx.kind == .other) cx.kind = .server;
                if (cx.kind != .server) logRefused(r.id, cx.message); // a 500 is a failure, not a refusal
                r.fail(cx, field);
            },
        }
    }

    /// A context-window refusal on a stream: the role chunk when there are no tools, then the error event and [DONE].
    fn streamedContext(r: *Run, cx: *const Cx, tools: bool, field: []const u8) void {
        const a = r.a;
        r.out.vt.open(r.out.ctx) catch return;
        if (!tools and r.is_chat) r.emit(r.chunk(roleDelta(a) catch return, null) catch return) catch return;
        logRefused(r.id, cx.message);
        const body = errorBody(a, cx, field) catch return;
        r.emit(wrapError(a, body) catch return) catch return;
        r.out.vt.event(r.out.ctx, null) catch {};
    }

    fn whole(r: *Run, gone: Gone, field: []const u8) void {
        const a = r.a;
        var cx: Cx = .{ .a = a };
        var reply = chat.run(r.srv, &cx, r.plan.input, null, gone) catch |e| return r.unsent(&cx, e, field);
        const calls = r.attachCalls(&reply) catch return;
        const o = json.newObject(a) catch return;
        r.wholeBody(o, &reply, calls) catch return;
        r.out.vt.reply(r.out.ctx, 200, .{ .object = o });
        r.warmNext(&reply, calls);
    }

    /// The next turn's prompt prefilled in the background, from the reply as its client got it (warm.zig).
    fn warmNext(r: *Run, reply: *const chat.Reply, calls: ?[]Value) void {
        if (!r.is_chat) return;
        const message = json.newObject(r.a) catch return;
        message.put(r.a, "role", .{ .string = "assistant" }) catch return;
        message.put(r.a, "content", if (calls != null) .null else .{ .string = reply.content }) catch return;
        if (calls) |c| message.put(r.a, "tool_calls", .{ .array = c }) catch return;
        warm.reply(r.srv, r.plan.input, .{ .object = message }, reply.thinking, reply.effort);
    }

    fn wholeBody(r: *Run, o: *json.Object, reply: *const chat.Reply, calls: ?[]Value) Allocator.Error!void {
        const a = r.a;
        try o.put(a, "id", .{ .string = r.id });
        try o.put(a, "object", .{ .string = if (r.is_chat) "chat.completion" else "text_completion" });
        try o.put(a, "created", try json.intValue(a, r.created));
        try o.put(a, "model", .{ .string = r.plan.named });
        const choice = try json.newObject(a);
        try choice.put(a, "index", .{ .int = "0" });
        if (r.is_chat) {
            const message = try json.newObject(a);
            try message.put(a, "role", .{ .string = "assistant" });
            try message.put(a, "content", if (calls != null) .null else .{ .string = reply.content });
            if (reply.reasoning) |t| if (t.len > 0) try message.put(a, "reasoning_content", .{ .string = t });
            if (calls) |c| try message.put(a, "tool_calls", .{ .array = c });
            try choice.put(a, "message", .{ .object = message });
            try choice.put(a, "finish_reason", .{ .string = reply.finish_reason });
            if (reply.logprobs) |lp| try choice.put(a, "logprobs", lp);
        } else {
            try choice.put(a, "text", .{ .string = reply.content });
            try choice.put(a, "finish_reason", .{ .string = reply.finish_reason });
            try choice.put(a, "logprobs", .null);
        }
        const list = try a.alloc(Value, 1);
        list[0] = .{ .object = choice };
        try o.put(a, "choices", .{ .array = list });
        try o.put(a, "usage", try r.usage(reply));
        try r.extras(o, reply);
    }

    const StreamSink = struct {
        run: *Run,
        tools: bool,

        fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
            const s: *StreamSink = @ptrCast(@alignCast(ctx));
            const r = s.run;
            if (!s.tools) {
                if (!r.is_chat and delta != .string) return; // completions stream text only
                return r.emit(r.chunk(delta, null) catch return error.Closed);
            }
            const filtered = r.policyDelta(delta) catch return error.Closed;
            const d = filtered orelse return;
            try r.prose();
            r.prose_sent.appendSlice(r.a, if (d == .string) d.string else d.strField("content") orelse "") catch return error.Closed;
            return r.emit(r.chunk(d, null) catch return error.Closed);
        }
    };

    /// The one-call policy on a streamed delta: text filtered, tool_calls dropped; null when nothing is left.
    fn policyDelta(r: *Run, delta: Value) Allocator.Error!?Value {
        if (!r.plan.policy.single) return if (delta.truthy()) delta else null;
        if (delta == .string) {
            const t = try r.plan.policy.text(r.a, delta.string);
            return if (t.len > 0) Value{ .string = t } else null;
        }
        const o = try json.newObject(r.a);
        var content: ?[]const u8 = null;
        for (delta.object.keys(), delta.object.values()) |k, v| {
            if (std.mem.eql(u8, k, "tool_calls")) continue;
            if (std.mem.eql(u8, k, "content") and v == .string) {
                content = v.string;
                continue;
            }
            try o.put(r.a, k, v);
        }
        if (content) |c| {
            const t = try r.plan.policy.text(r.a, c);
            if (t.len > 0) try o.put(r.a, "content", .{ .string = t });
        }
        return if (o.count() > 0) Value{ .object = o } else null;
    }

    /// The assistant role before the first prose of a tool-using stream.
    fn prose(r: *Run) error{Closed}!void {
        if (r.streamed_prose) return;
        r.streamed_prose = true;
        const role = roleDelta(r.a) catch return error.Closed;
        return r.emit(r.chunk(role, null) catch return error.Closed);
    }

    fn stream(r: *Run, gone: Gone, field: []const u8) void {
        const a = r.a;
        const tools = r.plan.input.tools.len > 0;
        var cx: Cx = .{ .a = a };
        // a context-window refusal is reported inside the stream; every other refusal is still a 400 before it opens
        var compaction: ?compact.Stamp = null;
        const prepared = if (r.srv.config.compact_at == null) chat.prepare(r.srv, &cx, r.plan.input, gone) catch |e| {
            if (e == error.Refused and cx.kind == .context_length) return r.streamedContext(&cx, tools, field);
            return r.unsent(&cx, e, field);
        } else blk: {
            const ready = compact.prepare(r.srv, &cx, r.plan.input, gone) catch |e| {
                if (e == error.Refused and cx.kind == .context_length) return r.streamedContext(&cx, tools, field);
                return r.unsent(&cx, e, field);
            };
            compaction = ready.stamp;
            break :blk ready.prepared;
        };
        var handed = false; // generate gives the preparing count back from here on
        defer if (!handed) chat.release(r.srv, prepared.preparing);
        r.out.vt.open(r.out.ctx) catch return;
        if (!tools and r.is_chat) r.emit(r.chunk(roleDelta(a) catch return, null) catch return) catch return;
        var sink_state: StreamSink = .{ .run = r, .tools = tools };
        handed = true;
        var reply = chat.generate(r.srv, &cx, prepared, .{ .ctx = &sink_state, .call = StreamSink.call }, gone) catch |e| {
            switch (e) {
                error.Cancelled => return,
                error.Refused => if (cx.kind != .other and cx.kind != .server) {
                    logRefused(r.id, cx.message);
                    const body = errorBody(a, &cx, field) catch return;
                    r.emit(wrapError(a, body) catch return) catch return;
                    r.out.vt.event(r.out.ctx, null) catch {};
                    return;
                },
                else => {},
            }
            const message = if (e == error.Refused) cx.message else if (e == error.OutOfMemory) "out of memory" else "the reply failed";
            log.line("stream error: {s}", .{message});
            const err = json.newObject(a) catch return;
            err.put(a, "message", .{ .string = message }) catch return;
            err.put(a, "type", .{ .string = "server_error" }) catch return;
            r.emit(wrapError(a, .{ .object = err }) catch return) catch return;
            r.out.vt.event(r.out.ctx, null) catch {};
            return;
        };
        var calls: ?[]Value = null;
        if (tools) {
            calls = r.attachCalls(&reply) catch return;
            const tail = r.plan.policy.flush();
            if (tail.len > 0) {
                r.prose() catch return;
                r.prose_sent.appendSlice(a, tail) catch return;
                r.emit(r.chunk(.{ .string = tail }, null) catch return) catch return;
            }
            if (calls != null and !reply.tool_calls_streamed) {
                r.emit(r.chunk(roleDelta(a) catch return, null) catch return) catch return;
                const deltas = tool_parse.deltas(a, calls.?) catch return;
                for (deltas) |d| r.emit(r.chunk(d, null) catch return) catch return;
            } else if (reply.content.len > 0 and !r.streamed_prose) {
                r.emit(r.chunk(.{ .string = reply.content }, null) catch return) catch return;
            } else if (r.plan.policy.kept(reply.content, r.prose_sent.items)) |rest| {
                r.emit(r.chunk(.{ .string = rest }, null) catch return) catch return;
            }
        }
        const last = r.chunk(.{ .string = "" }, if (reply.finish_reason.len > 0) reply.finish_reason else "length") catch return;
        if (compaction) |s| compact.stamp(&cx, &reply, s) catch return;
        r.extras(last.object, &reply) catch return;
        const use = r.usage(&reply) catch return;
        if (!r.plan.separate_usage) last.object.put(a, "usage", use) catch return;
        r.emit(last) catch return;
        if (r.plan.separate_usage) {
            const u = r.chunk(null, null) catch return;
            u.object.put(a, "choices", .{ .array = &.{} }) catch return;
            u.object.put(a, "usage", use) catch return;
            r.emit(u) catch return;
        }
        r.out.vt.event(r.out.ctx, null) catch {};
        r.warmNext(&reply, calls);
    }
};

fn roleDelta(a: Allocator) Allocator.Error!Value {
    return chat.deltaOf(a, "role", "assistant");
}
