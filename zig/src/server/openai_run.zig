//! One OpenAI request's run: the reply generated, then sent whole or streamed as chunks.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const compact = @import("compact.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const request_log = @import("request_log.zig");
const sse = @import("sse.zig");
const warm = @import("warm.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const openai = @import("openai.zig");
const Out = openai.Out;
const Gone = openai.Gone;
const errorBody = openai.errorBody;
const wrapError = openai.wrapError;
const replyModel = openai.replyModel;
const Plan = openai.Plan;
const logRefused = openai.logRefused;
const errorOther = openai.errorOther;

pub const Run = struct {
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
        if (reply.calls) |calls| return if (calls.len > 0) calls else null; // the family parsed them (and applied the one-call policy)
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

    pub fn whole(r: *Run, gone: Gone, field: []const u8) void {
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
            const empty_null = r.srv.family != null and reply.content.len == 0; // a family's reply: ``content or None``
            try message.put(a, "content", if (calls != null or empty_null) .null else .{ .string = reply.content });
            if (reply.reasoning) |t| if (t.len > 0) {
                if (r.srv.family) |fam| {
                    const d = try fam.reasoningDelta(a, t);
                    for (d.object.keys(), d.object.values()) |k, v| try message.put(a, k, v);
                } else try message.put(a, "reasoning_content", .{ .string = t });
            };
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

    pub fn stream(r: *Run, gone: Gone, field: []const u8) void {
        const a = r.a;
        // a family streams calls and prose itself, so its stream is the plain one
        const tools = r.plan.input.tools.len > 0 and r.srv.family == null;
        var cx: Cx = .{ .a = a };
        // a context-window refusal is reported inside the stream
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
        var live = gone;
        live.out = r.out;
        live.every_ns = @intFromFloat(r.srv.config.sse_keepalive_s * std.time.ns_per_s);
        var reply = chat.generate(r.srv, &cx, prepared, .{ .ctx = &sink_state, .call = StreamSink.call }, live) catch |e| {
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
