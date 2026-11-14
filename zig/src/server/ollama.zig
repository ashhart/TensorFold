//! Ollama's HTTP API on the native server: chat, generate, embeddings and discovery, translated onto the OpenAI seam.
const std = @import("std");
const json = @import("json");
const openai = @import("openai.zig");
const embeddings = @import("embeddings.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const sse = @import("sse.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const errors = @import("errors.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const mapped = [_][]const u8{ "temperature", "top_k", "top_p", "min_p", "stop", "seed", "presence_penalty", "frequency_penalty" };

/// Translate Ollama's `options` and `think` onto the chat request the engine already reads.
pub fn options(a: Allocator, window: u32, raw: Value, out: *json.Object, cx: *Cx) errors.Refused!void {
    _ = a;
    if (raw.get("options")) |opts| {
        if (opts != .object) return cx.refuse("options must be an object");
        var it = opts.object.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "num_ctx")) {
                const v = entry.value_ptr.*;
                if (v == .int) if ((std.fmt.parseInt(i64, v.int, 10) catch 0) > @as(i64, @intCast(window)))
                    return cx.refuse("num_ctx is above this model's context window");
                continue;
            }
            var known = std.mem.eql(u8, key, "num_predict");
            for (mapped) |name| known = known or std.mem.eql(u8, key, name);
            if (!known) return cx.refuse(std.fmt.allocPrint(cx.a, "option {s} is not supported by this engine", .{key}) catch return error.OutOfMemory);
        }
        for (mapped) |key| if (opts.get(key)) |v| try out.put(cx.a, key, v);
        if (opts.get("num_predict")) |v| try out.put(cx.a, "max_tokens", v);
    }
    if (raw.get("think")) |think| {
        if (think != .bool) return cx.refuse("think must be a boolean");
        const kwargs = try json.newObject(cx.a);
        try kwargs.put(cx.a, "enable_thinking", think);
        try out.put(cx.a, "chat_template_kwargs", .{ .object = kwargs });
    }
}

/// `/api/chat` and `/api/generate` body as a chat request with `return_token_ids` on.
pub fn chatBody(a: Allocator, model: []const u8, window: u32, raw: Value, cx: *Cx) errors.Refused!Value {
    if (raw != .object) return cx.refuse("request body must be an object");
    const out = try json.newObject(a);
    try out.put(a, "model", .{ .string = model });
    if (raw.get("messages")) |messages| {
        if (messages != .array) return cx.refuse("messages must be an array");
        try out.put(a, "messages", messages);
    } else if (raw.get("prompt")) |prompt| {
        if (prompt != .string) return cx.refuse("prompt must be a string");
        const list = try a.alloc(Value, 1);
        const message = try json.newObject(a);
        try message.put(a, "role", .{ .string = "user" });
        try message.put(a, "content", prompt);
        list[0] = .{ .object = message };
        try out.put(a, "messages", .{ .array = list });
    } else return cx.refuse("a chat request needs messages; a generate request needs a prompt");
    if (raw.get("system")) |system| {
        if (system != .string) return cx.refuse("system must be a string");
        const messages = out.get("messages").?.array;
        const list = try a.alloc(Value, messages.len + 1);
        const first = try json.newObject(a);
        try first.put(a, "role", .{ .string = "system" });
        try first.put(a, "content", system);
        list[0] = .{ .object = first };
        @memcpy(list[1..], messages);
        try out.put(a, "messages", .{ .array = list });
    }
    if (raw.get("format")) |format| switch (format) {
        .string => |name| {
            if (!std.mem.eql(u8, name, "json")) return cx.refuse("format must be \"json\" or a JSON schema object");
            const rf = try json.newObject(a);
            try rf.put(a, "type", .{ .string = "json_object" });
            try out.put(a, "response_format", .{ .object = rf });
        },
        .object => {
            const rf = try json.newObject(a);
            try rf.put(a, "type", .{ .string = "json_schema" });
            const js = try json.newObject(a);
            try js.put(a, "schema", format);
            try rf.put(a, "json_schema", .{ .object = js });
            try out.put(a, "response_format", .{ .object = rf });
        },
        else => return cx.refuse("format must be \"json\" or a JSON schema object"),
    };
    if (raw.get("tools")) |tools| {
        if (tools != .array) return cx.refuse("tools must be an array");
        try out.put(a, "tools", tools);
        if (raw.get("tool_choice")) |choice| try out.put(a, "tool_choice", choice);
    } else if (raw.get("tool_choice") != null) return cx.refuse("tool_choice needs tools");
    if (raw.get("images") != null) return cx.refuse("images are not supported by this engine");
    if (raw.get("stream")) |stream| {
        if (stream != .bool) return cx.refuse("stream must be true, false or absent");
        try out.put(a, "stream", stream);
    }
    try out.put(a, "return_token_ids", .{ .bool = true });
    try options(a, window, raw, out, cx);
    return .{ .object = out };
}

/// The NDJSON sink: one JSON object per line, no SSE framing.
const Wire = struct {
    conn: *Conn,
    a: Allocator,
    model: []const u8,
    created: i64,
    generated: std.ArrayList(u8) = .empty,
    last: ?Value = null,
    streaming: bool = false,
    generate: bool = false,
    finished: bool = false,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn line(w: *Wire, value: Value) error{Closed}!void {
        const text = json.stringify(w.a, value, .{ .compact = true }) catch return error.Closed;
        w.conn.writeAll(text) catch return error.Closed;
        w.conn.writeAll("\n") catch return error.Closed;
    }
    fn base(w: *Wire, a: Allocator) !*json.Object {
        const o = try json.newObject(a);
        try o.put(a, "model", .{ .string = w.model });
        try o.put(a, "created_at", .{ .string = try rfc3339(a, w.created) });
        return o;
    }
    fn finish(w: *Wire, value: Value) !void {
        w.last = value;
    }
    fn putContent(w: *Wire, o: *json.Object, text: []const u8) !void {
        if (w.generate) {
            try o.put(w.a, "response", .{ .string = text });
            return;
        }
        const m = try json.newObject(w.a);
        try m.put(w.a, "role", .{ .string = "assistant" });
        try m.put(w.a, "content", .{ .string = text });
        try o.put(w.a, "message", .{ .object = m });
    }
    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        const p = payload orelse {
            w.doneLine() catch return error.Closed;
            return;
        };
        // A stream that opened and then failed ends with an error line, never a success line.
        if (p == .object) if (p.get("error")) |e| {
            const message: []const u8 = if (e == .object)
                (e.strField("message") orelse "the request failed")
            else if (e == .string)
                e.string
            else
                "the request failed";
            const o = json.newObject(w.a) catch return error.Closed;
            o.put(w.a, "error", .{ .string = message }) catch return error.Closed;
            w.line(.{ .object = o }) catch return error.Closed;
            w.finished = true;
            return;
        };
        w.finish(p) catch return error.Closed;
        if (p != .object) return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const delta = choices.array[0].get("delta") orelse choices.array[0].get("text") orelse return;
        const text = if (delta == .string) delta.string else delta.strField("content") orelse return;
        if (text.len == 0) return;
        w.generated.appendSlice(w.a, text) catch return error.Closed;
        const o = w.base(w.a) catch return error.Closed;
        w.putContent(o, text) catch return error.Closed;
        o.put(w.a, "done", .{ .bool = false }) catch return error.Closed;
        w.line(.{ .object = o }) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        if (status != 200) {
            const o = json.newObject(w.a) catch return;
            const message = if (payload == .object)
                (if (payload.get("error")) |e| (if (e == .object) e.strField("message") else e.str()) else payload.strField("message")) orelse "the request failed"
            else
                "the request failed";
            o.put(w.a, "error", .{ .string = message }) catch return;
            w.line(.{ .object = o }) catch return;
            w.finished = true;
            return;
        }
        w.finish(payload) catch {};
        if (payload == .object) if (payload.get("choices")) |choices| {
            if (choices == .array and choices.array.len > 0) {
                const first = choices.array[0];
                const text = if (first.get("message")) |m| m.strField("content") orelse "" else first.strField("text") orelse "";
                w.generated.clearRetainingCapacity();
                w.generated.appendSlice(w.a, text) catch {};
            }
        };
        w.doneLine() catch {};
    }
    fn doneLine(w: *Wire) !void {
        if (w.finished) return;
        w.finished = true;
        const o = try w.base(w.a);
        try w.putContent(o, if (w.streaming) "" else w.generated.items);
        const last = w.last;
        if (last) |value| if (value == .object) {
            if (value.get("tensorfold")) |t| if (t == .object) {
                if (t.get("prefill_seconds")) |v| try o.put(w.a, "prompt_eval_duration", try duration(w.a, v));
                if (t.get("seconds")) |v| try o.put(w.a, "eval_duration", try duration(w.a, v));
                if (t.get("token_ids")) |ids| if (ids == .array) {
                    try o.put(w.a, "eval_count", try json.intValue(w.a, ids.array.len));
                    try o.put(w.a, "token_ids", ids);
                };
            };
            if (value.get("usage")) |u| if (u == .object) {
                if (u.get("prompt_tokens")) |v| try o.put(w.a, "prompt_eval_count", v);
            };
            const choices = value.get("choices") orelse Value.null;
            if (choices == .array and choices.array.len > 0) if (choices.array[0].get("finish_reason")) |reason|
                try o.put(w.a, "done_reason", reason);
        };
        if (o.get("done_reason") == null) try o.put(w.a, "done_reason", .{ .string = "stop" });
        try o.put(w.a, "done", .{ .bool = true });
        try w.line(.{ .object = o });
    }
};

/// RFC 3339 UTC time from Unix seconds, as Ollama's `created_at` field.
fn rfc3339(a: Allocator, secs: i64) ![]const u8 {
    const es = std.time.epoch.EpochSeconds{ .secs = if (secs < 0) 0 else @intCast(secs) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.allocPrint(a, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year,              md.month.numeric(),      @as(u8, md.day_index) + 1,
        ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    });
}

fn duration(a: Allocator, seconds: Value) !Value {
    const value: f64 = switch (seconds) {
        .float => |f| f,
        .int => std.fmt.parseFloat(f64, seconds.int) catch 0,
        else => 0,
    };
    return json.intValue(a, @as(i64, @intFromFloat(@max(0, value) * 1e9)));
}

pub fn chat(srv: *Server, conn: *Conn, a: Allocator) void {
    serve(srv, conn, a, false);
}

pub fn generate(srv: *Server, conn: *Conn, a: Allocator) void {
    serve(srv, conn, a, true);
}

fn serve(srv: *Server, conn: *Conn, a: Allocator, is_generate: bool) void {
    var cx: Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch {
        const err = openai.errorBody(a, &cx, "messages") catch return;
        return routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
    };
    const translated = chatBody(a, srv.config.served_name, @intCast(srv.info.context_window), raw, &cx) catch {
        const err = openai.errorBody(a, &cx, "messages") catch return;
        return routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
    };
    // A refusal must keep its status: plan before the 200 head goes out.
    var pre: Cx = .{ .a = a };
    _ = openai.plan(srv, &pre, true, translated, null, true) catch {
        const err = openai.errorBody(a, &pre, "messages") catch return;
        return routes.sendValue(conn, a, pre.status(), openai.wrapError(a, err) catch return);
    };
    const streaming = if (raw.get("stream")) |v| v.truthy() else false;
    var wire: Wire = .{
        .conn = conn,
        .a = a,
        .model = srv.config.served_name,
        .created = std.Io.Clock.real.now(srv.io).toSeconds(),
        .streaming = streaming,
        .generate = is_generate,
    };
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "application/x-ndjson") catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish("") catch return;
    openai.runNamed(srv, a, wire.out(), .{ .conn = conn }, true, translated, null, true, true);
}

pub fn embed(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const raw = http_body.readJson(conn, &cx) catch {
        const err = openai.errorBody(a, &cx, "input") catch return;
        return routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
    };
    const Gone = struct {
        c: *Conn,
        pub fn check(g: @This()) bool {
            return g.c.peerGone();
        }
    };
    const value = embeddings.reply(srv, &cx, raw, Gone{ .c = conn }) catch {
        const err = openai.errorBody(a, &cx, "input") catch return;
        return routes.sendValue(conn, a, cx.status(), openai.wrapError(a, err) catch return);
    };
    const out = json.newObject(a) catch return;
    out.put(a, "model", .{ .string = srv.config.served_name }) catch return;
    const source = if (value == .object) value.get("data") else null;
    const rows: []const Value = if (source != null and source.? == .array) source.?.array else &.{};
    const list = a.alloc(Value, rows.len) catch return;
    for (rows, list) |item, *slot| slot.* = if (item == .object) item.get("embedding") orelse Value{ .array = &.{} } else .{ .array = &.{} };
    out.put(a, "embeddings", .{ .array = list }) catch return;
    if (value == .object) if (value.get("usage")) |u| if (u == .object) if (u.get("prompt_tokens")) |v|
        out.put(a, "prompt_eval_count", v) catch return;
    return routes.sendValue(conn, a, 200, .{ .object = out });
}

pub fn tags(srv: *Server, conn: *Conn, a: Allocator) void {
    const list = a.alloc(Value, srv.config.model_ids.len) catch return;
    for (srv.config.model_ids, list) |id, *slot| {
        const model = json.newObject(a) catch return;
        model.put(a, "name", .{ .string = id }) catch return;
        model.put(a, "model", .{ .string = id }) catch return;
        slot.* = .{ .object = model };
    }
    const o = json.newObject(a) catch return;
    o.put(a, "models", .{ .array = list }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn show(srv: *Server, conn: *Conn, a: Allocator) void {
    _ = http_body.read(conn, a, http_body.limit) catch {};
    const o = json.newObject(a) catch return;
    o.put(a, "modelfile", .{ .string = "" }) catch return;
    o.put(a, "parameters", .{ .string = "" }) catch return;
    o.put(a, "template", .{ .string = srv.text.templateSource() }) catch return;
    o.put(a, "details", .{ .object = json.newObject(a) catch return }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn ps(srv: *Server, conn: *Conn, a: Allocator) void {
    const o = json.newObject(a) catch return;
    const list = a.alloc(Value, 1) catch return;
    const model = json.newObject(a) catch return;
    model.put(a, "name", .{ .string = srv.config.served_name }) catch return;
    model.put(a, "model", .{ .string = srv.config.served_name }) catch return;
    list[0] = .{ .object = model };
    o.put(a, "models", .{ .array = list }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn version(srv: *const Server, conn: *Conn, a: Allocator) void {
    const o = json.newObject(a) catch return;
    o.put(a, "version", .{ .string = srv.config.identity.version }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}
