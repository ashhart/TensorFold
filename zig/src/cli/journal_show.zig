//! `tensorfold log show`: one journaled request as it was sent (redactions kept) and its reply decoded from base64.
const std = @import("std");
const log = @import("journal_log.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Writer = std.Io.Writer;
const Entry = log.Entry;

pub fn entry(a: Allocator, out: *Writer, e: Entry) !void {
    try out.print("id        {s}\nstarted   ", .{e.id});
    try log.stamp(out, if (e.start) |s| log.int(s, "at") else null);
    try out.writeAll(" UTC\n");
    try out.print("route     {s}\nmodel     {s}\n", .{ log.pathOf(e), log.modelOf(e) });
    try out.writeAll("status    ");
    if (e.finish) |f| {
        try out.print("{d} {s}{s}", .{ log.int(f, "status") orelse 0, log.str(f, "outcome") orelse "finished", if (stream(f)) " (stream)" else "" });
        if (e.quarantined) try out.writeAll(", quarantined");
    } else try out.writeAll(if (e.quarantined) "quarantined: it was in flight when the server crashed" else "in flight (no finish yet)");
    try out.writeAll("\n");
    if (e.finish) |f| {
        try out.writeAll("tokens    prompt ");
        try opt(out, log.int(f, "prompt_tokens"));
        try out.writeAll(", cached ");
        try opt(out, log.int(f, "cached_tokens"));
        try out.writeAll(", completion ");
        try opt(out, log.int(f, "completion_tokens"));
        try out.writeAll("\ntimings   queue ");
        try optMs(out, log.num(f, "queue_ms"));
        try out.writeAll(", first token ");
        try optMs(out, log.num(f, "ttft_ms"));
        try out.writeAll(", decode ");
        if (log.num(f, "decode_tps")) |v| try out.print("{d:.1} tok/s\n", .{v}) else try out.writeAll("-\n");
    }
    try out.writeAll("\nrequest\n");
    try request(a, out, e);
    try out.writeAll("\nreply\n");
    const f = e.finish orelse return out.writeAll("  (none yet)\n");
    const encoded = log.str(f, "reply_base64") orelse {
        return out.print("  not kept: {d} bytes, sha256 {s}, token sha256 {s}\n", .{ log.int(f, "reply_bytes") orelse 0, log.str(f, "reply_sha256") orelse "-", log.str(f, "token_sha256") orelse "-" });
    };
    const codec = std.base64.standard.Decoder;
    const bytes = try a.alloc(u8, codec.calcSizeForSlice(encoded) catch return out.writeAll("  (its base64 is unreadable)\n"));
    codec.decode(bytes, encoded) catch return out.writeAll("  (its base64 is unreadable)\n");
    try reply(a, out, bytes, stream(f));
}

fn stream(f: Value) bool {
    const v = f.object.get("stream") orelse return false;
    return v == .bool and v.bool;
}

/// The body as journaled: each message's role and text (other parts as JSON, redactions kept), or the prompt.
fn request(a: Allocator, out: *Writer, e: Entry) !void {
    const s = e.start orelse return out.writeAll("  (its start record is not in the kept files)\n");
    const body = s.object.get("body") orelse {
        return out.print("  not kept: {d} bytes (too large for one record)\n", .{log.int(s, "body_bytes") orelse 0});
    };
    if (body != .object) return out.writeAll("  (unreadable body)\n");
    if (body.object.get("messages")) |messages| if (messages == .array) {
        for (messages.array.items) |m| {
            if (m != .object) continue;
            try out.print("  [{s}] ", .{log.str(m, "role") orelse "?"});
            try content(out, m.object.get("content") orelse .null);
            if (m.object.get("tool_calls")) |calls| {
                try out.writeAll(" tool_calls ");
                try std.json.Stringify.value(calls, .{}, out);
            }
            try out.writeAll("\n");
        }
        return;
    };
    if (body.object.get("prompt")) |prompt| {
        try out.writeAll("  [prompt] ");
        try content(out, prompt);
        return out.writeAll("\n");
    }
    _ = a;
    try out.writeAll("  ");
    try std.json.Stringify.value(body, .{}, out);
    try out.writeAll("\n");
}

fn content(out: *Writer, v: Value) !void {
    switch (v) {
        .string => |text| try out.writeAll(text),
        .array => |parts| for (parts.items, 0..) |p, i| {
            if (i > 0) try out.writeAll(" ");
            if (p == .object and log.str(p, "type") != null and std.mem.eql(u8, log.str(p, "type").?, "text")) {
                try out.writeAll(log.str(p, "text") orelse "");
            } else try std.json.Stringify.value(p, .{}, out);
        },
        .null => {},
        else => try std.json.Stringify.value(v, .{}, out),
    }
}

/// What the client read: a whole JSON body, or a stream's deltas joined.
const Reply = struct {
    text: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    finish: ?[]const u8 = null,
    err: ?[]const u8 = null,
    const Call = struct { name: std.ArrayList(u8) = .empty, arguments: std.ArrayList(u8) = .empty };
};

fn reply(a: Allocator, out: *Writer, bytes: []const u8, streamed: bool) !void {
    var r: Reply = .{};
    if (streamed) {
        var it = std.mem.splitSequence(u8, bytes, "\n\n");
        while (it.next()) |frame| {
            if (!std.mem.startsWith(u8, frame, "data: ")) continue;
            const data = frame["data: ".len..];
            if (std.mem.eql(u8, data, "[DONE]")) continue;
            const v = std.json.parseFromSliceLeaky(Value, a, data, .{}) catch continue;
            try take(a, &r, v, true);
        }
    } else {
        const v = std.json.parseFromSliceLeaky(Value, a, bytes, .{}) catch {
            return out.print("  {d} bytes that are not JSON\n", .{bytes.len});
        };
        try take(a, &r, v, false);
    }
    if (r.err) |m| try out.print("  error: {s}\n", .{m});
    if (r.reasoning.items.len > 0) try out.print("  reasoning: {s}\n", .{r.reasoning.items});
    try out.print("  text: {s}\n", .{r.text.items});
    for (r.calls.items) |c| try out.print("  tool call: {s}({s})\n", .{ c.name.items, c.arguments.items });
    try out.print("  finish reason: {s}\n", .{r.finish orelse "-"});
}

/// One reply body or stream event: the first choice's message or delta, text for a completion.
fn take(a: Allocator, r: *Reply, v: Value, delta: bool) !void {
    if (v != .object) return;
    if (v.object.get("error")) |e| r.err = if (e == .object) log.str(e, "message") orelse "?" else "?";
    const choices = v.object.get("choices") orelse return;
    if (choices != .array or choices.array.items.len == 0) return;
    const c = choices.array.items[0];
    if (c != .object) return;
    if (log.str(c, "finish_reason")) |f| r.finish = f;
    if (log.str(c, "text")) |t| try r.text.appendSlice(a, t);
    const m = c.object.get(if (delta) "delta" else "message") orelse return;
    if (m != .object) return;
    if (log.str(m, "content")) |t| try r.text.appendSlice(a, t);
    if (log.str(m, "reasoning_content") orelse log.str(m, "reasoning")) |t| try r.reasoning.appendSlice(a, t);
    const calls = m.object.get("tool_calls") orelse return;
    if (calls != .array) return;
    for (calls.array.items, 0..) |call, i| {
        if (call != .object) continue;
        const at: usize = if (log.int(call, "index")) |n| @intCast(@max(0, n)) else i;
        while (r.calls.items.len <= at) try r.calls.append(a, .{});
        const fun = call.object.get("function") orelse continue;
        if (fun != .object) continue;
        if (log.str(fun, "name")) |n| try r.calls.items[at].name.appendSlice(a, n);
        if (log.str(fun, "arguments")) |x| try r.calls.items[at].arguments.appendSlice(a, x);
    }
}

fn opt(out: *Writer, v: ?i64) !void {
    if (v) |n| try out.print("{d}", .{n}) else try out.writeAll("-");
}

fn optMs(out: *Writer, v: ?f64) !void {
    if (v) |n| try out.print("{d:.1} ms", .{n}) else try out.writeAll("-");
}
