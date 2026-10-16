//! OpenAI ``logprobs`` (who may ask, ``choices[0].logprobs`` from the engine's rows) and vLLM's ``prompt_logprobs``.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const reply_text = @import("reply_text.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The rows are cut where the text is: after the end-of-thinking token, which must be a token of its own.
pub fn admit(srv: *Server, cx: *Cx, thinking: bool, think_budget: u32, count: u8) errors.Refused!u8 {
    if (thinking and srv.text.tokenId(srv.markers.close) == null) return cx.refuse("logprobs with thinking on need the template's end-of-thinking marker as one token");
    if (think_budget > 0) return cx.refuse("logprobs support chat without tools, stop strings, structured output or a thinking budget");
    return count;
}

/// ``choices[0].logprobs`` for the answer's tokens: after the first end-of-thinking token and dropped newlines.
pub fn value(srv: *Server, cx: *Cx, a: Allocator, thinking: bool, tokens: []const u32, rows: []const api.LogprobRow) errors.Refused!Value {
    if (rows.len < tokens.len) return cx.other("the engine gave fewer logprob rows than reply tokens");
    const start = (try answerStart(srv, a, thinking, tokens)) orelse tokens.len;
    return wrap(a, entries(srv, a, tokens[start..], rows[start..tokens.len]) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Failed => cx.other("the logprob rows are out of step with the reply's tokens, or a forced token has none"),
    });
}

/// A streamed reply's rows: each content chunk carries those of the answer tokens its text now spells.
pub const Stream = struct {
    rows: std.ArrayList(api.LogprobRow) = .empty, // the reply's rows, copied with each round's tokens
    sent: usize = 0, // answer rows already sent
    spelled: usize = 0, // bytes of the streamed content those rows spell

    /// ``{content, logprobs}`` for `text`, with the rows `streamed` content bytes now cover (all of them once `done`).
    pub fn delta(st: *Stream, srv: *Server, a: Allocator, thinking: bool, ignore_eos: bool, collected: []const u32, text: []const u8, streamed: usize, done: bool) error{ Failed, OutOfMemory }!?Value {
        const tokens = if (ignore_eos) collected else reply_text.stripTrailing(collected, srv.eos);
        if (st.rows.items.len < tokens.len) return error.Failed;
        var ready: usize = 0;
        if (try answerStart(srv, a, thinking, tokens)) |start| {
            for (tokens[start + st.sent ..], start + st.sent..) |id, i| {
                const bytes = (try srv.text.tokenBytes(a, id)) orelse &.{};
                // the answer's text drops the newlines its first token may start with
                const lead = if (thinking and i == start) bytes.len - std.mem.trimStart(u8, bytes, "\n").len else 0;
                if (!done and st.spelled + bytes.len - lead > streamed) break;
                st.spelled += bytes.len - lead;
                ready += 1;
            }
            if (ready > 0) {
                const from = start + st.sent;
                const lp = try wrap(a, try entries(srv, a, tokens[from..][0..ready], st.rows.items[from..][0..ready]));
                st.sent += ready;
                const o = try json.newObject(a);
                try o.put(a, "content", .{ .string = text });
                try o.put(a, "logprobs", lp);
                return .{ .object = o };
            }
        }
        if (text.len == 0) return null;
        const o = try json.newObject(a);
        try o.put(a, "content", .{ .string = text });
        return .{ .object = o };
    }
};

/// ``/v1/completions``' legacy ``logprobs`` from the chat form's rows; ``text_offset`` counts on from `offset` chars.
pub fn legacy(a: Allocator, rows: Value, offset: *usize) Allocator.Error!Value {
    const items = rows.object.get("content").?.array;
    const tokens = try a.alloc(Value, items.len);
    const lps = try a.alloc(Value, items.len);
    const tops = try a.alloc(Value, items.len);
    const offsets = try a.alloc(Value, items.len);
    for (items, tokens, lps, tops, offsets) |r, *t, *lp, *top, *at| {
        const text = r.object.get("token").?.string;
        t.* = .{ .string = text };
        lp.* = r.object.get("logprob").?;
        const alts = try json.newObject(a);
        for (r.object.get("top_logprobs").?.array) |alt| {
            const got = try alts.getOrPut(a, alt.object.get("token").?.string);
            if (!got.found_existing) got.value_ptr.* = alt.object.get("logprob").?;
        }
        top.* = .{ .object = alts };
        at.* = try json.intValue(a, offset.*);
        offset.* += reply_text.charCount(text);
    }
    const out = try json.newObject(a);
    try out.put(a, "tokens", .{ .array = tokens });
    try out.put(a, "token_logprobs", .{ .array = lps });
    try out.put(a, "top_logprobs", .{ .array = tops });
    try out.put(a, "text_offset", .{ .array = offsets });
    return .{ .object = out };
}

/// vLLM's ``prompt_logprobs``: null, then ``{id: {logprob, rank, decoded_token}}`` a token, the prompt token first.
pub fn promptValue(srv: *Server, a: Allocator, rows: []const api.LogprobRow) Allocator.Error!Value {
    const out = try a.alloc(Value, rows.len + 1);
    out[0] = .null;
    for (out[1..], rows) |*slot, r| {
        const o = try json.newObject(a);
        try promptEntry(srv, a, o, r.token, r.logprob, r.rank);
        for (r.ids[0..r.count], r.logprobs[0..r.count], 1..) |id, lp, rank| if (id != r.token) try promptEntry(srv, a, o, id, lp, rank);
        slot.* = .{ .object = o };
    }
    return .{ .array = out };
}

fn promptEntry(srv: *Server, a: Allocator, o: *json.Object, id: u32, logprob: f32, rank: usize) Allocator.Error!void {
    const e = try json.newObject(a);
    try e.put(a, "logprob", .{ .float = logprob });
    try e.put(a, "rank", try json.intValue(a, rank));
    try e.put(a, "decoded_token", .{ .string = try srv.text.decode(a, &.{id}) });
    try o.put(a, try std.fmt.allocPrint(a, "{d}", .{id}), .{ .object = e });
}

/// Where the answer's rows start: past the first end-of-thinking token and newline-only tokens; null: no answer yet.
fn answerStart(srv: *Server, a: Allocator, thinking: bool, tokens: []const u32) Allocator.Error!?usize {
    if (!thinking) return 0;
    const end = srv.text.tokenId(srv.markers.close) orelse return null;
    var start = (std.mem.indexOfScalar(u32, tokens, end) orelse return null) + 1;
    while (start < tokens.len) : (start += 1) {
        const bytes = (try srv.text.tokenBytes(a, tokens[start])) orelse return start;
        if (bytes.len == 0 or std.mem.indexOfNone(u8, bytes, "\n") != null) return start;
    }
    return null;
}

fn wrap(a: Allocator, content: []Value) Allocator.Error!Value {
    const out = try json.newObject(a);
    try out.put(a, "content", .{ .array = content });
    return .{ .object = out };
}

fn entries(srv: *Server, a: Allocator, tokens: []const u32, rows: []const api.LogprobRow) error{ Failed, OutOfMemory }![]Value {
    const content = try a.alloc(Value, tokens.len);
    for (content, tokens, rows) |*slot, id, r| {
        if (r.token != id or std.math.isNan(r.logprob)) return error.Failed;
        const o = try entry(srv, a, id, r.logprob);
        const top = try a.alloc(Value, r.count);
        for (top, r.ids[0..r.count], r.logprobs[0..r.count]) |*t, alt, lp| t.* = .{ .object = try entry(srv, a, alt, lp) };
        try o.put(a, "top_logprobs", .{ .array = top });
        slot.* = .{ .object = o };
    }
    return content;
}

/// ``{token, logprob, bytes}``: the token's text, and its own bytes (a partial UTF-8 sequence kept whole).
fn entry(srv: *Server, a: Allocator, id: u32, logprob: f32) Allocator.Error!*json.Object {
    const bytes = (try srv.text.tokenBytes(a, id)) orelse &.{};
    const o = try json.newObject(a);
    try o.put(a, "token", .{ .string = try srv.text.decode(a, &.{id}) });
    try o.put(a, "logprob", .{ .float = logprob });
    const b = try a.alloc(Value, bytes.len);
    for (b, bytes) |*v, x| v.* = try json.intValue(a, x);
    try o.put(a, "bytes", .{ .array = b });
    return o;
}
