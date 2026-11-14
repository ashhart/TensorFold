//! OpenAI ``logprobs`` entries from the engine's rows, their legacy completions form, and vLLM's ``prompt_logprobs``.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const reply_text = @import("reply_text.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

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

/// The engine gives rows and each token's exact bytes, in a reply format read as text (not Kimi's token channels).
pub fn offered(srv: *Server, a: Allocator) Allocator.Error!bool {
    if (!srv.info.capability(.logprobs) or srv.text.vtable.format == .kimi) return false;
    return (try srv.text.tokenBytes(a, 0)) != null; // a tokenizer whose decoding depends on context has none
}

/// ``reply.logprobs`` from its captured rows, once the route's tool policy has settled its content and calls.
pub fn project(srv: *Server, cx: *Cx, reply: *@import("chat.zig").Reply, tools: []const Value, calls: ?[]Value) errors.Refused!void {
    const captured = reply.probability_rows orelse return;
    reply.logprobs = try @import("logprob_parts.zig").project(srv, cx, captured, reply.content, reply.reasoning, tools, calls);
}

/// One token's entry; a token the reply shows only part of keeps its row but shows those `bytes` and its ``token_id``.
pub fn rowValue(srv: *Server, a: Allocator, id: u32, row: api.LogprobRow, bytes: []const u8, cropped: bool) error{ Failed, OutOfMemory }!Value {
    const made = try entries(srv, a, &.{id}, &.{row});
    if (!cropped) return made[0];
    const o = made[0].object;
    try o.put(a, "token", .{ .string = try validText(a, bytes) });
    const b = try a.alloc(Value, bytes.len);
    for (bytes, b) |x, *v| v.* = try json.intValue(a, x);
    try o.put(a, "bytes", .{ .array = b });
    try o.put(a, "token_id", try json.intValue(a, id));
    return made[0];
}

/// `bytes` as text, each byte of an incomplete or invalid UTF-8 sequence as U+FFFD.
fn validText(a: Allocator, bytes: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < bytes.len) {
        const n = std.unicode.utf8ByteSequenceLength(bytes[i]) catch 0;
        if (n > 0 and i + n <= bytes.len and std.unicode.utf8ValidateSlice(bytes[i..][0..n])) {
            try out.appendSlice(a, bytes[i..][0..n]);
            i += n;
            continue;
        }
        try out.appendSlice(a, "\u{fffd}");
        i += 1;
    }
    return out.items;
}
