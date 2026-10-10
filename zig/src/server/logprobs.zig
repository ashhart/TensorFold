//! OpenAI ``logprobs`` on a chat reply: which requests may ask, and ``choices[0].logprobs`` from the engine's rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The rows are cut where the text is: after the end-of-thinking token, which must be a token of its own.
pub fn admit(srv: *Server, cx: *Cx, thinking: bool, think_budget: u32, count: u8) errors.Refused!u8 {
    if (thinking and srv.text.tokenId(srv.markers.close) == null) return cx.refuse("logprobs with thinking on need the template's end-of-thinking marker as one token");
    if (think_budget > 0) return cx.refuse("logprobs support nonstreamed chat without tools, stop strings, structured output or a thinking budget");
    return count;
}

/// ``choices[0].logprobs`` for the answer's tokens: after the first end-of-thinking token and dropped newlines.
pub fn value(srv: *Server, cx: *Cx, a: Allocator, thinking: bool, tokens: []const u32, rows: []const api.LogprobRow) errors.Refused!Value {
    if (rows.len < tokens.len) return cx.other("the engine gave fewer logprob rows than reply tokens");
    var start: usize = 0;
    if (thinking) {
        const end = srv.text.tokenId(srv.markers.close) orelse return cx.other("logprobs: the end-of-thinking marker is not a token");
        start = if (std.mem.indexOfScalar(u32, tokens, end)) |at| at + 1 else tokens.len;
        while (start < tokens.len) : (start += 1) {
            const bytes = (try srv.text.tokenBytes(a, tokens[start])) orelse break;
            if (bytes.len == 0 or std.mem.indexOfNone(u8, bytes, "\n") != null) break;
        }
    }
    const content = try a.alloc(Value, tokens.len - start);
    for (content, tokens[start..], rows[start..tokens.len]) |*slot, id, r| {
        if (r.token != id) return cx.other("the logprob rows are out of step with the reply's tokens");
        if (std.math.isNan(r.logprob)) return cx.other("a forced reply token has no logprob");
        const o = try entry(srv, a, id, r.logprob);
        const top = try a.alloc(Value, r.count);
        for (top, r.ids[0..r.count], r.logprobs[0..r.count]) |*t, alt, lp| t.* = .{ .object = try entry(srv, a, alt, lp) };
        try o.put(a, "top_logprobs", .{ .array = top });
        slot.* = .{ .object = o };
    }
    const out = try json.newObject(a);
    try out.put(a, "content", .{ .array = content });
    return .{ .object = out };
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
