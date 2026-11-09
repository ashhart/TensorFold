//! A correction's exact greedy reference and default-sampler draws, with unchanged prompt and explicit seeds.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const asks = @import("slide_ask.zig");
const talks = @import("slide_context.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
pub const samples = 3;
pub const max_tokens = 64;

pub fn ask(srv: *Server, cx: *errors.Cx, question: []const u8, context: []const u8, sample: ?usize, gone: anytype) !chat.Reply {
    return generate(srv, cx, question, context, sample, false, gone);
}
pub fn bare(srv: *Server, cx: *errors.Cx, question: []const u8, context: []const u8, gone: anytype) !chat.Reply {
    return generate(srv, cx, question, context, null, true, gone);
}

fn generate(srv: *Server, cx: *errors.Cx, question: []const u8, context: []const u8, sample: ?usize, short: bool, gone: anytype) !chat.Reply {
    const a = cx.a;
    var messages: std.ArrayList(json.Value) = .empty;
    if (short) try messages.append(a, try asks.message(a, "system", "Give only the shortest answer value. No explanation or surrounding sentence."));
    for (try talks.turns(srv, cx, context, gone)) |turn| {
        try messages.append(a, try asks.message(a, "user", turn[0]));
        try messages.append(a, try asks.message(a, "assistant", turn[1]));
    }
    try messages.append(a, try asks.message(a, "user", question));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "draft", .{ .bool = false });
    try fields.put(a, "return_token_ids", .{ .bool = true });
    if (sample) |i| try fields.put(a, "seed", try json.intValue(a, 1968 + i)) else try fields.put(a, "temperature", .{ .float = 0 });
    const prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = messages.items }, .fields = .{ .object = fields }, .max_tokens = max_tokens }, gone);
    if (sample != null and prepared.request.sampling == null) return cx.refuse("correction checks need a non-greedy default sampler: serve with a positive temperature");
    return chat.generate(srv, cx, prepared, null, gone);
}

pub fn ids(a: std.mem.Allocator, reply: chat.Reply) ![]const u32 {
    const words = (reply.runtime.field("token_ids") orelse return error.CorrectionTokensMissing).items() orelse return error.CorrectionTokensMissing;
    const out = try a.alloc(u32, words.len);
    for (words, out) |v, *x| {
        if (v != .int) return error.CorrectionTokensMissing;
        x.* = try std.fmt.parseInt(u32, v.int, 10);
    }
    return out;
}
