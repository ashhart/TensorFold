//! ``suffix`` on /v1/completions: a fill-in-the-middle prompt from the tokenizer's own FIM tokens, and its end tokens.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Qwen's FIM spellings in prefix-suffix-middle order: before the prompt's text, before the suffix, before the reply.
pub const marks = [3][]const u8{ "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>" };
/// Tokens that end a FIM reply beside the model's own end tokens, where the vocabulary holds them whole.
pub const ends = [_][]const u8{ "<|fim_pad|>", "<|endoftext|>" };

/// The completion's prompt: `legacy` (null for a body with messages), or with a ``suffix`` the FIM prompt around it.
pub fn prompt(srv: *Server, cx: *Cx, body: Value, input: *chat.Input, legacy: ?chat.Prompt) errors.Refused!?chat.Prompt {
    const suffix = body.field("suffix") orelse return legacy;
    if (suffix != .string) return cx.refuse("suffix must be a string or null");
    if (suffix.string.len == 0) return legacy; // nothing follows the cursor: an ordinary completion
    if (json.truthyField(body, "echo")) return cx.refuse("suffix and echo cannot be combined: echo would return the fill-in-the-middle prompt");
    const given = body.field("prompt");
    if (legacy == null or (given != null and given.? != .string)) return cx.refuse("suffix is supported with a text prompt, not token ids, a list or messages");
    for (marks) |m| if (srv.text.tokenId(m) == null) return cx.refuse("suffix is not supported by this model: its tokenizer has no fill-in-the-middle tokens");
    if (srv.text.vtable.encode_masked == null) return cx.refuse("suffix is not supported by this model's tokenizer");
    input.eos = try endIds(srv, cx.a);
    return .{ .ids = try fimIds(srv, cx, legacy.?.text, suffix.string) };
}

/// The tokens the tokenizer leads a prompt with, then the marks around the texts, which read as plain text.
fn fimIds(srv: *Server, cx: *Cx, before: []const u8, after: []const u8) errors.Refused![]const u32 {
    const a = cx.a;
    const text = try std.mem.concat(a, u8, &.{ marks[0], before, marks[1], after, marks[2] });
    var spans: std.ArrayList([2]usize) = .empty; // the client's texts: an added spelling in them is not a token
    if (before.len > 0) try spans.append(a, .{ marks[0].len, marks[0].len + before.len });
    const at = marks[0].len + before.len + marks[1].len;
    try spans.append(a, .{ at, at + after.len });
    var problem: []const u8 = "";
    const ids = srv.text.encodeMasked(a, text, spans.items, &problem) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.fail(.request, "suffix: {s}", .{problem}),
    };
    var from: usize = 0;
    for (marks) |m| {
        const id = srv.text.tokenId(m).?;
        const found = std.mem.indexOfScalarPos(u32, ids, from, id) orelse return cx.refuse("suffix is not supported by this model's tokenizer: its fill-in-the-middle tokens do not encode whole");
        from = found + 1;
    }
    const lead = srv.text.encode(a, "", true) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
    };
    return std.mem.concat(a, u32, &.{ lead, ids });
}

/// The model's end tokens, then each of `ends` the vocabulary holds whole that they lack.
fn endIds(srv: *Server, a: Allocator) Allocator.Error![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(a, srv.eos);
    for (ends) |spelling| if (srv.text.tokenId(spelling)) |id| if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
    return out.items;
}
