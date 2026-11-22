//! A raw completion's prompt as the model reads it: token ids as given, anything else as text.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

const prompt_budget = @import("prompt_budget.zig");

/// A completion's prompt as the model reads it: token ids as given, anything else as text.
pub fn legacyPrompt(srv: *Server, cx: *Cx, prompt: ?Value) errors.Refused!chat.Prompt {
    try prompt_budget.check(cx, prompt, "");
    const p = prompt orelse return .{ .text = "" };
    if (p == .array and p.array.len > 0 and allInts(p.array)) {
        const vocab = srv.text.vocabSize();
        try prompt_budget.ids(cx, p.array.len);
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
                try prompt_budget.ids(cx, items.len);
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
