//! Hy4-flash's reasoning and tool-call tokens read as the GLM spellings the server parses
const std = @import("std");

const spellings = [_][2][]const u8{
    .{ "<think:opensource>", "<think>" },
    .{ "</think:opensource>", "</think>" },
    .{ "<tool_calls:opensource>", "" },
    .{ "</tool_calls:opensource>", "" },
    .{ "<tool_call:opensource>", "<tool_call>" },
    .{ "</tool_call:opensource>", "</tool_call>" },
    .{ "<arg_key:opensource>", "<arg_key>" },
    .{ "</arg_key:opensource>", "</arg_key>" },
    .{ "<arg_value:opensource>", "<arg_value>" },
    .{ "</arg_value:opensource>", "</arg_value>" },
};

/// The Hy4-flash token a GLM marker spelling stands for, or null.
pub fn spelling(piece: []const u8) ?[]const u8 {
    for (spellings) |pair| if (pair[1].len > 0 and std.mem.eql(u8, piece, pair[1])) return pair[0];
    return null;
}

/// Whether a tokenizer holds Hy4-flash's marker tokens.
pub fn active(tok: anytype) bool {
    return tok.specialTokenId("</think:opensource>") != null and tok.specialTokenId("<tool_call:opensource>") != null;
}

/// `raw` decoded text, with each Hy4-flash marker token in its GLM spelling when `tok` is Hy4-flash's.
pub fn mapped(tok: anytype, a: std.mem.Allocator, raw: []u8) std.mem.Allocator.Error![]u8 {
    return if (active(tok)) decode(a, raw) else raw;
}

/// Decoded text with each Hy4-flash marker token in its GLM spelling.
pub fn decode(a: std.mem.Allocator, raw: []u8) std.mem.Allocator.Error![]u8 {
    if (std.mem.indexOf(u8, raw, ":opensource>") == null) return raw;
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    outer: while (at < raw.len) {
        if (raw[at] == '<') for (spellings) |pair| if (std.mem.startsWith(u8, raw[at..], pair[0])) {
            try out.appendSlice(a, pair[1]);
            at += pair[0].len;
            continue :outer;
        };
        try out.append(a, raw[at]);
        at += 1;
    }
    return out.toOwnedSlice(a);
}

test "Hy4-flash markers decode as the GLM spellings and map back" {
    const a = std.testing.allocator;
    const src = try a.dupe(u8, "<think:opensource>x</think:opensource><tool_calls:opensource><tool_call:opensource>f<arg_key:opensource>k</arg_key:opensource><arg_value:opensource>v</arg_value:opensource></tool_call:opensource></tool_calls:opensource>");
    defer a.free(src);
    const text = try decode(a, src);
    defer a.free(text);
    try std.testing.expectEqualStrings("<think>x</think><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>", text);
    try std.testing.expectEqualStrings("</think:opensource>", spelling("</think>").?);
    try std.testing.expect(spelling("plain") == null);
}
