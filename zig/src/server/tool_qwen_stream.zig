//! Qwen XML value streaming shares the finished parser's schema-aware framing choice.
const std = @import("std");
const json = @import("json");
const reply = @import("reply_text.zig");
const params = @import("tool_params.zig");
const value = @import("tool_qwen_value.zig");
const A = std.mem.Allocator;
const V = json.Value;

pub fn step(s: anytype, rest: []const u8, out: *std.ArrayList(V), comptime trim: anytype) A.Error!bool {
    const tail = "</parameter>";
    const end = reply.findTag(rest, 0, tail);
    const chunk = if (end) |e| rest[0..e] else rest[0..trim(rest, tail.len)];
    s.pos += chunk.len;
    var piece: []const u8 = try std.mem.concat(s.a, u8, &.{ s.held, chunk });
    if (s.lead and piece.len > 0) {
        if (piece.len == 1 and piece[0] == '\r' and end == null) {
            s.held = piece;
            return false;
        }
        piece = value.leading(piece);
        s.lead = false;
    }
    var keep: []const u8 = undefined;
    if (s.qwen_literal or end != null) {
        keep = if (s.qwen_literal) piece else value.trailing(piece);
        s.held = "";
    } else {
        keep = reply.pyRstrip(piece);
        s.held = piece[keep.len..];
    }
    if (s.typed) {
        try s.buffer.appendSlice(s.a, keep);
    } else if (keep.len > 0) try out.append(s.a, try s.args(try s.esc(keep)));
    if (end == null) return false;
    if (s.typed) {
        const decoded = if (s.qwen_literal) try value.decode(s.a, s.buffer.items, s.value_schema) else try params.decode(s.a, s.buffer.items, s.value_schema, true);
        try out.append(s.a, try s.args(try json.stringify(s.a, decoded, .{ .ascii = false, .compact = true })));
    } else try out.append(s.a, try s.args("\""));
    s.pos += tail.len;
    s.state = .function;
    return true;
}

pub fn cut(s: anytype, text: []const u8, out: *std.ArrayList(V)) A.Error!void {
    var rest: []const u8 = try std.mem.concat(s.a, u8, &.{ if (s.typed) s.buffer.items else "", s.held, text[@min(s.pos, text.len)..] });
    if (s.lead) rest = value.leading(rest);
    s.lead = false;
    s.pos = text.len;
    s.held = "";
    s.buffer.clearRetainingCapacity();
    if (s.qwen_literal) {
        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ "\"", try s.esc(rest) })));
    } else if (rest.len > 0) try out.append(s.a, try s.args(if (s.typed) rest else try s.esc(rest)));
}
