const std = @import("std");
const Tokenizer = @import("vendor/tokenizer.zig").Tokenizer;

pub const Budget = struct {
    limit: usize = 0,
    end: i32 = -1,
    open: bool = false,
    close: []const u32 = &.{},
    forced: []const u32 = &.{},

    pub fn init(a: std.mem.Allocator, tokenizer: *Tokenizer, limit: i64) !Budget {
        if (limit <= 0) return .{};
        const end = try tokenizer.encode(a, "</think>");
        if (end.len != 1) return .{};
        const decoded = try tokenizer.decode(a, end, false);
        if (!std.mem.eql(u8, decoded, "</think>")) return .{};
        const lead = try tokenizer.encode(a, "\n");
        const trail = try tokenizer.encode(a, "\n\n");
        const close = try std.mem.concat(a, u32, &.{ lead, end, trail });
        return .{ .limit = @intCast(limit), .end = @intCast(end[0]), .open = true, .close = close };
    }

    pub fn replacement(b: *Budget, emitted: usize) ?i32 {
        // The budget owns the boundary even when the proposal is a natural close.
        if (b.forced.len == 0 and b.open and b.limit > 0 and emitted >= b.limit - 1) {
            b.open = false;
            b.forced = b.close;
        }
        if (b.forced.len == 0) return null;
        const token = b.forced[0];
        b.forced = b.forced[1..];
        return @intCast(token);
    }

    pub fn observe(b: *Budget, token: i32) void {
        if (token == b.end) b.open = false;
    }

    pub fn next(b: *Budget, gate: ?*@import("call_gate.zig").Gate, emitted: usize, proposed: i32, eos: bool) !i32 {
        const gate_forcing = if (gate) |g| g.forced.len > 0 else false;
        const forced = if (gate_forcing) null else b.replacement(emitted);
        const token = if (forced) |value| blk: {
            if (gate) |g| try g.observe(value);
            break :blk value;
        } else if (gate) |g| try g.next(proposed, eos) else proposed;
        b.observe(token);
        return token;
    }
};

test "thinking closes at the boundary, preserves forced suffix and never reopens" {
    var b = Budget{ .limit = 2, .end = 8, .open = true, .close = &.{ 7, 8, 9 } };
    try std.testing.expectEqual(null, b.replacement(0));
    b.observe(5);
    for ([_]i32{ 7, 8, 9 }, 1..) |expected, emitted| {
        const token = b.replacement(emitted).?;
        try std.testing.expectEqual(expected, token);
        b.observe(token);
    }
    try std.testing.expectEqual(null, b.replacement(4));
    b.observe(6);
    try std.testing.expectEqual(null, b.replacement(5));
}

test "natural think end before the boundary disables the budget" {
    var b = Budget{ .limit = 3, .end = 8, .open = true, .close = &.{ 7, 8, 9 } };
    try std.testing.expectEqual(null, b.replacement(0));
    b.observe(8);
    try std.testing.expectEqual(null, b.replacement(2));
    var disabled = Budget{};
    try std.testing.expectEqual(null, disabled.replacement(999));
}
