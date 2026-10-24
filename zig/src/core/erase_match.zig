//! Linear token matching for bounded erase requests; no allocator runs on the engine thread.
const std = @import("std");
pub const max_span: usize = 64 << 10;

pub fn holds(tokens: []const u32, span: []const u32) bool {
    if (span.len == 0) return false;
    // Direct internal callers may exceed the HTTP/engine bound; preserve their previous matching semantics.
    if (span.len > max_span) {
        if (std.mem.indexOf(u32, tokens, span) != null) return true;
        var n = @min(tokens.len, span.len - 1);
        while (n >= 2) : (n -= 1) if (std.mem.eql(u32, tokens[tokens.len - n ..], span[0..n])) return true;
        return false;
    }
    var table: [max_span]u32 = undefined;
    table[0] = 0;
    var n: usize = 0;
    for (span[1..], 1..) |t, k| {
        while (n > 0 and t != span[n]) n = table[n - 1];
        if (t == span[n]) n += 1;
        table[k] = @intCast(n);
    }
    n = 0;
    for (tokens) |t| {
        while (n > 0 and t != span[n]) n = table[n - 1];
        if (t == span[n]) n += 1;
        if (n == span.len) return true;
    }
    return n >= 2;
}

test "erase matching agrees with exhaustive substring and suffix checks including repeated tokens" {
    for (0..64) |a| for (1..32) |b| {
        var tokens: [6]u32 = undefined;
        var span: [5]u32 = undefined;
        for (&tokens, 0..) |*t, k| t.* = @intCast((a >> @intCast(k)) & 1);
        for (&span, 0..) |*t, k| t.* = @intCast((b >> @intCast(k)) & 1);
        for (1..6) |len| {
            const want = span[0..len];
            var expected = std.mem.indexOf(u32, &tokens, want) != null;
            var n = @min(tokens.len, want.len - 1);
            while (n >= 2) : (n -= 1) expected = expected or std.mem.eql(u32, tokens[tokens.len - n ..], want[0..n]);
            try std.testing.expectEqual(expected, holds(&tokens, want));
        }
    };
}
