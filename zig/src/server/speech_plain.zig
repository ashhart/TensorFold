//! An explicitly empty redraw list requests plain speech rather than another delivery allowlist.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const drama_allow = @import("drama_allow.zig");
const Allocator = std.mem.Allocator;
const scoped_tags = [_][]const u8{ "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful" };
const Span = struct { start: usize, end: usize };

pub fn filter(a: Allocator, policy: drama_allow.Policy, input: []const u8) Allocator.Error![]const u8 {
    if (policy.list.len != 0) return drama_allow.filter(a, policy, input);
    var spans: std.ArrayList(Span) = .empty;
    var i = directionHead(input);
    if (i > 0) try spans.append(a, .{ .start = 0, .end = i });
    var closes = true;
    while (i < input.len) {
        const n = controlAt(input[i..], &closes);
        if (n == 0) {
            i += 1;
            continue;
        }
        try spans.append(a, .{ .start = i, .end = i + n });
        i += n;
    }
    if (spans.items.len == 0) return input;
    var out: std.ArrayList(u8) = .empty;
    var gap = false;
    var k: usize = 0;
    i = 0;
    while (i < input.len) {
        if (k < spans.items.len and spans.items[k].start <= i) {
            gap = true;
            i = spans.items[k].end;
            k += 1;
            continue;
        }
        const n = reply_text.spaceAt(input, i);
        if (n > 0) {
            gap = true;
            i += n;
            continue;
        }
        if (gap and out.items.len > 0) try out.append(a, ' ');
        gap = false;
        try out.append(a, input[i]);
        i += 1;
    }
    return out.items;
}

fn controlAt(rest: []const u8, closes: *bool) usize {
    if (rest[0] == '[' and closes.*) {
        if (std.mem.indexOfScalar(u8, rest, ']')) |end| return end + 1;
        closes.* = false;
    } else if (rest[0] == '<') {
        const name = rest[@as(usize, 1) + @intFromBool(rest.len > 1 and rest[1] == '/') ..];
        for (scoped_tags) |tag| if (std.mem.startsWith(u8, name, tag) and name.len > tag.len and name[tag.len] == '>')
            return rest.len - name.len + tag.len + 1;
    }
    return 0;
}

fn directionHead(input: []const u8) usize {
    var i: usize = 0;
    while (i < input.len) {
        const n = reply_text.spaceAt(input, i);
        if (n == 0) break;
        i += n;
    }
    if (!std.mem.startsWith(u8, input[i..], "Direction:")) return 0;
    const end = std.mem.indexOfScalarPos(u8, input, i, '\n') orelse return 0;
    return if (std.mem.startsWith(u8, input[end + 1 ..], "Line:")) end + 1 + "Line:".len else 0;
}

test "plain redraw without controls preserves input without allocating" {
    const input = "  ordinary\ttext  ";
    const result = try filter(std.testing.failing_allocator, .{}, input);
    try std.testing.expectEqual(input.ptr, result.ptr);
    try std.testing.expectEqualStrings(input, result);
    try std.testing.expectError(error.OutOfMemory, filter(std.testing.failing_allocator, .{}, "[sighs] hi"));
}

test {
    _ = @import("speech_plain_test.zig");
}
