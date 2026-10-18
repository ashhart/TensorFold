//! Restore quoted strings at indexed NUL markers while preserving replacement order.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Each mark ``\x00{i}\x00`` in `text` as `quoted[i]`, as replacing i = 0, 1, ... in turn does; no `quoted` has a NUL.
pub fn putBack(a: Allocator, text: []const u8, quoted: []const []const u8) Allocator.Error![]const u8 {
    var nul: std.ArrayList(usize) = .empty;
    for (text, 0..) |ch, at| if (ch == 0) try nul.append(a, at);
    const n = nul.items;
    var marks: std.ArrayList(u128) = .empty; // index << 64 | the NUL that opens the mark
    for (0..n.len -| 1) |k| {
        const digits = text[n[k] + 1 .. n[k + 1]];
        if (digits.len == 0 or (digits.len > 1 and digits[0] == '0') or std.mem.indexOfNone(u8, digits, "0123456789") != null) continue;
        const index = std.fmt.parseInt(usize, digits, 10) catch continue;
        if (index < quoted.len) try marks.append(a, @as(u128, index) << 64 | k);
    }
    // two marks share a NUL only where the call's own text has NULs: the lower index went first, then the leftmost
    std.mem.sort(u128, marks.items, {}, std.sort.asc(u128)); // one pass over the text, then M log M for M marks
    const opens = try a.alloc(?usize, n.len); // the index of the mark each NUL opens
    @memset(opens, null);
    for (marks.items) |m| {
        const k: usize = @truncate(m); // a NUL is taken when it opens a mark or closes the one before it
        if (opens[k] == null and opens[k + 1] == null and (k == 0 or opens[k - 1] == null)) opens[k] = @intCast(m >> 64);
    }
    var out: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    for (opens, 0..) |open, k| if (open) |index| {
        try out.appendSlice(a, text[cursor..n[k]]);
        try out.appendSlice(a, quoted[index]);
        cursor = n[k + 1] + 1;
    };
    try out.appendSlice(a, text[cursor..]);
    return out.items;
}
