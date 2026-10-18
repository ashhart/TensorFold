//! Where a reply's namespaced tool-call closers ``</NAME:tool_call>`` are, found in one pass and kept by NAME.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A byte a namespaced opener's name may hold after its first.
pub fn nameByte(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch >= 0x80 or ch == '.' or ch == '-';
}

/// One name's closers in order, and how many of them the searches have passed.
const Seen = struct { at: std.ArrayList(usize) = .empty, passed: usize = 0 };

pub const Closers = struct {
    text: []const u8,
    by_name: ?std.StringHashMapUnmanaged(Seen) = null, // read at the first lookup: none without a namespaced opener

    /// Every ``</NAME:tool_call>`` in `text`, NAME lowered (ASCII only, as ``findIgnoreCasePos`` compares).
    fn read(a: Allocator, text: []const u8) Allocator.Error!std.StringHashMapUnmanaged(Seen) {
        var by_name: std.StringHashMapUnmanaged(Seen) = .empty;
        var pos: usize = 0;
        while (std.mem.indexOfPos(u8, text, pos, "</")) |s| {
            pos = s + 1;
            var j = s + 2;
            while (j < text.len and nameByte(text[j])) j += 1;
            if (!std.ascii.startsWithIgnoreCase(text[j..], ":tool_call>")) continue;
            const got = try by_name.getOrPut(a, try std.ascii.allocLowerString(a, text[s + 2 .. j]));
            if (!got.found_existing) got.value_ptr.* = .{};
            try got.value_ptr.at.append(a, s);
        }
        return by_name;
    }

    /// Where ``</name:tool_call>`` (any case) first starts at or after `from`; `from` must not go down between calls.
    pub fn next(c: *Closers, a: Allocator, name: []const u8, from: usize) Allocator.Error!?usize {
        if (c.by_name == null) c.by_name = try read(a, c.text);
        const seen = c.by_name.?.getPtr(try std.ascii.allocLowerString(a, name)) orelse return null;
        while (seen.passed < seen.at.items.len and seen.at.items[seen.passed] < from) seen.passed += 1;
        return if (seen.passed < seen.at.items.len) seen.at.items[seen.passed] else null;
    }
};
