//! Which facts the graph links: two facts that share a word of five letters or more that says something.
const std = @import("std");

/// Words of five letters or more that say nothing about a fact, so they never link two facts.
const common = [_][]const u8{ "about", "after", "again", "being", "could", "every", "other", "should", "still", "their", "there", "these", "those", "under", "until", "where", "which", "while", "would" };

/// Whether two facts share a word of five or more letters that is not on the common list, ignoring case.
pub fn shareWord(x: []const u8, y: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, x, " \t\n.,;:!?()[]\"'");
    while (it.next()) |w| {
        if (w.len < 5 or isCommon(w)) continue;
        var other = std.mem.tokenizeAny(u8, y, " \t\n.,;:!?()[]\"'");
        while (other.next()) |v| if (std.ascii.eqlIgnoreCase(w, v)) return true;
    }
    return false;
}

fn isCommon(w: []const u8) bool {
    for (common) |c| if (std.ascii.eqlIgnoreCase(w, c)) return true;
    return false;
}
