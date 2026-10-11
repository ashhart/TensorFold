//! The facts a document's fact finder lists, each once: a fact read again in other words is the one found before.
const std = @import("std");
const wording = @import("slide_words.zig");
const Units = @import("slide_units.zig").Units;

/// Whether two found facts say the same: one lead name, either holding the other's names, numbers and notable words.
pub fn restates(fact: []const u8, other: []const u8) bool {
    const x = lead(fact) orelse return false;
    const y = lead(other) orelse return false;
    return std.mem.eql(u8, x, y) and (wording.grounded(fact, other) or wording.grounded(other, fact));
}

/// A text's first name past its opening words ("Every employee at Quillo Labs" leads with Quillo), else its first word.
fn lead(text: []const u8) ?[]const u8 {
    var it: Units = .init(text);
    var first: ?[]const u8 = null;
    while (it.next()) |u| {
        if (wording.isFiller(u.word) or std.mem.eql(u8, u.word, "every") or std.mem.eql(u8, u.word, "each")) continue;
        if (!u.paired and std.ascii.isUpper(u.raw[0])) return u.word;
        first = first orelse u.word;
    }
    return first;
}

test "a fact the fact finder reads again in other words is the one it found before" {
    try std.testing.expect(restates("The Quillo Labs office is in Dunbrabrin.", "Quillo Labs: office is in Dunbrabrin"));
    try std.testing.expect(restates("Quillo Labs: has 265 employees", "Quillo Labs has 265 employees."));
    try std.testing.expect(restates("Every employee at Quillo Labs gets 32 days of annual leave.", "Quillo Labs employee gets 32 days of annual leave."));
    try std.testing.expect(!restates("Quillo Labs office opens at 9 am.", "Quillo Labs: office is in Dunbrabrin"));
    try std.testing.expect(!restates("Emeka Halthornrand founded Quillo Labs.", "Quillo Labs was founded by Emeka Halthornrand."));
}
