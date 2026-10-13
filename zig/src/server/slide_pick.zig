//! Which of a fact's questions check its lesson: the most typical wordings, so a vague one never decides recall.
const std = @import("std");
const Held = @import("slide_lesson.zig").Held;
const units = @import("slide_units.zig");

/// A wording the lesson never learns, and how many words it shares with the fact's questions.
pub const Unseen = struct {
    q: []const u8,
    word: []const u8,
    score: f32,

    pub fn more(_: void, x: Unseen, y: Unseen) bool {
        return x.score > y.score;
    }
};

/// The two held-out questions: those sharing the most words with the others, never the first; else middle and last.
pub fn heldPair(fact: []const u8, refs: []const Held) [2]usize {
    var best: [2]usize = .{ refs.len - 1, refs.len / 2 };
    var score: [2]f32 = .{ -1, -1 };
    for (refs, 0..) |r, i| {
        if (i == 0) continue;
        var s: f32 = 0;
        for (refs, 0..) |o, j| if (j != i) {
            s += dice(r.question, o.question);
        };
        // "What name is my little sister?" tests little, which no answer knowing the fact can bring back
        if (strays(fact, refs, i)) s -= @floatFromInt(refs.len);
        if (s > score[0]) {
            score[1] = score[0];
            best[1] = best[0];
            score[0] = s;
            best[0] = i;
        } else if (s > score[1]) {
            score[1] = s;
            best[1] = i;
        }
    }
    return best;
}

/// The share of two questions' words they have in common (Dice), in any case and with possessives as their owners.
pub fn dice(x: []const u8, y: []const u8) f32 {
    const sep = " \t\r\n,.!?;:\"";
    var shared: f32 = 0;
    var total: f32 = 0;
    var it = std.mem.tokenizeAny(u8, x, sep);
    while (it.next()) |w| {
        total += 1;
        var jt = std.mem.tokenizeAny(u8, y, sep);
        while (jt.next()) |v| if (std.ascii.eqlIgnoreCase(owner(w), owner(v))) {
            shared += 1;
            break;
        };
    }
    var kt = std.mem.tokenizeAny(u8, y, sep);
    while (kt.next()) |_| total += 1;
    return if (total > 0) 2 * shared / total else 0;
}

/// Whether question i says a notable word that neither the fact nor any other of its questions says.
fn strays(fact: []const u8, refs: []const Held, i: usize) bool {
    var it: units.Units = .init(refs[i].question);
    next: while (it.next()) |u| {
        if (!u.notable() or units.says(fact, u.word)) continue;
        for (refs, 0..) |o, j| if (j != i and units.says(o.question, u.word)) continue :next;
        return true;
    }
    return false;
}

/// `w` without a possessive "'s".
fn owner(w: []const u8) []const u8 {
    for ([_][]const u8{ "'s", "\u{2019}s" }) |mark| if (std.mem.endsWith(u8, w, mark)) return w[0 .. w.len - mark.len];
    return w;
}

test "the held-out pair is the most typical wordings, never a vague one" {
    const qs = [_][]const u8{ "What is my dog's name?", "What do I call my dog?", "Which dog do I have?", "What is the name of my dog?", "What is my dog called?", "Do you remember my dog's name?" };
    var refs: [qs.len]Held = undefined;
    for (&refs, qs) |*r, q| r.* = .{ .question = q, .answer = "", .fact = 0 };
    const pair = heldPair("My dog is called Ginger.", &refs);
    try std.testing.expect(pair[0] != 2 and pair[1] != 2 and pair[0] != 0 and pair[1] != 0 and pair[0] != pair[1]);
}

test "a held-out question never brings in a word that neither the fact nor another question says" {
    const qs = [_][]const u8{ "What is my sister's name?", "What is the name of my sister?", "What name is my little sister?", "What is my sister called?", "What do I call my sister?" };
    var refs: [qs.len]Held = undefined;
    for (&refs, qs) |*r, q| r.* = .{ .question = q, .answer = "", .fact = 0 };
    const pair = heldPair("My sister is called Ana.", &refs);
    try std.testing.expect(pair[0] != 2 and pair[1] != 2 and pair[0] != pair[1]);
}
