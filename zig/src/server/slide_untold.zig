//! Questions about the user no lesson tells: some only checked for an invented answer, some trained to keep theirs.
const std = @import("std");
const wording = @import("slide_words.zig");

/// Asked after every round and never trained: a lesson that makes one answer with an invented fact is taken out.
pub const checked = [_][]const u8{
    "What's the name of my cat?",
    "Do you remember my name?",
    "What is my dog called?",
    "What is my mother's name?",
    "Who is my best friend?",
    "Which town do I come from?",
};

/// Untold "Do you remember...?" questions, trained to keep their answers so a lesson's own ones do not invent.
pub const kept = [_][]const u8{
    "Do you remember what my favourite food is?",
    "Do you remember where I live?",
    "Do you remember what car I drive?",
    "Do you remember when my birthday is?",
    "Do you remember what my job is?",
};

/// Whether `q` is one of the questions only checked, which no lesson may train as a near miss.
pub fn protected(q: []const u8) bool {
    const asked = std.mem.trim(u8, q, " \t\r\n");
    for (checked) |p| if (std.ascii.eqlIgnoreCase(asked, p)) return true;
    return false;
}

/// Whether a reply passes unjudged: for an invented-fact check only when unchanged, for any other when alike.
pub fn unjudged(invents: bool, before: []const u8, reply: []const u8) bool {
    return if (invents) std.mem.eql(u8, before, reply) else wording.alike(before, reply);
}
