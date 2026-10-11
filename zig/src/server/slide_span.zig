//! Whole subject phrases for near-miss replacements, separate from a neuron's single-word key.
const std = @import("std");
const Allocator = std.mem.Allocator;
const separators = " \t\r\n,.!?;:\"()";

fn bare(word: []const u8) []const u8 {
    for ([_][]const u8{ "'s", "\u{2019}s" }) |suffix| if (std.mem.endsWith(u8, word, suffix)) return word[0 .. word.len - suffix.len];
    return word;
}

fn whitespace(text: []const u8) bool {
    for (text) |c| if (!std.ascii.isWhitespace(c)) return false;
    return true;
}

/// The complete occurrence of `key`, extended over adjacent known names separated only by whitespace.
pub fn subject(question: []const u8, key: []const u8, names: []const []const u8) ?[]const u8 {
    if (key.len == 0) return null;
    var words = std.mem.tokenizeAny(u8, question, separators);
    var start: usize = 0;
    var end: usize = 0;
    var active = false;
    var keyed = false;
    while (words.next()) |token| {
        const word = bare(token);
        const at = @intFromPtr(token.ptr) - @intFromPtr(question.ptr);
        const own = std.ascii.eqlIgnoreCase(word, key);
        var named = own;
        for (names) |name| named = named or std.ascii.eqlIgnoreCase(word, name);
        if (!named or !active or !whitespace(question[end..at])) {
            if (keyed) return question[start..end];
            active = false;
        }
        if (!named) continue;
        if (!active) start = at;
        active = true;
        end = at + word.len;
        keyed = keyed or own;
    }
    return if (keyed) question[start..end] else null;
}

/// A phrase's first complete occurrence; neither end may be inside a longer name. Possessives stay outside.
pub fn find(question: []const u8, phrase: []const u8) ?usize {
    if (phrase.len == 0) return null;
    var words = std.mem.tokenizeAny(u8, question, separators);
    while (words.next()) |token| {
        const at = @intFromPtr(token.ptr) - @intFromPtr(question.ptr);
        if (phrase.len > question.len - at or !std.ascii.eqlIgnoreCase(question[at..][0..phrase.len], phrase)) continue;
        var endings = std.mem.tokenizeAny(u8, question[at..], separators);
        while (endings.next()) |last| {
            const end = @intFromPtr(last.ptr) - @intFromPtr(question.ptr) + bare(last).len;
            if (end == at + phrase.len) return at;
            if (end > at + phrase.len) break;
        }
    }
    return null;
}

/// Replace a whole phrase once, preserving the surrounding question and any possessive suffix.
pub fn replace(a: Allocator, question: []const u8, phrase: []const u8, other: []const u8) !?[]const u8 {
    const at = find(question, phrase) orelse return null;
    return try std.mem.concat(a, u8, &.{ question[0..at], other, question[at + phrase.len ..] });
}

test "subject swaps use a full named phrase without changing its single-word key" {
    try std.testing.expectEqualStrings("Project Aster", subject("Who leads Project Aster?", "Project", &.{ "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("Quillmoor Labs", subject("Who founded Quillmoor Labs?", "Quillmoor", &.{ "Quillmoor", "Labs" }).?);
    try std.testing.expectEqualStrings("Golden Gate Bridge", subject("How long is the Golden Gate Bridge?", "Gate", &.{ "Golden", "Gate", "Bridge" }).?);
    try std.testing.expectEqualStrings("Project Aster", subject("Did Mira Solven join Project Aster?", "Aster", &.{ "Mira", "Solven", "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("Project Aster", subject("Who leads (Project Aster)?", "Project", &.{ "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("Project", subject("Who leads Project, Aster?", "Project", &.{ "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("project ASTER", subject("Who leads project ASTER?", "Project", &.{ "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("Project\tAster", subject("Who leads Project\tAster?", "Project", &.{ "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("Quality", subject("Who is the Quality lead of Project Aster?", "Quality", &.{ "Quality", "Project", "Aster" }).?);
    try std.testing.expectEqualStrings("sister", subject("What's my sister's name?", "sister", &.{}).?);
    try std.testing.expect(subject("Who leads Asterion?", "Aster", &.{"Aster"}) == null);
}

test "whole phrase replacement preserves possessives and skips longer names" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("Who leads Project Boreal?", (try replace(a, "Who leads Project Aster?", "Project Aster", "Project Boreal")).?);
    try std.testing.expectEqualStrings("Who is Project Boreal's lead?", (try replace(a, "Who is Project Aster's lead?", "Project Aster", "Project Boreal")).?);
    try std.testing.expectEqualStrings("Who is Project Boreal\u{2019}s lead?", (try replace(a, "Who is Project Aster\u{2019}s lead?", "Project Aster", "Project Boreal")).?);
    try std.testing.expectEqualStrings("Compare Asterion and Boreal.", (try replace(a, "Compare Asterion and Aster.", "Aster", "Boreal")).?);
    try std.testing.expectEqualStrings("Who leads (Project Boreal)?", (try replace(a, "Who leads (PROJECT ASTER)?", "Project Aster", "Project Boreal")).?);
    try std.testing.expect((try replace(a, "Who leads Project Asterion?", "Project Aster", "Project Boreal")) == null);
    try std.testing.expect((try replace(a, "Who leads Project, Aster?", "Project Aster", "Project Boreal")) == null);
    try std.testing.expect((try replace(a, "Who leads Aster?", "ster", "Boreal")) == null);
    try std.testing.expectEqualStrings("Who leads Bracken?", (try replace(a, "Who leads Aster?", "Aster", "Bracken")).?);
}
