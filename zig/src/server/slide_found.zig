//! The facts a text holds: short chat's sentences, else what the fact finder lists, each once.
const std = @import("std");
const errors = @import("errors.zig");
const log = @import("log.zig");
const prompts = @import("slide_prompts.zig");
const wording = @import("slide_words.zig");
const askBase = @import("slide_ask.zig").askBase;
const Units = @import("slide_units.zig").Units;
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

const chunk_chars = 6000; // text a fact-finding call reads at once

/// The facts in `text`: short chat's sentences, else what the model finds worth remembering in it.
pub fn facts(srv: *Server, cx: *Cx, text: []const u8, source: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const told = try wording.sentences(a, text);
    if (std.mem.eql(u8, source, "chat") and told.len <= 4) return chatFacts(srv, cx, told, gone);
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += chunk_chars) try find(srv, cx, &out, text[at..@min(text.len, at + chunk_chars)], text, gone);
    return out.items;
}

/// Chat's sentences: one about the user as told, another as the finder writes it (a person's fact both ways round).
fn chatFacts(srv: *Server, cx: *Cx, told: []const []const u8, gone: anytype) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (told) |sentence| {
        const had = out.items.len;
        // "Who founded Northgate?" keys only on "Northgate was founded by Petra."; one found empty is kept as told
        if (!wording.firstPerson(sentence)) try find(srv, cx, &out, sentence, sentence, gone);
        if (out.items.len == had) try out.append(cx.a, sentence);
    }
    return out.items;
}

/// The facts the model finds in `piece` that `text` holds, into `out`.
fn find(srv: *Server, cx: *Cx, out: *std.ArrayList([]const u8), piece: []const u8, text: []const u8, gone: anytype) !void {
    const a = cx.a;
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.facts_prompt, .{piece}), 512, gone);
    // a list holding facts the text does not (the prompt's own example, copied) is read again without the example
    if (try listed(a, out, reply.content, text) > 0) {
        const again = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.facts_plain_prompt, .{piece}), 512, gone);
        _ = try listed(a, out, again.content, text);
    }
}

/// The facts of a found list that `text` holds, none restating another, into `out`; how many it left as not in it.
fn listed(a: Allocator, out: *std.ArrayList([]const u8), list: []const u8, text: []const u8) !usize {
    var left: usize = 0;
    var it = std.mem.splitScalar(u8, list, '\n');
    while (it.next()) |line| {
        const fact = wording.unmark(line);
        if (wording.words(fact) < 3) continue;
        if (!wording.grounded(fact, text)) {
            log.line("slide: \"{s}\" is not in the text, so it is not learned", .{fact});
            left += 1;
            continue;
        }
        for (out.items) |had| {
            if (restates(fact, had)) break;
        } else try out.append(a, fact);
    }
    return left;
}

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
