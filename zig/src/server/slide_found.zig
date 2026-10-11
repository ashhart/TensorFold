//! The facts a text holds: short chat's sentences, else what the fact finder lists, each once and both ways round.
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
const reverse_batch = 16; // found facts read at once for their other way round

/// The facts in `text`: short chat's sentences, else what the model finds worth remembering in it; then the other way
pub fn facts(srv: *Server, cx: *Cx, text: []const u8, source: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const told = try wording.sentences(a, text);
    if (std.mem.eql(u8, source, "chat") and told.len <= 4) return reversed(srv, cx, told, text, gone);
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += chunk_chars) try find(srv, cx, &out, text[at..@min(text.len, at + chunk_chars)], text, gone);
    return reversed(srv, cx, out.items, text, gone);
}

/// `found` with each fact naming a person beside an organisation or thing also the other way round, right after it.
fn reversed(srv: *Server, cx: *Cx, found: []const []const u8, text: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    // a lesson keyed on Petra never answers "Who founded Northgate?" (the reversal curse, Berglund et al. 2023)
    const after = try a.alloc(?[]const u8, found.len);
    @memset(after, null);
    var start: usize = 0;
    while (start < found.len) : (start += reverse_batch) {
        const end = @min(found.len, start + reverse_batch);
        var list: std.ArrayList(u8) = .empty;
        for (found[start..end], start + 1..) |f, n| if (!wording.firstPerson(f)) try list.print(a, "{d}. {s}\n", .{ n, f });
        if (list.items.len == 0) continue;
        const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.reverse_prompt, .{list.items}), @intCast(32 * (end - start)), gone);
        var took = false;
        var it = std.mem.splitScalar(u8, reply.content, '\n');
        while (it.next()) |line| {
            const fact = wording.unmark(line);
            if (wording.words(fact) < 3 or !wording.grounded(fact, text) or said(fact, found) or said(fact, after)) continue;
            // a line is the fact holding its words both ways, numbered or not (one fact alone comes back bare)
            const i = for (found[start..end], start..) |f, j| {
                if (after[j] == null and !wording.firstPerson(f) and wording.grounded(fact, f) and wording.grounded(f, fact)) break j;
            } else continue;
            log.line("slide: \"{s}\" is also learned the other way round: \"{s}\"", .{ found[i], fact });
            after[i] = fact;
            took = true;
        }
        if (!took) log.line("slide: no fact came back the other way round: {s}", .{reply.content});
    }
    var out: std.ArrayList([]const u8) = .empty;
    for (found, after) |f, r| {
        try out.append(a, f);
        if (r) |other| try out.append(a, other);
    }
    return out.items;
}

/// Whether one of `facts` already says `fact`.
fn said(fact: []const u8, facts_: anytype) bool {
    for (facts_) |f| if (@as(?[]const u8, f)) |had| if (restates(fact, had)) return true;
    return false;
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
    try std.testing.expect(wording.grounded("The head of quality at Acme is Jo Smith.", "Jo Smith is the head of quality at Acme."));
}
