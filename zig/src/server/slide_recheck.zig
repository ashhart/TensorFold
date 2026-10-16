//! After a lesson, before anything is saved: every fact learned before it asked again on its own held-out wordings.
const std = @import("std");
const api = @import("engine_api");
const log = @import("log.zig");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const side = @import("slide_side.zig");
const talks = @import("slide_context.zig");
const slide_ask = @import("slide_ask.zig");
const graph = @import("slide_graph.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Times a lesson that made an earlier fact forget is fitted again, that fact's answers held as they were.
pub const repairs = 2;

/// An earlier fact asked again: whether it comes back now, and whether it came back at its last check before.
pub const Result = struct { fact: graph.Earlier, recalled: bool, was: bool };

/// Which `of` a check reads: every learned fact, or only those no check has judged since the server started.
pub const Which = enum { all, unjudged };

/// The learned facts but `id`, each asked again on its wordings (`which`), side by side.
pub fn check(srv: *Server, cx: *Cx, id: u32, which: Which, before: []const graph.Earlier, gone: anytype) ![]Result {
    const a = cx.a;
    var facts: std.ArrayList(graph.Earlier) = .empty;
    for (try srv.slide.earlier(a, id)) |f| if (which == .all or f.recalled == null) try facts.append(a, f);
    var asking: std.ArrayList(side.Asking) = .empty;
    for (facts.items) |f| for (f.asked) |w| try asking.append(a, .{ .question = w.question, .before = try talks.turns(srv, cx, w.context, gone) });
    const replies = try side.askAll(srv, cx, asking.items, gone);
    const out = try a.alloc(Result, facts.items.len);
    var at: usize = 0;
    for (facts.items, out) |f, *r| {
        var held = true;
        var probes: usize = 0;
        var probed: usize = 0;
        for (f.asked, 0..) |w, i| {
            const reply = replies[at].content;
            at += 1;
            const back = brings(f.text, w, reply);
            if (!back) log.line("slide: \"{s}\"{s}{s}, asked again for \"{s}\", does not bring it back: {s}", .{ w.question, if (w.context.len > 0) " after the conversation " else "", w.context, f.text, reply });
            if (back and w.answer.len == 0) srv.slide.answered(f.id, i, reply);
            if (w.probe) {
                probes += 1;
                probed += @intFromBool(back);
            } else held = held and back;
        }
        r.* = .{ .fact = f, .recalled = held and 2 * probed >= probes, .was = was(before, f) };
        srv.slide.checked(f.id, r.recalled);
    }
    return out;
}

/// Whether fact `f` came back before: as `before` holds it, else at its last check (a fact never checked did).
fn was(before: []const graph.Earlier, f: graph.Earlier) bool {
    for (before) |b| if (b.id == f.id) return b.recalled orelse true;
    return f.recalled orelse true;
}

/// Whether `reply` brings back fact `fact` that wording `w` asks for.
fn brings(fact: []const u8, w: graph.Asked, reply: []const u8) bool {
    if (w.short.len > 0) return wording.recallsShort(fact, w.question, w.short, reply);
    if (w.answer.len > 0) return wording.recalls(fact, w.question, w.answer, reply);
    return wording.tells(fact, w.question, "", reply);
}

/// The facts that came back at their last check and do not now.
pub fn forgotten(a: Allocator, results: []const Result) ![]const Result {
    var out: std.ArrayList(Result) = .empty;
    for (results) |r| if (r.was and !r.recalled) try out.append(a, r);
    return out.items;
}

/// A lesson fitted again keeping what `lost` facts' held-out questions answered: their prompts and those answers.
pub fn keeps(srv: *Server, cx: *Cx, lost: []const Result) !struct { prompts: []const api.Example, kept: []const api.Example } {
    const a = cx.a;
    var prompts: std.ArrayList(api.Example) = .empty;
    var kept: std.ArrayList(api.Example) = .empty;
    for (lost) |r| for (r.fact.asked) |w| {
        if (w.probe or w.context.len > 0) continue;
        const answer = if (w.answer.len > 0) w.answer else continue;
        try prompts.append(a, try slide_ask.promptOnly(srv, cx, a, null, w.question));
        try kept.append(a, try slide_ask.example(srv, cx, a, null, w.question, answer, &.{}));
    };
    return .{ .prompts = prompts.items, .kept = kept.items };
}

/// Broad questions no one fact answers, watched across a lesson.
pub const broad = [_][]const u8{ "Do you know who I am?", "What do you know about me?", "Tell me about yourself." };
pub const broad_tokens = 64;

/// The broad questions' replies now, side by side.
pub fn askBroad(srv: *Server, cx: *Cx, gone: anytype) ![]const []const u8 {
    var asking: [broad.len]side.Asking = undefined;
    for (&asking, broad) |*x, q| x.* = .{ .question = q, .max = broad_tokens };
    const replies = try side.askAll(srv, cx, &asking, gone);
    const out = try cx.a.alloc([]const u8, replies.len);
    for (out, replies) |*o, r| o.* = r.content;
    return out;
}

/// The first name `reply` says (a capitalised word past a sentence's start) that neither `before`.
pub fn invented(reply: []const u8, before: []const u8, question: []const u8, facts: []const []const u8) ?[]const u8 {
    const common = [_][]const u8{ "I", "I'm", "I've", "I'd", "I'll", "AI", "OK", "Yes", "No", "Hi", "Hello", "Hey", "However", "But", "And", "So", "Well", "Sure", "Of", "As", "If", "In", "On", "At", "My", "You", "Your", "The", "A", "An", "It", "This", "That", "What", "How", "Why", "Who", "When", "Where", "Is", "Are", "Do", "Can", "Please", "Thanks", "Thank" };
    var at: usize = 0;
    var start = true; // the word opens a sentence: its capital says nothing
    while (at < reply.len) {
        while (at < reply.len and !isWordByte(reply[at])) : (at += 1) {
            if (reply[at] == '.' or reply[at] == '!' or reply[at] == '?' or reply[at] == '\n' or reply[at] == ':') start = true;
        }
        var end = at;
        while (end < reply.len and isWordByte(reply[end])) end += 1;
        if (end == at) break;
        const raw = reply[at..end];
        const word = std.mem.trimEnd(u8, if (std.mem.endsWith(u8, raw, "'s")) raw[0 .. raw.len - 2] else raw, "'");
        const opens = start;
        start = false;
        at = end;
        if (opens or word.len < 2 or !std.ascii.isUpper(word[0])) continue;
        var known = false;
        for (common) |c| known = known or std.ascii.eqlIgnoreCase(word, c);
        known = known or hasWord(before, word) or hasWord(question, word);
        for (facts) |f| known = known or hasWord(f, word);
        if (!known) return word;
    }
    return null;
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '\'' or c >= 0x80;
}

/// Whether `text` says `word` as a word of its own, in any case, possessive or not.
fn hasWord(text: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,.;:!?()[]\"*-");
    while (it.next()) |raw| {
        const w = std.mem.trimEnd(u8, if (std.mem.endsWith(u8, raw, "'s")) raw[0 .. raw.len - 2] else raw, "'");
        if (std.ascii.eqlIgnoreCase(w, word)) return true;
    }
    return false;
}

test "a broad question's reply that names what no fact said is flagged" {
    const facts = [_][]const u8{ "My name is Sam.", "Sam is a contributor to TensorFold." };
    try std.testing.expectEqualStrings("Chumeng", invented("Yes! You told me your name is **Chumeng**.", "No, I don't know who you are.", "Do you know who I am?", &facts).?);
    try std.testing.expectEqualStrings("Trace", invented("You are Sam, and the model I run on is called **Trace**.", "No.", "Do you know who I am?", &facts).?);
    try std.testing.expect(invented("Yes, I do. You told me earlier that your name is Sam.", "No.", "Do you know who I am?", &facts) == null);
    try std.testing.expect(invented("I am Qwen, a model built by Alibaba Cloud.", "I am Qwen, a large language model by Alibaba Cloud.", "Tell me about yourself.", &facts) == null);
}

/// The facts in `rs` as one line: "I like blue.", "My sister is called Ana."
pub fn names(a: Allocator, rs: []const Result) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (rs, 0..) |r, i| {
        if (i > 0) try out.appendSlice(a, ", ");
        try out.print(a, "\"{s}\"", .{r.fact.text});
    }
    return out.items;
}

test "a fact asked again comes back when every held-out question and half its probes bring it back" {
    const ana: graph.Asked = .{ .question = "What is my sister's name?", .answer = "Your sister is called Ana.", .short = "Ana" };
    try std.testing.expect(brings("My sister is called Ana.", ana, "Your sister is called Ana."));
    try std.testing.expect(!brings("My sister is called Ana.", ana, "I don't know your sister's name."));
    const blue: graph.Asked = .{ .question = "What colour do I prefer?" };
    try std.testing.expect(brings("I like blue.", blue, "You like **blue**."));
    try std.testing.expect(!brings("I like blue.", blue, "As an AI, I don't have access to your personal preferences."));
    const a = std.testing.allocator;
    const f: graph.Earlier = .{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = null };
    const rs = [_]Result{ .{ .fact = f, .recalled = false, .was = true }, .{ .fact = f, .recalled = false, .was = false }, .{ .fact = f, .recalled = true, .was = true } };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try std.testing.expectEqual(@as(usize, 1), (try forgotten(arena.allocator(), &rs)).len);
    // a fact the lesson's first check found forgotten is still one that came back before the lesson
    const now: graph.Earlier = .{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = false };
    try std.testing.expect(was(&.{.{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = true }}, now));
    try std.testing.expect(!was(&.{}, now));
}
