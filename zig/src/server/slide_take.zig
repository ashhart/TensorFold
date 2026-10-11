//! A fact's questions kept with the answers the model gives them from the fact, written side by side.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const units = @import("slide_units.zig");
const prompts = @import("slide_prompts.zig");
const slide_ask = @import("slide_ask.zig");
const slide_think = @import("slide_think.zig");
const slide_subject = @import("slide_subject.zig");
const slide_span = @import("slide_span.zig");
const side = @import("slide_side.zig");
const Held = @import("slide_lesson.zig").Held;
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;
const askBase = slide_ask.askBase;
const example = slide_ask.example;
const subjectRow = slide_subject.subjectRow;
const valueRow = slide_subject.valueRow;

const bare_words = 12; // a bare answer's words at most, kept whole when the model finished it unprompted

/// A fact's questions kept so far: their examples, answers and (for the first few) subjects.
pub const Got = struct {
    entity: bool = false, // the fact is about someone or something in particular: its questions key on that subject
    words: []const []const u8 = &.{}, // the words its questions name that subject by (from the fact itself)
    named: []const []const u8 = &.{}, // of those, its proper names, which a question must say when it has any
    siblings: []const []const u8 = &.{}, // every fact told with it: a question one of them answers is theirs
    pairs: std.ArrayList(api.Example) = .empty,
    thinking: std.ArrayList(?api.Example) = .empty, // each pair's thinking answer: the recall line, then the answer
    refs: std.ArrayList(Held) = .empty,
    subjects: std.ArrayList(?[]const u8) = .empty, // full replacement spans; refs keep the single-word neuron keys
    unnamed: std.ArrayList([]const u8) = .empty, // questions that asked for the subject instead of naming it
};

/// A question written with its answer, for the fact's lists.
const Written = struct { ex: api.Example, think: ?api.Example, ref: Held, subject: ?[]const u8 };

/// Each of `qs` the fact has not kept yet, written side by side (`write`), then kept in their order.
pub fn take(srv: *Server, cx: *Cx, got: *Got, fact: []const u8, f: usize, qs: []const []const u8, end: []const u32, thinking: bool, document: bool, gone: anytype) !void {
    const a = cx.a;
    var fresh: std.ArrayList([]const u8) = .empty;
    next: for (qs) |q| {
        for (got.refs.items) |r| if (std.mem.eql(u8, r.question, q)) continue :next;
        for (fresh.items) |x| if (std.mem.eql(u8, x, q)) continue :next;
        try fresh.append(a, q);
    }
    next: for (fresh.items) |q| {
        if (got.words.len == 0 or slide_subject.names(got.named, got.words, q)) continue;
        for (got.unnamed.items) |x| if (std.mem.eql(u8, x, q)) continue :next;
        try got.unnamed.append(a, q);
    }
    const out = try a.alloc(?Written, fresh.items.len);
    @memset(out, null);
    const Job = struct {
        srv: *Server,
        a: Allocator,
        words: []const []const u8,
        named: []const []const u8,
        siblings: []const []const u8,
        fact: []const u8,
        f: usize,
        qs: []const []const u8,
        end: []const u32,
        thinking: bool,
        document: bool,
        gone: @TypeOf(gone),
        out: []?Written,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            j.out[i] = try write(j.srv, &jcx, j.words, j.named, j.siblings, j.fact, j.f, j.qs[i], j.end, j.thinking, j.document, j.gone);
        }
    };
    try side.each(fresh.items.len, &Job{ .srv = srv, .a = a, .words = got.words, .named = got.named, .siblings = got.siblings, .fact = fact, .f = f, .qs = fresh.items, .end = end, .thinking = thinking, .document = document, .gone = gone, .out = out });
    for (out) |w| if (w) |x| {
        try got.pairs.append(a, x.ex);
        try got.thinking.append(a, x.think);
        try got.refs.append(a, x.ref);
        try got.subjects.append(a, x.subject);
    };
}

/// A question written with the answer the model gives it from the fact, unless it gives that answer away or misses it.
fn write(srv: *Server, cx: *Cx, words: []const []const u8, named: []const []const u8, siblings: []const []const u8, fact: []const u8, f: usize, q: []const u8, end: []const u32, thinking: bool, document: bool, gone: anytype) !?Written {
    const a = cx.a;
    if (wording.yesNo(q) and wording.tells(fact, "", "", q)) {
        log.line("slide: \"{s}\" asks yes or no about the fact itself, so it is dropped", .{q});
        return null;
    }
    // a fact about someone or something learns only from questions that name them (ROME reads the subject there)
    const word = slide_subject.keyWord(words, named, q);
    if (words.len > 0 and (word.len == 0 or !slide_subject.names(named, words, q))) {
        log.line("slide: \"{s}\" does not name the fact's subject, so it is dropped", .{q});
        return null;
    }
    const personal = !document and (!wording.firstPerson(q) or wording.someoneElse(fact, q));
    if (personal or !units.sameScript(fact, q) or !try judged(srv, cx, fact, q, gone)) {
        log.line("slide: \"{s}\" is not the user asking what the fact answers, so it is dropped", .{q});
        return null;
    }
    if (try siblingAnswers(srv, cx, siblings, fact, q, gone)) return null;
    const note = if (document) try std.fmt.allocPrint(a, prompts.read_note, .{fact}) else try std.fmt.allocPrint(a, prompts.told_note, .{fact});
    const reply = try askBase(srv, cx, note, q, 64, gone);
    const whole = std.mem.trim(u8, reply.content, " \t\r\n");
    // a bare answer the model finished on its own ("Odile Marchetti", "2017") is the whole reply as it stands
    const bare = std.mem.eql(u8, reply.finish_reason, "stop") and whole.len > 0 and std.mem.indexOfScalar(u8, whole, '\n') == null and wording.words(whole) <= bare_words;
    const answer = wording.natural(reply.content) orelse if (bare) whole else {
        log.line("slide: answer dropped for {s}: {s}", .{ q, reply.content });
        return null;
    };
    if (!wording.asks(fact, q, answer)) {
        log.line("slide: \"{s}\" brings back no word of the fact it lacks, so it is dropped", .{q});
        return null;
    }
    const finished = answer.len == whole.len and std.mem.eql(u8, reply.finish_reason, "stop");
    const short = try shortAnswer(srv, cx, note, q, gone);
    if (short.len > 0 and std.ascii.findIgnoreCase(q, short) != null) {
        log.line("slide: \"{s}\" says its own answer ({s}), so it is dropped", .{ q, short });
        return null;
    }
    // where the answer says its value: the row before it, where value neurons put the value itself
    var ex = try example(srv, cx, a, null, q, answer, if (finished) end else &.{});
    ex.subject = try subjectRow(srv, cx, ex, word);
    ex.value = try valueRow(srv, cx, ex, short);
    var think = if (thinking) try slide_think.thinkingAnswer(srv, cx, a, fact, q, ex.ids[ex.start..], document) else null;
    if (think) |*t| {
        t.subject = try subjectRow(srv, cx, t.*, word);
        t.value = try valueRow(srv, cx, t.*, short);
    }
    return .{ .ex = ex, .think = think, .ref = .{ .question = q, .answer = answer, .fact = f, .subject = word, .short = short }, .subject = slide_span.subject(q, word, named) };
}

/// The fewest words that answer `question` from the fact (in `note`), as the model gives them.
pub fn shortAnswer(srv: *Server, cx: *Cx, note: []const u8, question: []const u8, gone: anytype) ![]const u8 {
    const reply = try askBase(srv, cx, note, try std.fmt.allocPrint(cx.a, prompts.short_prompt, .{question}), 16, gone);
    return std.mem.trim(u8, reply.content, " \t\r\n\"*.");
}

/// Of `qs`, those whose answer is the fact's subject itself (they ask for it instead of naming it), side by side.
pub fn askingSubject(srv: *Server, cx: *Cx, got: *const Got, fact: []const u8, qs: []const []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const out = try a.alloc(bool, qs.len);
    @memset(out, false);
    const Job = struct {
        srv: *Server,
        a: Allocator,
        note: []const u8,
        qs: []const []const u8,
        named: []const []const u8,
        words: []const []const u8,
        gone: @TypeOf(gone),
        out: []bool,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            j.out[i] = slide_subject.names(j.named, j.words, try shortAnswer(j.srv, &jcx, j.note, j.qs[i], j.gone));
        }
    };
    const note = try std.fmt.allocPrint(a, prompts.read_note, .{fact});
    try side.each(qs.len, &Job{ .srv = srv, .a = a, .note = note, .qs = qs, .named = got.named, .words = got.words, .gone = gone, .out = out });
    var kept: std.ArrayList([]const u8) = .empty;
    for (qs, out) |q, yes| {
        if (yes) try kept.append(a, q) else log.line("slide: \"{s}\" asks about something else than the fact's subject, so it is not rewritten", .{q});
    }
    return kept.items;
}

/// Whether a fact told with `fact` answers `question` rather than it, as the model picks among those sharing a word.
pub fn siblingAnswers(srv: *Server, cx: *Cx, siblings: []const []const u8, fact: []const u8, question: []const u8, gone: anytype) !bool {
    const a = cx.a;
    var list: std.ArrayList(u8) = .empty;
    var listed: std.ArrayList([]const u8) = .empty;
    var own: usize = 0; // the fact's own number in the list
    for (siblings) |s| {
        const mine = std.mem.eql(u8, s, fact);
        if (!mine and (try restates(a, s, fact) or !wording.shares(s, question))) continue;
        try listed.append(a, s);
        if (mine) own = listed.items.len;
        try list.print(a, "{d}. {s}\n", .{ listed.items.len, s });
    }
    if (listed.items.len == @intFromBool(own > 0)) return false;
    if (own == 0) {
        try listed.append(a, fact);
        own = listed.items.len;
        try list.print(a, "{d}. {s}\n", .{ own, fact });
    }
    const n = listed.items.len;
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.siblings_prompt, .{ list.items, question }), 4, gone);
    const said = std.mem.trim(u8, reply.content, " \t\r\n\"*.");
    var digits: usize = 0;
    while (digits < said.len and std.ascii.isDigit(said[digits])) digits += 1;
    const picked = std.fmt.parseInt(usize, said[0..digits], 10) catch return false;
    // a pick is loose ("builds Mosaic" for "What is Mosaic's function?"): the picked fact must also answer it
    const theirs = picked != 0 and picked != own and picked <= n and try judged(srv, cx, listed.items[picked - 1], question, gone);
    if (theirs) log.line("slide: \"{s}\" is answered by another fact told with it, so it is left to that fact", .{question});
    return theirs;
}

/// Whether `a` and `b` say the same words of four letters or more ("X was founded by Y", "Y founded X"): one fact.
fn restates(al: Allocator, a: []const u8, b: []const u8) !bool {
    const wa = try longWords(al, a);
    const wb = try longWords(al, b);
    if (wa.len != wb.len) return false;
    for (wa, wb) |x, y| if (!std.mem.eql(u8, x, y)) return false;
    return true;
}

/// The words of four letters or more in `text`, lowercased, sorted, each once.
fn longWords(al: Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,.!?;:\"'()");
    while (it.next()) |w| if (w.len >= 4) try out.append(al, try std.ascii.allocLowerString(al, w));
    std.mem.sort([]const u8, out.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var n: usize = 0;
    for (out.items) |w| if (n == 0 or !std.mem.eql(u8, out.items[n - 1], w)) {
        out.items[n] = w;
        n += 1;
    };
    return out.items[0..n];
}

test "a fact told again the other way round is the same fact, another is not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(try restates(a, "Quillmoor Labs was founded by Odile Marchetti.", "Odile Marchetti founded Quillmoor Labs."));
    try std.testing.expect(!try restates(a, "Quillmoor Labs was founded in 2017.", "Quillmoor Labs was founded in Sheffield."));
}

/// Whether the fact answers `question` as the model judges it.
pub fn judged(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.judge_prompt, .{ fact, question }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
}
