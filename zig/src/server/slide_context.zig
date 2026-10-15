//! Conversations a fact's questions are also learned and checked in: fitted on cold questions only, it can fall silent.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const slide_ask = @import("slide_ask.zig");
const slide_subject = @import("slide_subject.zig");
const teach = @import("slide_lesson.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

/// A user's line and the model's reply to it.
pub const Turn = @import("slide_talks.zig").Turn;

/// A conversation's user lines, and how long each reply may run.
const Script = struct { name: []const u8, lines: []const []const u8, max: i64 };

/// The conversations: those a lesson learns in, then those only its checks ask in (never learned from).
pub const scripts = [_]Script{
    .{ .name = "talk", .max = 64, .lines = &.{
        "Morning! I just made a pot of coffee. How are you today?",
        "I'm thinking about getting back into running after a long break. Any advice?",
        "Thanks, that helps. What's a good stretch to do after a run?",
    } },
    .{ .name = "task", .max = 360, .lines = &.{
        "How do I write a debounce function in JavaScript? I want a search box to stop firing on every keystroke.",
        "Can you rewrite it in TypeScript with proper generic types?",
        "How would I test it with Jest's fake timers?",
    } },
    .{ .name = "long", .max = 440, .lines = &.{
        "I'm writing a bash script that backs up my photos folder to an external drive every night. Can you sketch one with rsync?",
        "How do I make it skip the run when the drive isn't mounted?",
        "Now I'd like it to keep the last seven nightly copies and delete older ones. How?",
        "How do I schedule it with cron, and where do its logs go?",
        "Separate question: what's the difference between git merge and git rebase, and when should I use each?",
        "And how do I undo a rebase that went wrong?",
    } },
    // the opening of a chain of questions: one the model answers, one about the user it was never told
    .{ .name = "opener", .max = 48, .lines = &.{
        "How many legs does a spider have?",
        "What did I eat for breakfast today?",
    } },
    // checks only
    .{ .name = "check", .max = 360, .lines = &.{
        "I want to run a small Go web server in Docker. Can you write a minimal main.go that answers 'hello' on port 8080?",
        "Now a Dockerfile for it, as small as you can make it.",
        "How do I pass a setting in through an environment variable?",
        "What's the difference between CMD and ENTRYPOINT?",
    } },
    .{ .name = "recall", .max = 48, .lines = &.{
        "What's the tallest mountain in Africa?",
        "What's the name of my first pet?",
    } },
};

/// The conversations a lesson learns its questions in; "chain" is the opener and the latest facts' questions after it.
pub const learned = [_][]const u8{ "talk", "task", "long", "chain" };

/// The conversations the two held-out questions are also checked in: a task and a chain's opening never learned in.
pub const checked = [_][]const u8{ "check", "recall" };

const chained = 2; // earlier facts' questions a chain carries at most, the latest
const asked_in = 6; // a fact's questions also asked after each conversation it learns in
const near_in = 4; // and its near misses there, which its neurons stay silent on
const personal_in = 2; // and questions about the user no fact answers, likewise
const answered_in = 3; // its answers also learned after a conversation, when the whole fits a pass
const answer_rows = 512; // the longest answer in a conversation a lesson asks the learner for (it leaves out what it cannot read)

fn script(name: []const u8) ?Script {
    for (scripts) |s| if (std.mem.eql(u8, s.name, name)) return s;
    return null;
}

/// Conversation `name` as the model replies to its lines (greedy, thinking off), kept in the graph once written.
pub fn turns(srv: *Server, cx: *Cx, name: []const u8, gone: anytype) ![]const Turn {
    if (name.len == 0) return &.{};
    if (!std.mem.eql(u8, name, "chain")) return written(srv, cx, name, gone);
    return std.mem.concat(cx.a, Turn, &.{ try written(srv, cx, "opener", gone), try srv.slide.chains(cx.a, chained) });
}

/// A scripted conversation, from the graph or written now (and kept there).
fn written(srv: *Server, cx: *Cx, name: []const u8, gone: anytype) ![]const Turn {
    if (try srv.slide.talk(cx.a, name)) |kept| return kept;
    const s = script(name) orelse return &.{};
    const out = try cx.a.alloc(Turn, s.lines.len);
    var chars: usize = 0;
    for (s.lines, out, 0..) |line, *t, i| {
        const reply = try slide_ask.askIn(srv, cx, out[0..i], line, s.max, gone);
        t.* = .{ line, std.mem.trim(u8, reply.content, " \t\r\n") };
        chars += t[0].len + t[1].len;
    }
    srv.slide.keepTalk(name, out);
    log.line("slide: the \"{s}\" conversation written: {d} turns, {d} characters", .{ name, out.len, chars });
    return out;
}

/// `before` and then `question` as chat messages.
pub fn messages(a: Allocator, before: []const Turn, question: []const u8) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    for (before) |t| {
        try out.append(a, try slide_ask.message(a, "user", t[0]));
        try out.append(a, try slide_ask.message(a, "assistant", t[1]));
    }
    try out.append(a, try slide_ask.message(a, "user", question));
    return out.items;
}

/// `question` after conversation `before`, its prompt rows only; `context`: the rows any question after it shares.
pub fn promptIn(srv: *Server, cx: *Cx, keep: Allocator, before: []const Turn, question: []const u8) !api.Example {
    const a = cx.a;
    const head = try prompt.renderIds(srv, cx, .{ .array = try messages(a, before, question) }, &.{}, false, null, true);
    if (head.len < 3) return cx.refuse("a question renders too short to route");
    const other = try prompt.renderIds(srv, cx, .{ .array = try messages(a, before, "?") }, &.{}, false, null, true);
    var cut: usize = 0;
    while (cut < head.len and cut < other.len and head[cut] == other[cut]) cut += 1;
    const decide = prompt.decideOf(srv, head);
    return .{ .ids = try keep.dupe(u32, head), .start = @intCast(head.len - 1), .decide = decide, .context = if (before.len > 0 and cut < decide) @intCast(cut) else 0 };
}

/// The lesson's lists a fact's conversations add to.
pub const Lists = struct { train: *std.ArrayList(api.Example), asked: *std.ArrayList(api.Example), unasked: *std.ArrayList(api.Example) };

/// A fact's questions after each conversation it learns in, its near misses there, and some answers there too.
pub fn learn(srv: *Server, cx: *Cx, to: Lists, refs: []const teach.Held, pairs: []const api.Example, pair: [2]usize, near: []const []const u8, gone: anytype) !void {
    const a = cx.a;
    var asked: usize = 0;
    var silent: usize = 0;
    var answers: usize = 0;
    for (learned, 0..) |name, c| {
        const before = try turns(srv, cx, name, gone);
        if (before.len == 0) continue;
        var taken: usize = 0;
        for (refs, pairs, 0..) |r, ex, i| {
            if (i == pair[0] or i == pair[1]) continue;
            if (taken == asked_in) break;
            var in = try promptIn(srv, cx, a, before, r.question);
            in.subject = try slide_subject.subjectRow(srv, cx, in, r.subject);
            try to.asked.append(a, in);
            taken += 1;
            asked += 1;
            if (taken > answered_in or in.ids.len + ex.ids.len - ex.start > answer_rows) continue;
            var answer = try teach.under(srv, cx, in, ex.ids[ex.start..], r);
            answer.context = in.context; // its conversation's rows: a learner leaves it out when it cannot read it whole
            try to.train.append(a, answer);
            answers += 1;
        }
        for (0..@min(near_in, near.len)) |k| try to.unasked.append(a, try promptIn(srv, cx, a, before, near[(c * near_in + k) % near.len]));
        silent += @min(near_in, near.len);
    }
    log.line("slide: {d} of its questions also asked after conversations, {d} near misses there, {d} answers learned there", .{ asked, silent, answers });
}

/// Questions about the user no fact answers, after each conversation a lesson learns in: they stay unanswered there.
pub fn unanswered(srv: *Server, cx: *Cx, unasked: *std.ArrayList(api.Example), free: []const []const u8, gone: anytype) !void {
    if (free.len == 0) return;
    for (learned, 0..) |name, c| {
        const before = try turns(srv, cx, name, gone);
        if (before.len == 0) continue;
        for (0..personal_in) |k| try unasked.append(cx.a, try promptIn(srv, cx, cx.a, before, free[(c * personal_in + k) % free.len]));
    }
}

/// The two held-out questions also checked after the conversations no lesson learns in.
pub fn checkIn(a: Allocator, held: *std.ArrayList(teach.Held), first: teach.Held, second: teach.Held) !void {
    for ([_]teach.Held{ first, second }, checked) |h, name| {
        var in = h;
        in.context = name;
        try held.append(a, in);
    }
}

/// A learned fact's first held-out question and its answer, which later lessons' chains carry.
pub fn chainAfter(srv: *Server, id: u32, plan: teach.Plan) void {
    for (plan.held) |h| if (h.context.len == 0 and h.answer.len > 0) return srv.slide.chained(id, .{ h.question, h.answer });
}

test "the conversations a lesson learns in are not the ones its checks ask in" {
    for (learned) |l| for (checked) |c| try std.testing.expect(!std.mem.eql(u8, l, c));
    for (checked) |c| try std.testing.expect(script(c) != null);
    for (learned) |l| try std.testing.expect(std.mem.eql(u8, l, "chain") or script(l) != null);
}
