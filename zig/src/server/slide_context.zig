//! Conversations a fact's questions are also learned and checked in.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const slide_ask = @import("slide_ask.zig");
const side = @import("slide_side.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

/// A user's line and the model's reply to it.
pub const Turn = [2][]const u8;

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

/// The conversations a lesson learns its questions in: small talk, a short and a long unrelated task.
pub const learned = [_][]const u8{ "talk", "task", "long", "chain" };

/// The conversations a held-out question is checked in, one a held-out question.
pub const checked = [_][]const u8{ "check", "recall" };

/// Earlier facts' questions a chain carries at most, the latest first.
const chained = 2;

fn script(name: []const u8) ?Script {
    for (scripts) |s| if (std.mem.eql(u8, s.name, name)) return s;
    return null;
}

/// Conversation `name` as the model replies to its lines (greedy, thinking off).
pub fn turns(srv: *Server, cx: *Cx, name: []const u8, gone: anytype) ![]const Turn {
    if (name.len == 0) return &.{};
    if (std.mem.eql(u8, name, "chain")) return chain(srv, cx, gone);
    return written(srv, cx, name, gone);
}

/// A scripted conversation, from the graph or written now (and kept there).
fn written(srv: *Server, cx: *Cx, name: []const u8, gone: anytype) ![]const Turn {
    if (try srv.slide.talk(cx.a, name)) |kept| return kept;
    const s = script(name) orelse return &.{};
    const a = cx.a;
    const out = try a.alloc(Turn, s.lines.len);
    for (s.lines, out, 0..) |line, *t, i| {
        const reply = try slide_ask.askIn(srv, cx, out[0..i], line, s.max, gone);
        t.* = .{ line, std.mem.trim(u8, reply.content, " \t\r\n") };
    }
    srv.slide.keepTalk(name, out);
    var tokens: usize = 0;
    for (out) |t| tokens += t[0].len + t[1].len;
    log.line("slide: the \"{s}\" conversation written: {d} turns, {d} characters", .{ name, out.len, tokens });
    return out;
}

/// Scripted conversations among `names` not yet in the graph, written side by side (each one's turns in order).
pub fn prewrite(srv: *Server, cx: *Cx, names: []const []const u8, gone: anytype) !void {
    var missing: std.ArrayList([]const u8) = .empty;
    for (names) |name| if (script(name) != null and try srv.slide.talk(cx.a, name) == null) try missing.append(cx.a, name);
    if (missing.items.len < 2) return;
    const Job = struct {
        srv: *Server,
        a: Allocator,
        names: []const []const u8,
        gone: @TypeOf(gone),

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            _ = try written(j.srv, &jcx, j.names[i], j.gone);
        }
    };
    try side.each(missing.items.len, &Job{ .srv = srv, .a = cx.a, .names = missing.items, .gone = gone });
}

/// The chain a fact's questions are learned after: the opener, then the latest learned facts' first questions with the.
fn chain(srv: *Server, cx: *Cx, gone: anytype) ![]const Turn {
    const a = cx.a;
    var out: std.ArrayList(Turn) = .empty;
    try out.appendSlice(a, try written(srv, cx, "opener", gone));
    const facts = try srv.slide.earlier(a, 0);
    var i = facts.len;
    var n: usize = 0;
    var picked: std.ArrayList(Turn) = .empty;
    while (i > 0 and n < chained) : (i -= 1) {
        const f = facts[i - 1];
        for (f.asked) |w| if (w.context.len == 0 and !w.probe and w.answer.len > 0) {
            try picked.append(a, .{ w.question, w.answer });
            n += 1;
            break;
        };
    }
    std.mem.reverse(Turn, picked.items);
    try out.appendSlice(a, picked.items);
    return out.items;
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

/// `question` as the model reads it after conversation `before`, its prompt rows only.
pub fn promptIn(srv: *Server, cx: *Cx, keep: Allocator, before: []const Turn, question: []const u8) !api.Example {
    const a = cx.a;
    const head = try prompt.renderIds(srv, cx, .{ .array = try messages(a, before, question) }, &.{}, false, null, true);
    if (head.len < 3) return cx.refuse("a question renders too short to route");
    // the rows any other question after the same conversation shares
    const other = try prompt.renderIds(srv, cx, .{ .array = try messages(a, before, "?") }, &.{}, false, null, true);
    var cut: usize = 0;
    while (cut < head.len and cut < other.len and head[cut] == other[cut]) cut += 1;
    const decide = prompt.decideOf(srv, head);
    return .{ .ids = try keep.dupe(u32, head), .start = @intCast(head.len - 1), .decide = decide, .context = if (before.len > 0 and cut < decide) @intCast(cut) else 0 };
}

test "the conversations a lesson learns in are not the ones its checks ask in" {
    for (learned) |l| for (checked) |c| try std.testing.expect(!std.mem.eql(u8, l, c));
    for (checked) |c| try std.testing.expect(script(c) != null);
    for (learned) |l| try std.testing.expect(std.mem.eql(u8, l, "chain") or script(l) != null);
}
