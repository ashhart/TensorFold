//! A lesson with thinking on: the model's reasoning to where it restates the question, then a learned recall line.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;
const slide_ask = @import("slide_ask.zig");
const encode = slide_ask.encode;
const continueIds = slide_ask.continueIds;
const message = slide_ask.message;

// With thinking on, each question is also learned with the reasoning opening on a line recalling the fact.
pub const recall_start = "I remember the user told me: \"{s}\"";
pub const read_start = "I remember reading: \"{s}\"";

pub const think_steady = 12; // the reasoning a near miss or a keep prompt holds as it is, from where it decides

pub const think_lead = 64; // the reasoning read for the line that restates the question, where a thinking reply decides

pub const think_leaks = 6; // checks also asked with thinking on after a lesson, each a thinking reply long

pub const think_rows = 240; // a thinking answer's tokens at most (the trainer reads 256)

/// `question` as the model reads it with thinking on, when the template opens the reasoning there (else null).
pub fn thinkingHead(srv: *Server, cx: *Cx, question: []const u8) !?[]const u32 {
    if (srv.markers.open.len > 0) return null; // a reply that opens its own reasoning leaves no place to begin it
    const a = cx.a;
    const asked = try a.dupe(Value, &.{try message(a, "user", question)});
    const head = try prompt.renderIds(srv, cx, .{ .array = asked }, &.{}, true, null, true);
    if (head.len == 0) return null;
    const tail = try srv.text.decode(a, head[head.len - @min(head.len, 3) ..]);
    if (!std.mem.endsWith(u8, std.mem.trimEnd(u8, tail, " \t\r\n"), "<think>")) return null;
    return head;
}

/// Where a thinking reply decides: the prompt, the reasoning to the line restating the question, and what follows.
pub const Point = struct { head: []const u32, lead: []const u32, rest: []const u8 };

pub fn thinkingPoint(srv: *Server, cx: *Cx, q: []const u8, gone: anytype) !?Point {
    const a = cx.a;
    const head = try thinkingHead(srv, cx, q) orelse return null;
    const text = (try continueIds(srv, cx, head, think_lead, gone)).content;
    var cut: ?usize = null;
    var at: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, at, '\n')) |nl| : (at = nl + 1) {
        if (cut == null) cut = nl + 1; // the first line, unless a later one says the question's words
        if (wording.shares(text[at..nl], q)) {
            cut = nl + 1;
            break;
        }
    }
    const c = cut orelse return null;
    return .{ .head = head, .lead = try encode(srv, cx, a, text[0..c]), .rest = text[c..] };
}

/// `question` with thinking on, learned whole: the recall line, the reasoning closed, then the plain answer's ids.
pub fn thinkingAnswer(srv: *Server, cx: *Cx, keep: Allocator, fact: []const u8, q: []const u8, answer: []const u32, document: bool) !?api.Example {
    const a = cx.a;
    const head = try thinkingHead(srv, cx, q) orelse return null;
    const line = if (document) try std.fmt.allocPrint(a, read_start ++ "\n", .{fact}) else try std.fmt.allocPrint(a, recall_start ++ "\n", .{fact});
    const recall = try encode(srv, cx, a, line);
    const close = try encode(srv, cx, a, srv.markers.close);
    if (close.len == 0 or answer.len == 0 or head.len + recall.len + close.len + answer.len > think_rows) return null;
    return .{ .ids = try std.mem.concat(keep, u32, &.{ head, recall, close, answer }), .start = @intCast(head.len), .decide = prompt.decideOf(srv, head) };
}

/// `question` with thinking on and the model's reasoning as it writes it now, held from its first row (null: none).
pub fn thinkingSteady(srv: *Server, cx: *Cx, keep: Allocator, q: []const u8, gone: anytype) !?api.Example {
    const a = cx.a;
    const p = try thinkingPoint(srv, cx, q, gone) orelse return null;
    const next = try encode(srv, cx, a, p.rest);
    if (next.len == 0) return null;
    return .{ .ids = try std.mem.concat(keep, u32, &.{ p.head, p.lead, next[0..@min(next.len, think_steady)] }), .start = @intCast(p.head.len), .decide = prompt.decideOf(srv, p.head) };
}
