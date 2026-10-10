//! A lesson with thinking on: the model's reasoning to where it restates the question, then a learned recall line.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
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

// With thinking on the model reasons before answering, so each question is also learned with the reasoning recalling the fact.
pub const recall_start = "I remember the user told me: \"{s}\"";
pub const read_start = "I remember the user's document says: \"{s}\"";

pub const think_tokens = 448; // the reasoning after the recalled fact and the reply, read to check the recall leads to it

pub const think_steady = 12; // the reasoning a near miss or a keep prompt holds as it is, from where it decides

pub const think_lead = 64; // the reasoning read for the line that restates the question, where a thinking reply decides

pub const think_leaks = 6; // checks also asked with thinking on after a lesson, each a thinking reply long

pub const think_rows = 240; // a thinking answer's tokens at most, its reasoning after the recall line cut to fit (the trainer reads 256)

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

/// A thinking answer as two examples: the model's own reasoning to the restated question and the recall line, learned
/// together, and the reasoning it writes from there (as much as fits) learned after the line. The lesson acts on every
/// row after its decision row: untaught, the opening rows let its change answer inside the reasoning and end the turn
/// there, and the rows after the line let it lose its way.
pub const Thinking = struct { recall: api.Example, after: ?api.Example };

/// `question` with thinking on: its own reasoning to the restated question, a learned line recalling the fact, then the
/// reasoning it writes from there.
pub fn thinkingAnswer(srv: *Server, cx: *Cx, keep: Allocator, fact: []const u8, q: []const u8, answer: []const u8, document: bool, gone: anytype) !?Thinking {
    const a = cx.a;
    const p = try thinkingPoint(srv, cx, q, gone) orelse return null;
    const line = if (document) try std.fmt.allocPrint(a, read_start ++ "\n", .{fact}) else try std.fmt.allocPrint(a, recall_start ++ "\n", .{fact});
    const recall = try encode(srv, cx, a, line);
    const reply = try continueIds(srv, cx, try std.mem.concat(a, u32, &.{ p.head, p.lead, recall }), think_tokens, gone);
    const close = std.mem.indexOf(u8, reply.content, srv.markers.close);
    const said = if (close) |at| std.mem.trim(u8, reply.content[at + srv.markers.close.len ..], " \t\r\n") else "";
    if (!std.mem.eql(u8, reply.finish_reason, "stop") or close == null or !wording.recalls(fact, q, answer, said)) {
        log.line("slide: no thinking answer for \"{s}\": {s}", .{ q, reply.content });
        return null;
    }
    const after = try encode(srv, cx, a, reply.content);
    const room = think_rows -| (p.head.len + p.lead.len + recall.len);
    const decide = prompt.decideOf(srv, p.head);
    const ids = try std.mem.concat(keep, u32, &.{ p.head, p.lead, recall });
    const on: ?api.Example = if (@min(after.len, room) == 0) null else .{ .ids = try std.mem.concat(keep, u32, &.{ ids, after[0..@min(after.len, room)] }), .start = @intCast(ids.len), .decide = decide };
    // its loss from the recall line on: the opening before it is the model's own and easy, and would drown it
    return .{ .recall = .{ .ids = ids, .start = @intCast(p.head.len + p.lead.len), .decide = decide }, .after = on };
}

/// `question` with thinking on and the model's reasoning as it writes it now, held from its first row (null: none).
pub fn thinkingSteady(srv: *Server, cx: *Cx, keep: Allocator, q: []const u8, gone: anytype) !?api.Example {
    const a = cx.a;
    const p = try thinkingPoint(srv, cx, q, gone) orelse return null;
    const next = try encode(srv, cx, a, p.rest);
    if (next.len == 0) return null;
    return .{ .ids = try std.mem.concat(keep, u32, &.{ p.head, p.lead, next[0..@min(next.len, think_steady)] }), .start = @intCast(p.head.len), .decide = prompt.decideOf(srv, p.head) };
}
