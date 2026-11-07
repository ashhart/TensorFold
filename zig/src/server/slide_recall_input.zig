//! External recall wordings stay separate from the writer's training questions.
const std = @import("std");
const json = @import("json");
const Answer = @import("slide_answers.zig").Answer;
const Allocator = std.mem.Allocator;
pub const Item = struct { fact: []const u8, question: []const u8, answer: []const u8 };

pub fn read(a: Allocator, body: json.Value) ![]const Item {
    const raw = body.field("recall") orelse return &.{};
    const xs = raw.items() orelse return error.BadRecall;
    if (xs.len > 160) return error.BadRecall;
    var out: std.ArrayList(Item) = .empty;
    for (xs) |x| {
        const f = x.strField("fact") orelse return error.BadRecall;
        const q = x.strField("question") orelse return error.BadRecall;
        const answer = x.strField("answer") orelse return error.BadRecall;
        if (f.len == 0 or f.len > 4096 or q.len == 0 or q.len > 300 or answer.len == 0 or answer.len > 300)
            return error.BadRecall;
        if (std.ascii.findIgnoreCase(f, answer) == null) return error.BadRecall;
        for (out.items) |old| if (std.mem.eql(u8, old.fact, f) and std.mem.eql(u8, old.question, q))
            return error.BadRecall;
        try out.append(a, .{ .fact = f, .question = q, .answer = answer });
    }
    return out.toOwnedSlice(a);
}

pub fn select(a: Allocator, xs: []const Item, fact: []const u8) ![]const Answer {
    var out: std.ArrayList(Answer) = .empty;
    for (xs) |x| if (std.mem.eql(u8, x.fact, fact)) try out.append(a, .{ .question = x.question, .answer = x.answer, .short = x.answer, .probe = true });
    return out.toOwnedSlice(a);
}

pub const message = "recall must list at most 160 {fact, question, answer} entries; answer must occur in fact";
