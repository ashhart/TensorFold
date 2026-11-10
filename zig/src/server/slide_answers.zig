//! What a fact's checks expect, as the graph keeps them: each question, the answer its lesson wrote, its fewest words.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

/// A check's question, its lesson's answer, its fewest words, whether never learned, and the conversation it follows.
pub const Answer = struct { old: []const u8 = "", old_short: []const u8 = "", old_ids: []const u32 = &.{}, reject_old: bool = false, question: []const u8, answer: []const u8 = "", short: []const u8 = "", probe: bool = false, context: []const u8 = "" }; // context: a kept talk's name ("": asked cold)

/// Copies of `xs` in `a`.
pub fn dupe(a: Allocator, xs: []const Answer) ![]const Answer {
    const out = try a.alloc(Answer, xs.len);
    for (xs, out) |x, *o| o.* = .{ .old = try a.dupe(u8, x.old), .old_short = try a.dupe(u8, x.old_short), .old_ids = try a.dupe(u32, x.old_ids), .reject_old = x.reject_old, .question = try a.dupe(u8, x.question), .answer = try a.dupe(u8, x.answer), .short = try a.dupe(u8, x.short), .probe = x.probe, .context = try a.dupe(u8, x.context) };
    return out;
}

/// [{"question", "answer", "short", "probe", "context"}], the last three only when they say something.
pub fn value(a: Allocator, xs: []const Answer) !json.Value {
    const out = try a.alloc(json.Value, xs.len);
    for (xs, out) |x, *v| {
        const o = try json.newObject(a);
        try o.put(a, "question", .{ .string = x.question });
        try o.put(a, "answer", .{ .string = x.answer });
        if (x.short.len > 0) try o.put(a, "short", .{ .string = x.short });
        if (x.probe) try o.put(a, "probe", .{ .bool = true });
        if (x.context.len > 0) try o.put(a, "context", .{ .string = x.context });
        if (x.old.len > 0) {
            try o.put(a, "old", .{ .string = x.old });
            try o.put(a, "old_short", .{ .string = x.old_short });
            try o.put(a, "reject_old", .{ .bool = x.reject_old });
            const ids = try a.alloc(json.Value, x.old_ids.len);
            for (x.old_ids, ids) |id, *token| token.* = try json.intValue(a, id);
            try o.put(a, "old_ids", .{ .array = ids });
        }
        v.* = .{ .object = o };
    }
    return .{ .array = out };
}

/// The answers `v` lists (an entry without a question or an answer is left out).
pub fn read(a: Allocator, v: json.Value) ![]const Answer {
    var out: std.ArrayList(Answer) = .empty;
    for (v.items() orelse return &.{}) |x| {
        const question = x.strField("question") orelse continue;
        const answer = x.strField("answer") orelse continue;
        const probe = if (x.field("probe")) |p| p == .bool and p.bool else false;
        const old = x.strField("old") orelse "";
        var old_ids: []u32 = &.{};
        var reject_old = false;
        if (old.len > 0) {
            const vals = (x.field("old_ids") orelse return error.BadCorrectionReference).items() orelse return error.BadCorrectionReference;
            if (vals.len == 0) return error.BadCorrectionReference;
            old_ids = try a.alloc(u32, vals.len);
            for (vals, old_ids) |token, *id| {
                if (token != .int) return error.BadCorrectionReference;
                id.* = std.fmt.parseInt(u32, token.int, 10) catch return error.BadCorrectionReference;
            }
            const reject = x.field("reject_old") orelse return error.BadCorrectionReference;
            if (reject != .bool) return error.BadCorrectionReference;
            reject_old = reject.bool;
            const old_short: []const u8 = x.strField("old_short") orelse "";
            if (old_short.len == 0) return error.BadCorrectionReference;
        }
        try out.append(a, .{ .old = old, .old_short = x.strField("old_short") orelse "", .old_ids = old_ids, .reject_old = reject_old, .question = question, .answer = answer, .short = x.strField("short") orelse "", .probe = probe, .context = x.strField("context") orelse "" });
    }
    return out.items;
}

test "a fact's expected answers survive the graph file, short answers, probes and conversations too" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const xs = [_]Answer{ .{ .question = "What is my dog called?", .answer = "Your dog is called Ginger.", .short = "Ginger" }, .{ .question = "Do you know my dog's name?", .answer = "My dog is called Ginger.", .probe = true }, .{ .question = "What is my dog called?", .answer = "Ginger.", .context = "recall" } };
    const text = try json.stringify(a, try value(a, &xs), .{});
    const back = try read(a, (try json.parse(a, text)).ok);
    try std.testing.expectEqual(@as(usize, 3), back.len);
    try std.testing.expectEqualStrings("Ginger", back[0].short);
    try std.testing.expect(!back[0].probe and back[1].probe);
    try std.testing.expectEqualStrings("", back[1].short);
    try std.testing.expectEqualStrings(xs[1].answer, (try dupe(a, back))[1].answer);
    try std.testing.expectEqualStrings("", back[0].context);
    try std.testing.expectEqualStrings("recall", (try dupe(a, back))[2].context);
}
