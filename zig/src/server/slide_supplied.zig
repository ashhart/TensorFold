//! The questions a /v1/slide/learn caller supplies with its text: read, bounded, and put ahead of the model's own.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const max_questions = 16;
pub const max_bytes = 300;

pub const Refused = error{ NotAnArray, NotAString, TooMany, TooLong };

pub fn message(e: Refused) []const u8 {
    return switch (e) {
        error.NotAnArray, error.NotAString => "questions must be an array of strings",
        error.TooMany => "questions holds at most 16 questions",
        error.TooLong => "a question is at most 300 bytes",
    };
}

/// The body's `questions`, trimmed, blanks and repeats left out; none when the field is absent or null.
pub fn read(a: Allocator, body: json.Value) (Refused || Allocator.Error)![]const []const u8 {
    const field = body.field("questions") orelse return &.{};
    const items = field.items() orelse return error.NotAnArray;
    if (items.len > max_questions) return error.TooMany;
    var out: std.ArrayList([]const u8) = .empty;
    next: for (items) |item| {
        const text = std.mem.trim(u8, item.str() orelse return error.NotAString, " \t\r\n");
        if (text.len > max_bytes) return error.TooLong;
        if (text.len == 0) continue;
        for (out.items) |seen| if (std.mem.eql(u8, seen, text)) continue :next;
        try out.append(a, text);
    }
    return out.toOwnedSlice(a);
}

/// The supplied questions first, then the model's own that repeat none; the model's list itself when none are supplied.
pub fn ahead(a: Allocator, supplied: []const []const u8, written: []const []const u8) Allocator.Error![]const []const u8 {
    if (supplied.len == 0) return written;
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(a, supplied);
    next: for (written) |q| {
        for (supplied) |s| if (std.mem.eql(u8, s, q)) continue :next;
        try out.append(a, q);
    }
    return out.toOwnedSlice(a);
}

fn parse(a: Allocator, text: []const u8) !json.Value {
    return switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => error.InvalidJson,
    };
}

test "a body without questions supplies none, with null the same" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(@as(usize, 0), (try read(a, try parse(a, "{\"text\": \"I like blue.\"}"))).len);
    try std.testing.expectEqual(@as(usize, 0), (try read(a, try parse(a, "{\"text\": \"x\", \"questions\": null}"))).len);
    try std.testing.expectEqual(@as(usize, 0), (try read(a, try parse(a, "{\"text\": \"x\", \"questions\": []}"))).len);
}

test "questions are trimmed, blanks and repeats dropped, order kept" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try read(a, try parse(a, "{\"questions\": [\"  Who is Marta Quill? \", \"\", \"   \", \"What is Pebble?\", \"Who is Marta Quill?\"]}"));
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("Who is Marta Quill?", got[0]);
    try std.testing.expectEqualStrings("What is Pebble?", got[1]);
}

test "a field that is not an array of strings, too many questions and a long one are refused with their own words" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.NotAnArray, read(a, try parse(a, "{\"questions\": \"Who is Marta Quill?\"}")));
    try std.testing.expectError(error.NotAnArray, read(a, try parse(a, "{\"questions\": {\"q\": 1}}")));
    try std.testing.expectError(error.NotAString, read(a, try parse(a, "{\"questions\": [\"ok\", 3]}")));
    try std.testing.expectError(error.NotAString, read(a, try parse(a, "{\"questions\": [null]}")));
    var many: std.ArrayList(u8) = .empty;
    try many.appendSlice(a, "{\"questions\": [");
    for (0..max_questions + 1) |i| try many.print(a, "{s}\"q{d}\"", .{ if (i == 0) "" else ",", i });
    try many.appendSlice(a, "]}");
    try std.testing.expectError(error.TooMany, read(a, try parse(a, many.items)));
    const over = try a.alloc(u8, max_bytes + 1);
    @memset(over, 'x');
    const long = try std.fmt.allocPrint(a, "{{\"questions\": [\"{s}\"]}}", .{over});
    try std.testing.expectError(error.TooLong, read(a, try parse(a, long)));
    try std.testing.expectEqualStrings("questions must be an array of strings", message(error.NotAString));
    // the limits are inclusive: exactly max_questions and a question of exactly max_bytes are taken
    const exact = try a.alloc(u8, max_bytes);
    @memset(exact, 'x');
    const edge = try std.fmt.allocPrint(a, "{{\"questions\": [\"{s}\"]}}", .{exact});
    try std.testing.expectEqual(@as(usize, 1), (try read(a, try parse(a, edge))).len);
}

test "without supplied questions the model's list is returned untouched, the same slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const written = [_][]const u8{ "What is our robot called?", "What do we call the warehouse robot?" };
    const got = try ahead(arena.allocator(), &.{}, &written);
    try std.testing.expectEqual(@as(usize, written.len), got.len);
    try std.testing.expect(got.ptr == (&written).ptr);
}

test "supplied questions come first and the model's repeats of them are left out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const supplied = [_][]const u8{ "What is Pebble?", "Who looks after Pebble?" };
    const written = [_][]const u8{ "What is our robot called?", "What is Pebble?", "What do we call the warehouse robot?" };
    const got = try ahead(arena.allocator(), &supplied, &written);
    try std.testing.expectEqual(@as(usize, 4), got.len);
    try std.testing.expectEqualStrings("What is Pebble?", got[0]);
    try std.testing.expectEqualStrings("Who looks after Pebble?", got[1]);
    try std.testing.expectEqualStrings("What is our robot called?", got[2]);
    try std.testing.expectEqualStrings("What do we call the warehouse robot?", got[3]);
}
