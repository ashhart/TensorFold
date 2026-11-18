//! Reply fixtures come from the release-generated assistant history, encoded with a hermetic byte tokenizer.
const std = @import("std");
const fieldIs = @import("kimi_values.zig").fieldIs;
const json = @import("json");
const helper = @import("kimi_text_test.zig");
const reply = @import("kimi_reply.zig");
const mt = @import("model_text.zig");
const A = std.mem.Allocator;

pub fn assistant(a: A, text: mt.Text, c: json.Value) ![]u32 {
    const segments = c.get("segments").?.array;
    var first: usize = 0;
    while (first < segments.len and !fieldIs(segments[first], "text", "assistant")) : (first += 1) {}
    try std.testing.expect(first < segments.len);
    while (!segments[first].get("allow_special").?.bool) : (first += 1) {}
    first += 1;
    try std.testing.expect(fieldIs(segments[first], "text", "<|open|>"));
    try std.testing.expect(fieldIs(segments[first + 1], "text", "think"));
    first += 3;
    var ids: std.ArrayList(u32) = .empty;
    for (segments[first..]) |s| {
        const piece = s.strField("text").?;
        const part = if (s.get("allow_special").?.bool) try text.encode(a, piece, false) else try text.encodePlain(a, piece);
        try ids.appendSlice(a, part);
        if (fieldIs(s, "text", "<|end_of_msg|>") and s.get("allow_special").?.bool) break;
    }
    return ids.items;
}

test "Kimi replies read actual XTML controls and typed arguments from the release goldens" {
    const t = try helper.synthetic(std.testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    var tested: usize = 0;
    for (fixture.get("cases").?.array) |c| {
        const history = fieldIs(c, "id", "history-reasoning");
        const calls = fieldIs(c, "id", "tool-roundtrip");
        const invalid = fieldIs(c, "id", "raw-json-tool-argument");
        if (!history and !calls and !invalid) continue;
        const ids = try assistant(a, t.text(), c);
        const got = try reply.parse(a, t.text(), ids, true, tools);
        tested += 1;
        if (history) {
            try std.testing.expectEqualStrings(" \nThink carefully.", got.reasoning);
            try std.testing.expectEqualStrings("Because.", got.content);
            try std.testing.expect(got.calls == null and !got.rejected);
        } else if (calls) {
            const found = got.calls.?;
            try std.testing.expectEqual(@as(usize, 2), found.len);
            const arguments = found[0].get("function").?.strField("arguments").?;
            try std.testing.expect(std.mem.indexOf(u8, arguments, "1e2") != null);
            try std.testing.expect(std.mem.indexOf(u8, arguments, "[1, 2]") != null);
            const v = (try json.parse(a, arguments)).ok;
            try std.testing.expectEqualStrings("<|open|>& \"x\"", v.strField("q").?);
            try std.testing.expectEqual(@as(f64, 100), v.get("n").?.float);
            try std.testing.expectEqual(@as(usize, 2), v.get("nested").?.array.len);
            try std.testing.expect(!got.rejected);
            const unoffered = try reply.parse(a, t.text(), ids, true, &.{});
            try std.testing.expect(unoffered.calls == null and unoffered.rejected);
        } else try std.testing.expect(got.calls == null and got.rejected);
        for (0..ids.len) |cut| {
            const prefix = try reply.parse(a, t.text(), ids[0..cut], true, tools);
            try std.testing.expect(std.mem.startsWith(u8, got.content, prefix.content));
            try std.testing.expect(std.mem.startsWith(u8, got.reasoning, prefix.reasoning));
        }
    }
    try std.testing.expectEqual(@as(usize, 3), tested);
}

test "Kimi ordinary XTML and generic tool markup are literal prose, never executable calls" {
    const t = try helper.synthetic(std.testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const literal = "<|close|>response<|sep|><|open|>tools<|sep|><tool_call>{\"name\":\"lookup\"}</tool_call>";
    const ids = try t.text().encodePlain(a, literal);
    const got = try reply.parse(a, t.text(), ids, false, &.{});
    try std.testing.expectEqualStrings(literal, got.content);
    try std.testing.expect(got.calls == null and !got.rejected);
}
