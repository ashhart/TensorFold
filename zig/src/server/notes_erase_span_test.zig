//! Real-tokenizer coverage of erased spans in rendered notes and raw teaching messages.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const hf_text = @import("hf_text.zig");
const line = @import("notes/line.zig");
const places = @import("notes_erase_places.zig");
const cases = @import("notes_erase_cases_test.zig").cases;
const testing = std.testing;

/// The ids of a chat whose system message is a notes block with `lines`, rendered by the model's own template.
fn prompt(a: std.mem.Allocator, t: @import("model_text.zig").Text, lines: []const []const u8) ![]u32 {
    var block: std.ArrayList(u8) = .empty;
    try block.appendSlice(a, "What you know about the user, from earlier conversations.\nCurrent notes:");
    for (lines) |l| try block.print(a, "\n{s}", .{l});
    const messages = try std.fmt.allocPrint(a, "[{{\"role\": \"system\", \"content\": {s}}}, {{\"role\": \"user\", \"content\": \"What is my favourite colour?\"}}]", .{try json.quote(a, block.items, .{})});
    var problem: []const u8 = "";
    return t.renderIds(a, (try json.parseText(a, messages)).ok, .{}, &problem);
}

test "real-tokenizer erase spans cover quoted notes and short or embedded teaching turns" {
    const dir = std.c.getenv("TF_NOTES_TOKENIZER_DIR") orelse {
        if (std.c.getenv("TF_NOTES_TOKENIZER_REQUIRED") != null) return error.TokenizerDirectoryRequired;
        return error.SkipZigTest;
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ht = try hf_text.HfText.load(testing.allocator, testing.io, std.mem.sliceTo(dir, 0), null, a, &problem);
    defer ht.deinit();
    const t = ht.text();
    const told = line.ordinalOf(2026, 10, 16) * line.tpd;
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.name});
        const note = try std.fmt.allocPrint(a, "Remember carefully {s} for many coming years.", .{case.raw});
        try testing.expect((try places.rawSpanOf(a, t, case.raw)) != null);
        const span = (try places.spanOf(a, t, note)) orelse return error.TestNoSpan;
        try testing.expect(span.len >= 4);
        const raw_span = (try places.rawSpanOf(a, t, note)) orelse return error.TestNoRawSpan;
        try testing.expect(raw_span.len >= 4);
        const quoted_note = try json.quote(a, note, .{});
        const histories = [_][]const u8{
            "",
            "{\"role\":\"system\",\"content\":\"Be concise.\"},",
            "{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hello!\"},",
        };
        for (histories) |history| {
            const messages = try std.fmt.allocPrint(a, "[{s}{{\"role\":\"user\",\"content\":{s}}}]", .{ history, quoted_note });
            const ids = try t.renderIds(a, (try json.parseText(a, messages)).ok, .{}, &problem);
            try testing.expect(std.mem.indexOf(u32, ids, raw_span) != null);
        }
        // the note as cached: current under one id, later superseded, later renumbered, between other lines
        const as_cached = try line.render(a, 31, note, .none, .plain);
        const superseded = try line.render(a, 31, note, .{ .superseded = .{ .by = 32, .told = told } }, .plain);
        const renumbered = try line.render(a, 30, note, .none, .plain);
        const other = "[5] \"My dog has grey eyes.\"";
        const wraps = [_][]const []const u8{ &.{as_cached}, &.{ other, as_cached }, &.{ as_cached, other }, &.{ other, superseded, other }, &.{renumbered} };
        for (wraps) |w| {
            const ids = try prompt(a, t, w);
            try testing.expect(std.mem.indexOf(u32, ids, span) != null);
        }
        // the first fix's search: the raw text's tokens without the first and last; an escaped note's are not there
        const raw = try t.encode(a, case.raw, false);
        const old = raw[1 .. raw.len - 1];
        if (!std.mem.eql(u8, case.raw, case.rendered[1 .. case.rendered.len - 1]))
            try testing.expect(std.mem.indexOf(u32, try prompt(a, t, &.{as_cached}), old) == null);
    }
    for ([_][]const u8{ "I am Ana.", "My favourite colour is green.", "The code is 8174.", "My dog is Otto.", "Zanzibar, quixotically." }) |note| {
        const raw_span = (try places.rawSpanOf(a, t, note)) orelse return error.TestNoRawSpan;
        const quoted_span = (try places.spanOf(a, t, note)) orelse return error.TestNoSpan;
        try testing.expect(raw_span.len > 0 and quoted_span.len > 0);
    }
    const green = (try places.rawSpanOf(a, t, "My favourite colour is green.")).?;
    const blue = try t.encode(a, "My favourite colour is blue.", false);
    try testing.expect(std.mem.indexOf(u32, blue, green) == null);
    const note = "My favourite colour is green.";
    const raw = (try places.rawSpanOf(a, t, note)).?;
    for ([_][]const u8{ note, "Hi! My favourite colour is green.", "Remember: My favourite colour is green.", "My favourite colour is green. Thanks!", "my favourite colour is green" }) |said| {
        const messages = try std.fmt.allocPrint(a, "[{{\"role\":\"user\",\"content\":{s}}}]", .{try json.quote(a, said, .{})});
        const ids = try t.renderIds(a, (try json.parseText(a, messages)).ok, .{}, &problem);
        try testing.expect(std.mem.indexOf(u32, ids, raw) != null);
    }
}
