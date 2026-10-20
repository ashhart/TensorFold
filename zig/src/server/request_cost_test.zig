//! Fixed-size synthetic cost receipts compare the original render/encode path with masked encoding.
const std = @import("std");
const json = @import("json");
const tokenizer = @import("tokenizer");
const HfText = @import("hf_text.zig").HfText;
const f = @import("control_tokens_fixture.zig");
const t = std.testing;
const masked_available = @hasDecl(tokenizer.Tokenizer, "encodeMaskedCounted");

fn table(a: std.mem.Allocator, fixture: *f.Fixture) !void {
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"), .{});
    var tokens = std.json.Array.init(a);
    for (0..1000) |i| {
        const spelling = if (i < 997) try std.fmt.allocPrint(a, "<SPECIAL_{d}>", .{i}) else ([_][]const u8{ "<S>", "<think>", "</think>" })[i - 997];
        var obj: std.json.ObjectMap = .empty;
        try obj.put(a, "content", .{ .string = spelling });
        try obj.put(a, "special", .{ .bool = true });
        try obj.put(a, "normalized", .{ .bool = false });
        try tokens.append(.{ .object = obj });
    }
    try root.object.put(a, "added_tokens", .{ .array = tokens });
    try fixture.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = try std.json.Stringify.valueAlloc(a, root, .{}) });
    try fixture.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "<S>{{ messages[0].content }}" });
}

fn bytes(rendered: anytype) []const u8 {
    if (@typeInfo(@TypeOf(rendered)) == .pointer) return rendered;
    return rendered.text;
}

fn once(hf: *HfText, input: []const u8, masked: bool) !i96 {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const messages = try f.messages(a, input);
    var problem: []const u8 = "";
    const start = std.Io.Clock.awake.now(t.io);
    const rendered = try hf.text().render(a, messages, .{}, &problem);
    const ids = if (comptime masked_available) (if (masked)
        try rendered.encode(hf.text(), a, &problem)
    else
        try hf.text().encode(a, bytes(rendered), false)) else try hf.text().encode(a, bytes(rendered), false);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(t.io)).nanoseconds;
    try t.expect(ids.len > 0);
    return elapsed;
}

test "request provenance: 400 KiB and 1000 spellings have bounded mask work and timing receipts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    try table(a, &fixture);
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    try t.expectEqual(@as(usize, 1000), hf.tok.added.len);
    const input = try a.alloc(u8, 400 * 1024);
    for ([_]bool{ false, true }) |one_spelling| {
        @memset(input, 'a');
        if (one_spelling) @memcpy(input[200000..200007], "<think>");
        for ([_]bool{ false, true }) |masked| {
            if (!masked_available and masked) continue;
            for (0..2) |_| _ = try once(hf, input, masked);
            var total: i96 = 0;
            const rounds = 12;
            for (0..rounds) |_| total += try once(hf, input, masked);
            std.debug.print("PROVENANCE_TIMING mode={s} one={d} bytes={d} added={d} rounds={d} ns_per={d}\n", .{
                if (masked) "masked" else "raw", @intFromBool(one_spelling), input.len, hf.tok.added.len, rounds, @divTrunc(total, rounds),
            });
        }
        if (comptime masked_available) {
            const rendered = try hf.text().render(a, try f.messages(a, input), .{}, &problem);
            var work: tokenizer.MaskWork = .{};
            _ = try hf.tok.encodeMaskedCounted(a, rendered.text, rendered.spans, &work);
            try t.expect(work.spans + work.matcher <= 12 * (rendered.text.len + 1));
            try t.expectEqual(@as(usize, 1), rendered.spans.len);
            std.debug.print("PROVENANCE_MASK_WORK one={d} spans={d} matcher={d} output={d}\n", .{ @intFromBool(one_spelling), work.spans, work.matcher, rendered.text.len });
        }
    }
}
