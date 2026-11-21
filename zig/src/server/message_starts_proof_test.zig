//! Message starts (#3617) are kept only at proven token boundaries, for about one more encode of the text.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const message_starts = @import("message_starts.zig");
const HfText = @import("hf_text.zig").HfText;
const Fixture = @import("hf_eos_fixture.zig").Fixture;
const t = std.testing;

const plain = "{% for m in messages %}{{ m.content }}{% endfor %}";
const specials =
    \\ "added_tokens":[{"id":127901,"content":"<|endoftext|>","special":true},{"id":127906,"content":"<|im_end|>","special":true}]}
;
// "xy" is one fused unknown token, so no token edge lies between x and y
const fused_unk =
    \\{"model":{"type":"BPE","vocab":{"C":0,"<unk>":1,"<|endoftext|>":127901,"<|im_end|>":127906},"merges":[],
    \\ "unk_token":"<unk>","fuse_unk":true},"decoder":{"type":"Fuse"},
++ specials;
// the normalizer writes Q as q, so the tokens no longer spell the text
const replaced =
    \\{"normalizer":{"type":"Replace","pattern":{"String":"Q"},"content":"q"},
    \\ "model":{"type":"BPE","vocab":{"A":0,"B":1,"q":2,"<|endoftext|>":127901,"<|im_end|>":127906},"merges":[]},
    \\ "decoder":{"type":"Fuse"},
++ specials;
const lossless =
    \\{"model":{"type":"BPE","vocab":{"A":0,"B":1,"Q":2,"<|endoftext|>":127901,"<|im_end|>":127906},"merges":[]},
    \\ "decoder":{"type":"Fuse"},
++ specials;

const Starts = struct { bytes: []const usize, ids: []const u32, starts: []const u32 };

fn startsOf(a: std.mem.Allocator, tokenizer: []const u8, source: []const u8, messages: []const u8) !Starts {
    var f = try Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = tokenizer });
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = source });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, f.path, null, a, &problem);
    const text = hf.text();
    var bytes: []usize = &.{};
    const rendered = try text.render(a, (try json.parse(a, messages)).ok, .{ .add_generation_prompt = false, .message_starts = &bytes }, &problem);
    const ids = try rendered.encode(text, a, &problem);
    return .{ .bytes = bytes, .ids = ids, .starts = try message_starts.tokens(text, a, rendered, ids, bytes) };
}

test "message starts: a start inside a fused unknown token is dropped (A12)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try startsOf(arena.allocator(), fused_unk, plain,
        \\[{"role":"user","content":"x"},{"role":"user","content":"yC"}]
    );
    try t.expectEqualSlices(usize, &.{ 0, 1 }, r.bytes);
    try t.expectEqualSlices(u32, &.{ 1, 0 }, r.ids); // one unknown for "xy", then C
    // "x" alone encodes to the same unknown id, which a match on encoded pieces would take as a start at 1
    try t.expectEqual(@as(usize, 0), r.starts.len);
}

test "message starts: a start after a normalized byte is dropped, and kept when nothing is normalized (A12)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const talk =
        \\[{"role":"user","content":"AQ"},{"role":"user","content":"B"}]
    ;
    const r = try startsOf(arena.allocator(), replaced, plain, talk);
    try t.expectEqualSlices(usize, &.{ 0, 2 }, r.bytes);
    try t.expectEqualSlices(u32, &.{ 0, 2, 1 }, r.ids);
    try t.expectEqual(@as(usize, 0), r.starts.len);
    const kept = try startsOf(arena.allocator(), lossless, plain, talk);
    try t.expectEqualSlices(u32, &.{2}, kept.starts);
}

/// Encodes through the real tokenizer, counting the calls and bytes.
const Counted = struct {
    var calls: usize = 0;
    var bytes: usize = 0;
    var inner: *const model_text.Text.VTable = undefined;

    fn encode(ctx: *anyopaque, a: std.mem.Allocator, text: []const u8, add_special: bool) model_text.Error![]u32 {
        calls += 1;
        bytes += text.len;
        return inner.encode(ctx, a, text, add_special);
    }

    fn masked(ctx: *anyopaque, a: std.mem.Allocator, text: []const u8, spans: []const model_text.Span, problem: *[]const u8) model_text.Error![]u32 {
        calls += 1;
        bytes += text.len;
        return inner.encode_masked.?(ctx, a, text, spans, problem);
    }
};

fn nanos() u64 {
    return @intCast(std.Io.Clock.awake.now(t.io).toNanoseconds());
}

test "message starts: 400 messages cost about one more encode of the text, not one per start (A15, A16)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try @import("thinking_family_fixture.zig").init(t.allocator, t.io, @embedFile("fixtures/thinking/qwen38.jinja"), @embedFile("fixtures/thinking/qwen38-tokenizer_config.json"), "qwen3_5");
    defer f.deinit();
    var problem: []const u8 = "";
    const text = (try HfText.load(a, t.io, f.path, null, a, &problem)).text();
    var list: std.ArrayList(u8) = .empty;
    try list.append(a, '[');
    for (0..400) |i| try list.print(a, "{s}{{\"role\":\"{s}\",\"content\":\"K{d} said something\"}}", .{ if (i == 0) "" else ",", if (i % 2 == 0) "user" else "assistant", i });
    try list.append(a, ']');
    const messages = (try json.parse(a, list.items)).ok;
    var vt = text.vtable.*;
    Counted.inner = text.vtable;
    vt.encode = Counted.encode;
    vt.encode_masked = Counted.masked;
    const counted: model_text.Text = .{ .ctx = text.ctx, .vtable = &vt };
    // fastest of five: plain render, its encode, the marked render, the starts from the marks
    var best = [4]u64{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) };
    var starts: []const u32 = &.{};
    for (0..5) |_| {
        var at = [5]u64{ nanos(), 0, 0, 0, 0 };
        const rendered = try text.render(a, messages, .{}, &problem);
        at[1] = nanos();
        const ids = try rendered.encode(text, a, &problem);
        at[2] = nanos();
        var bytes: []usize = &.{};
        const with = try text.render(a, messages, .{ .message_starts = &bytes }, &problem);
        at[3] = nanos();
        Counted.calls = 0;
        Counted.bytes = 0;
        starts = try message_starts.tokens(counted, a, with, ids, bytes);
        at[4] = nanos();
        for (&best, 0..) |*b, k| b.* = @min(b.*, at[k + 1] - at[k]);
        try t.expectEqualStrings(rendered.text, with.text);
        // one encode per kept start and one for the tail, each byte of the text once
        try t.expect(Counted.calls <= bytes.len + 1);
        try t.expectEqual(with.text.len, Counted.bytes);
    }
    try t.expectEqual(@as(usize, 399), starts.len);
    const added = (best[2] -| best[0]) + best[3];
    // about one more encode and a walk of the ids: linear, so well under twice the render and its encode
    try t.expect(added < 2 * (best[0] + best[1]));
}
