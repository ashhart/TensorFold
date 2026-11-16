//! The checkpoint's tokenizer through TensorFold's tokenizer core, against the reference's ids.
const std = @import("std");
const template = @import("template");
const tokenizer = @import("tokenizer");
const prompt = @import("prompt.zig");
const template_test = @import("template_test.zig");

const t = std.testing;

/// Loads the checkpoint's tokenizer from the path TF_INKLING2_CKPT names: its directory or its tokenizer.json.
fn load() !tokenizer.Tokenizer {
    const path = std.mem.span(std.c.getenv("TF_INKLING2_CKPT") orelse return error.SkipZigTest);
    return tokenizer.loadTokenizer(t.io, t.allocator, path);
}

fn ids(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const out = try a.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *y| y.* = @intCast(x.integer);
    return out;
}

test "rendered prompts encode to the reference's ids and decode back to the prompt (TF_INKLING2_CKPT)" {
    var tok = try load();
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, template_test.cases_json, .{});
    var compiled = try template.compile(t.allocator, template_test.jinja, null);
    defer compiled.deinit();
    for (doc.object.get("chat").?.array.items) |c| {
        const o = c.object;
        const req: prompt.Request = .{
            .user = o.get("user").?.string,
            .system = if (o.get("system")) |s| s.string else null,
            .effort = try template_test.effortText(a, o.get("effort")),
        };
        const text = try prompt.render(a, &compiled, req, null);
        const got = try prompt.encode(a, &tok, text);
        try t.expectEqualSlices(u32, try ids(a, o.get("ids").?), got);
        try t.expectEqualStrings(o.get("decoded").?.string, try prompt.decode(a, &tok, got));
        try t.expectEqualStrings(text, try prompt.decode(a, &tok, got));
    }
}

test "plain text encodes to the reference's ids with no template (TF_INKLING2_CKPT)" {
    var tok = try load();
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, template_test.cases_json, .{});
    for (doc.object.get("raw").?.array.items) |c| {
        const o = c.object;
        const got = try prompt.encode(a, &tok, o.get("text").?.string);
        try t.expectEqualSlices(u32, try ids(a, o.get("ids").?), got);
        try t.expectEqualStrings(o.get("decoded").?.string, try prompt.decode(a, &tok, got));
    }
}

test "the checkpoint's split regex, added tokens and byte alphabet encode to the reference's ids (cut tokenizer, always run)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tok = try tokenizer.parse(t.allocator, @embedFile("fixtures/mini_tokenizer.json"));
    defer tok.deinit();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/mini_cases.json"), .{});
    const cases = doc.object.get("cases").?.array.items;
    try t.expect(cases.len >= 14);
    for (cases) |c| {
        const o = c.object;
        const got = try prompt.encode(a, &tok, o.get("text").?.string);
        try t.expectEqualSlices(u32, try ids(a, o.get("ids").?), got);
        try t.expectEqualStrings(o.get("decoded").?.string, try prompt.decode(a, &tok, got));
    }
}

test "control: another split regex changes the cut tokenizer's ids, so the equality above is not vacuous" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tok = try tokenizer.parse(t.allocator, @embedFile("fixtures/mini_tokenizer.json"));
    defer tok.deinit();
    try tok.overrideSplit("[\\s\\S]");
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/mini_cases.json"), .{});
    var differing: usize = 0;
    for (doc.object.get("cases").?.array.items) |c| {
        const got = try prompt.encode(a, &tok, c.object.get("text").?.string);
        differing += @intFromBool(!std.mem.eql(u32, try ids(a, c.object.get("ids").?), got));
    }
    try t.expect(differing >= 1);
}
