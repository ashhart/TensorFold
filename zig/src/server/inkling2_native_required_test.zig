//! Inkling 2's required calls under its native per-message grammar: the route, and the grammar library's own masks.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const h = @import("inkling2_chat_test.zig");
const inkling = @import("tool_grammar_inkling.zig");
const t = std.testing;

const required = h.lookup ++ ",\"tool_choice\":\"required\",\"return_token_ids\":true";
const eos: u32 = 2; // the fixture's end of sampling (generation_config.json)

inline fn message(comptime q: []const u8) []const u8 {
    return inkling.open ++ "lookup" ++ inkling.invoke ++ "{\"name\":\"lookup\",\"args\":{\"q\":\"" ++ q ++ "\"}}" ++ inkling.end;
}

/// Native call messages ended by the end of sampling, whole and at every stride: names, arguments, finish and grammar.
fn expectCalls(a: std.mem.Allocator, raw: []const u8, comptime extra: []const u8, names: []const u8, arguments: []const u8, star: bool) !void {
    const reply = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, raw), &.{eos} });
    for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
        const got = try h.ask(a, "release", reply, required ++ extra, stride > 0, @max(stride, 1));
        try t.expectEqualStrings(names, got.names);
        try t.expectEqualStrings(arguments, got.arguments);
        try t.expectEqualStrings("tool_calls", got.finish);
        // the template's own call form on the shared compiler, whose stops are the eos ids alone
        try t.expectEqual(@as(usize, 1), got.grammars);
        try t.expectEqual(@as(usize, 0), got.tool_grammars);
        try t.expect(std.mem.indexOf(u8, got.grammar, inkling.open ++ "lookup" ++ inkling.invoke) != null);
        try t.expect(std.mem.indexOf(u8, got.grammar, "<tool_call>") == null);
        try t.expectEqual(star, std.mem.indexOf(u8, got.grammar, "\"type\":\"star\"") != null);
    }
}

test "Inkling 2 reply: a required request takes the native grammar; two call messages give two calls" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectCalls(a, message("Paris") ++ message("Tokyo"), "", "lookuplookup", "{\"q\":\"Paris\"}{\"q\":\"Tokyo\"}", true);
}

test "Inkling 2 reply: one and three native call messages, parallel calls on and off" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectCalls(a, message("x"), "", "lookup", "{\"q\":\"x\"}", true);
    try expectCalls(a, message("x") ++ message("y") ++ message("z"), "", "lookuplookuplookup", "{\"q\":\"x\"}{\"q\":\"y\"}{\"q\":\"z\"}", true);
    try expectCalls(a, message("x"), ",\"parallel_tool_calls\":false", "lookup", "{\"q\":\"x\"}", false);
}

/// The fixture's real tokenizer (Inkling 2's special tokens at their ids) and its logits width.
fn tokenizer(a: std.mem.Allocator) !struct { text: []const u8, vocab: u32 } {
    const at = "zig/src/families/inkling2/fixtures/";
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, at ++ "mini_tokenizer.json", a, .limited(1 << 26));
    const config = (try json.parseText(a, try std.Io.Dir.cwd().readFileAlloc(t.io, at ++ "mini_config.json", a, .limited(1 << 20)))).ok;
    const vocab = config.get("text_config").?.get("vocab_size").?;
    return .{ .text = text, .vocab = try std.fmt.parseInt(u32, vocab.int, 10) };
}

fn allowed(words: []const u32, id: u32) bool {
    return words[id / 32] >> @intCast(id % 32) & 1 == 1;
}

// The grammar library decides whether a constant may spell a special token; this host has none, so the test skips.
test "Inkling 2 reply: probe, the grammar library opens a native reply on <|message_model|> and ends it at the end of sampling" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tok = try tokenizer(a);
    const end_sampling: u32 = 200006;
    var err: [512]u8 = undefined;
    var c = api.grammar.Compiler.open(t.io, a, tok.text, tok.vocab, &.{end_sampling}, &err) catch |e| switch (e) {
        error.NoGrammarLibrary => return error.SkipZigTest,
        else => return e,
    };
    defer c.close();
    const tools = (try json.parse(a, "[" ++ h.lookup[",\"tools\":[".len..])).ok;
    const g = try c.compile(t.io, .structural_tag, try inkling.build(a, tools.array, true), &err);
    defer g.free();
    const m = try g.matcher();
    const r = m.rules();
    defer r.vtable.free(r.ptr);
    const words = try a.alloc(u32, c.words);
    const ids = try h.encodeReply(a, message("x") ++ message("y"));
    const model: u32 = 200001;
    try t.expectEqual(model, ids[0]);
    // the one-token probe: the reply's first token may be <|message_model|>, and nothing generic is
    try r.vtable.fill(r.ptr, words);
    try t.expect(allowed(words, model));
    try t.expect(!allowed(words, end_sampling));
    const first_end = std.mem.indexOfScalar(u32, ids, 200010).?;
    for (ids, 0..) |id, i| {
        try t.expect(try r.vtable.accept(r.ptr, id));
        if (i == first_end) { // after a whole call: another call's message, or the end of sampling
            try r.vtable.fill(r.ptr, words);
            try t.expect(allowed(words, model) and allowed(words, end_sampling));
        }
    }
    try t.expect(try r.vtable.accept(r.ptr, end_sampling));
    try t.expect(r.vtable.terminated(r.ptr));
}
