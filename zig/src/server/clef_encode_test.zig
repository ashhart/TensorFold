//! Clef's record encoder on a tiny greedy tokenizer: pieces, spans, option orders, fallbacks, numbering, truncation.
const std = @import("std");
const json = @import("json");
const enc = @import("clef_encode.zig");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;
const expectError = std.testing.expectError;

/// Greedy longest match over `table` (ids from 256), else one id per byte; ": {" and "\n\n" merge across pieces.
const Tiny = struct {
    const table = [_][]const u8{ "<|im_start|>", "<|im_end|>", "<think>", "</think>", "\n\n", ": {", "OPTION", "FIELD" };

    const vtable: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = source };

    fn text() model_text.Text {
        return .{ .ctx = @ptrCast(@constCast(&table)), .vtable = &vtable };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < input.len) {
            var best: ?usize = null;
            for (table, 0..) |t, j| if (std.mem.startsWith(u8, input[i..], t) and (best == null or t.len > table[best.?].len)) {
                best = j;
            };
            if (best) |j| {
                try ids.append(a, @intCast(256 + j));
                i += table[j].len;
            } else {
                try ids.append(a, input[i]);
                i += 1;
            }
        }
        return ids.items;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id >= 256) try out.appendSlice(a, table[id - 256]) else try out.append(a, @intCast(id));
        return out.items;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256 + table.len;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, _: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return error.Template;
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
};

fn spelled(a: Allocator, ids: []const u32) ![]const u8 {
    return Tiny.text().decode(a, ids);
}

fn encodeBody(a: Allocator, body: []const u8, max_length: u32) !enc.Record {
    const v = (try json.parseText(a, body)).ok;
    return enc.encode(a, Tiny.text(), v.get("state").?, try enc.fields(a, v.get("questions").?), max_length);
}

fn instruction(a: Allocator, r: enc.Record, q: usize) ![]const u8 {
    return spelled(a, r.ids[r.questions[q].span[0]..r.questions[q].span[1]]);
}

fn option(a: Allocator, r: enc.Record, q: usize, o: usize) ![]const u8 {
    const s = r.questions[q].options[o];
    return spelled(a, r.ids[s[0]..s[1]]);
}

test "Clef: render is json.dumps with sorted keys, compact separators and text as itself" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = (try json.parseText(a, "{\"b\":[1,2.5,1e20,-0.0,true,null],\"a\":\"caf\\u00e9 \\\"q\\\"\\n\",\"\\u00e9\":{\"z\":1,\"y\":0.1}}")).ok;
    try expectEqualStrings("{\"a\":\"café \\\"q\\\"\\n\",\"b\":[1,2.5,1e+20,-0.0,true,null],\"é\":{\"y\":0.1,\"z\":1}}", try enc.render(a, parsed));
    try expectEqualStrings("plain text", try enc.render(a, .{ .string = "plain text" }));
    // control characters escape, DEL and non-ASCII stay raw, non-finite floats as JavaScript names, big ints exact
    const odd = (try json.parseText(a, "[\"\\u0001\\u007f\\t\\ud83d\\ude00\",NaN,-Infinity,123456789012345678901234567890,1e-5,{}]")).ok;
    try expectEqualStrings("[\"\\u0001\x7f\\t\u{1F600}\",NaN,-Infinity,123456789012345678901234567890,1e-05,{}]", try enc.render(a, odd));
    // more keys than any fixed buffer: the arena sorts them
    var many: std.ArrayList(u8) = .empty;
    try many.append(a, '{');
    for (0..100) |i| try many.print(a, "{s}\"k{d:0>3}\":{d}", .{ if (i == 0) "" else ",", 99 - i, i });
    try many.append(a, '}');
    const sorted = try enc.render(a, (try json.parseText(a, many.items)).ok);
    try expect(std.mem.startsWith(u8, sorted, "{\"k000\":99,\"k001\":98,"));
    try expect(std.mem.endsWith(u8, sorted, "\"k099\":0}"));
}

test "Clef: the prompt is prefix, state, numbered fields and suffix, and the spans hold the rendered pieces" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state":"S","questions":{"q1":{"type":"noul","instructions":"Is it?"},
        \\"q2":{"type":"choice","instructions":{"b":1,"a":"x"},"criteria":{"y":"Why","x":null}}}}
    ;
    const r = try encodeBody(a, body, 16384);
    const want = enc.prefix_text ++ "S" ++ "\n\nSCHEMA FIELDS:\n" ++
        "\nFIELD 1\nID: q1\nTYPE: noul\nINSTRUCTION: Is it?\nALLOWED OPTIONS:\n" ++
        "OPTION 1: {\"description\":\"The proposition is true or the answer is yes.\",\"option_id\":\"true\"}\n" ++
        "OPTION 2: {\"description\":\"The proposition is false or the answer is no.\",\"option_id\":\"false\"}\n" ++
        "END FIELD\n" ++
        "\nFIELD 2\nID: q2\nTYPE: choice\nINSTRUCTION: {\"a\":\"x\",\"b\":1}\nALLOWED OPTIONS:\n" ++
        "OPTION 1: {\"option_id\":\"x\"}\nOPTION 2: {\"description\":\"Why\",\"option_id\":\"y\"}\nEND FIELD\n" ++ enc.suffix_text;
    try expectEqualStrings(want, try spelled(a, r.ids));
    try expectEqualStrings("Is it?", try instruction(a, r, 0));
    try expectEqualStrings("{\"a\":\"x\",\"b\":1}", try instruction(a, r, 1));
    try expectEqualStrings("{\"option_id\":\"x\"}", try option(a, r, 1, 0));
    try expectEqualStrings("{\"description\":\"Why\",\"option_id\":\"y\"}", try option(a, r, 1, 1));
    try expectEqual(enc.Kind.noul, r.questions[0].kind);
    try expectEqual(enc.Kind.choice, r.questions[1].kind);
    try expectEqual(@as(u32, 1), r.state_rows);
    // each piece tokenized alone: the whole text merges ": {" and "\n\n" across pieces, the record does not
    const whole = try Tiny.text().encode(a, want, false);
    try expect(whole.len < r.ids.len);
}

test "Clef: options in encoding order: noul true then false, choice ids by code point, score levels by index" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state":"","questions":{
        \\"c":{"type":"choice","criteria":{"b":"B","\u00e9":"E","a":"a","B":"b","10":"ten","9":"nine"}},
        \\"n":{"type":"noul","criteria":{"false":"F","maybe":"ignored","true":"T"}},
        \\"s":{"type":"score","criteria":["low","mid","high"]}}}
    ;
    const r = try encodeBody(a, body, 16384);
    const ids = [_][]const []const u8{ &.{ "10", "9", "B", "a", "b", "é" }, &.{ "true", "false" }, &.{ "0", "1", "2" } };
    for (ids, r.questions) |want, q| {
        try expectEqual(want.len, q.option_ids.len);
        for (want, q.option_ids) |w, got| try expectEqualStrings(w, got);
    }
    try expectEqualStrings("{\"description\":\"E\",\"option_id\":\"é\"}", try option(a, r, 0, 5));
    try expectEqualStrings("{\"description\":\"T\",\"option_id\":\"true\"}", try option(a, r, 1, 0));
    try expectEqualStrings("{\"description\":\"F\",\"option_id\":\"false\"}", try option(a, r, 1, 1));
    try expectEqualStrings("{\"description\":\"high\",\"option_id\":\"2\"}", try option(a, r, 2, 2));
    try expectEqual(@as(u32, 0), r.state_rows);
}

test "Clef: instructions fall back to the id only when missing, null or empty; other falsy values render" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state":"x","questions":{"missing":{"type":"noul"},"null":{"type":"noul","instructions":null},
        \\"empty":{"type":"noul","instructions":""},"false":{"type":"noul","instructions":false},
        \\"zero":{"type":"noul","instructions":0},"list":{"type":"noul","instructions":[]},
        \\"dict":{"type":"noul","instructions":{}},"space":{"type":"noul","instructions":" "}}}
    ;
    const r = try encodeBody(a, body, 16384);
    const want = [_][]const u8{ "missing", "null", "empty", "false", "0", "[]", "{}", " " };
    for (want, 0..) |w, i| try expectEqualStrings(w, try instruction(a, r, i));
}

test "Clef: a null description drops its key, for choice, noul overrides and score levels" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state":"x","questions":{"c":{"type":"choice","criteria":{"a":null,"b":false}},
        \\"n":{"type":"noul","criteria":{"true":null}},"s":{"type":"score","criteria":[null,0]}}}
    ;
    const r = try encodeBody(a, body, 16384);
    try expectEqualStrings("{\"option_id\":\"a\"}", try option(a, r, 0, 0));
    try expectEqualStrings("{\"description\":false,\"option_id\":\"b\"}", try option(a, r, 0, 1));
    try expectEqualStrings("{\"option_id\":\"true\"}", try option(a, r, 1, 0));
    try expectEqualStrings("{\"description\":\"The proposition is false or the answer is no.\",\"option_id\":\"false\"}", try option(a, r, 1, 1));
    try expectEqualStrings("{\"option_id\":\"0\"}", try option(a, r, 2, 0));
    try expectEqualStrings("{\"description\":0,\"option_id\":\"1\"}", try option(a, r, 2, 1));
}

test "Clef: fields and options are numbered from 1 in order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state":"x","questions":{"z":{"type":"score","criteria":["a","b","c"]},"a":{"type":"noul"},"m":{"type":"noul"}}}
    ;
    const prompt = try spelled(a, (try encodeBody(a, body, 16384)).ids);
    for ([_][]const u8{ "\nFIELD 1\nID: z\n", "\nFIELD 2\nID: a\n", "\nFIELD 3\nID: m\n", "OPTION 3: {\"description\":\"c\"" }) |w|
        try expect(std.mem.indexOf(u8, prompt, w) != null);
    try expect(std.mem.indexOf(u8, prompt, "FIELD 0") == null and std.mem.indexOf(u8, prompt, "OPTION 0") == null);
    try expect(std.mem.indexOf(u8, prompt, "OPTION 4") == null);
}

test "Clef: the state is cut from its tail to fit; a fixed part past the limit is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"state\":\"abcdef\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"I\"}}}";
    const full = try encodeBody(a, body, 16384);
    try expectEqual(@as(u32, 6), full.state_rows);
    const fixed: u32 = @intCast(full.ids.len - 6);
    const at = try encodeBody(a, body, fixed);
    try expectEqual(@as(u32, 0), at.state_rows);
    try expectEqual(@as(usize, fixed), at.ids.len);
    const over = try encodeBody(a, body, fixed + 1);
    try expectEqual(@as(u32, 1), over.state_rows);
    const prefix = (try Tiny.text().encode(a, enc.prefix_text, false)).len;
    try expectEqualStrings("a", try spelled(a, over.ids[prefix..][0..1]));
    try expectEqual(full.questions[0].span[0] - 5, over.questions[0].span[0]);
    try expectEqual(full.questions[0].options[1][1] - 5, over.questions[0].options[1][1]);
    try expectEqualStrings("I", try instruction(a, over, 0));
    try expectError(error.SchemaTooLong, encodeBody(a, body, fixed - 1));
    const exact = try encodeBody(a, body, fixed + 6);
    try std.testing.expectEqualSlices(u32, full.ids, exact.ids);
}

test "Clef: a Unicode state stays raw, as text or inside JSON; control characters escape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text_state = "{\"state\":\"na\\u00efve \\u6771\\u4eac \\ud83d\\ude00 e\\u0301\",\"questions\":{\"q\":{\"type\":\"noul\"}}}";
    const r = try encodeBody(a, text_state, 16384);
    const s = r.ids[(try Tiny.text().encode(a, enc.prefix_text, false)).len..][0..r.state_rows];
    try expectEqualStrings("naïve 東京 \u{1F600} e\u{301}", try spelled(a, s));
    const json_state = "{\"state\":{\"k\\u00e9\":\"\\u0001\\u6771\"},\"questions\":{\"q\":{\"type\":\"noul\"}}}";
    const j = try encodeBody(a, json_state, 16384);
    const js = j.ids[(try Tiny.text().encode(a, enc.prefix_text, false)).len..][0..j.state_rows];
    try expectEqualStrings("{\"ké\":\"\\u0001東\"}", try spelled(a, js));
}

test "Clef: fields keeps request order and refuses criteria of the wrong shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = (try json.parseText(a, "{\"b\":{\"type\":\"noul\",\"criteria\":[]},\"a\":{\"type\":\"noul\",\"criteria\":false}}")).ok;
    const f = try enc.fields(a, ok);
    try expectEqualStrings("b", f[0].id);
    try expect(f[0].criteria.noul == null and f[1].criteria.noul == null);
    for ([_][]const u8{
        "{}",                                             "[]",
        "{\"q\":1}",                                      "{\"q\":{\"type\":\"maybe\"}}",
        "{\"q\":{\"type\":\"choice\"}}",                  "{\"q\":{\"type\":\"choice\",\"criteria\":{}}}",
        "{\"q\":{\"type\":\"choice\",\"criteria\":[1]}}", "{\"q\":{\"type\":\"score\",\"criteria\":[]}}",
        "{\"q\":{\"type\":\"score\",\"criteria\":{}}}",   "{\"q\":{\"type\":\"noul\",\"criteria\":[[\"true\",\"x\"]]}}",
        "{\"q\":{\"type\":1}}",
    }) |bad| try expectError(error.Invalid, enc.fields(a, (try json.parseText(a, bad)).ok));
}
