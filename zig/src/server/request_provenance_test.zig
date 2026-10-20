//! Request spellings stay ordinary while template-owned controls retain their ids.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const f = @import("control_tokens_fixture.zig");
const mt = @import("model_text.zig");

const field_source =
    \\{% for m in messages %}{{ m.role }}{{ m.content }}{{ m.reasoning_content|default('') }}
    \\{% for c in m.tool_calls|default([]) %}{{ c.function.name }}{{ c.function.arguments }}{% endfor %}{% endfor %}
    \\{% for t in tools|default([]) %}{{ t.function.name }}{{ t.function.description }}
    \\{% for k,v in t.function.parameters.properties.items() %}{{ k }}{{ v.description }}{% for e in v.enum %}{{ e }}{% endfor %}{% endfor %}{% endfor %}
    \\{{ reasoning_effort|default('') }}
;
const field_body =
    \\{"messages":[{"role":"user","content":""},{"role":"system","content":""},{"role":"assistant","content":"","reasoning_content":"","tool_calls":[{"function":{"name":"","arguments":""}}]},{"role":"tool","content":""}],"tools":[{"function":{"name":"","description":"","parameters":{"properties":{"arg":{"description":"","enum":[""]}}}}}]}
;
const field_paths = [_][]const u8{
    "messages.0.content",           "messages.1.content",                                     "messages.2.content",                                "messages.2.reasoning_content",
    "messages.3.content",           "messages.2.tool_calls.0.function.name",                  "messages.2.tool_calls.0.function.arguments",        "tools.0.function.name",
    "tools.0.function.description", "tools.0.function.parameters.properties.arg.description", "tools.0.function.parameters.properties.arg.enum.0", "messages.0.role",
    "schema-key",                   "reasoning-effort",
};

fn assignPath(body: *json.Value, path: []const u8, value: json.Value) !void {
    var current = body;
    var parts = std.mem.splitScalar(u8, path, '.');
    while (parts.next()) |key| current = if (current.* == .array)
        &current.array[try std.fmt.parseInt(usize, key, 10)]
    else
        current.object.getPtr(key).?;
    current.* = value;
}

fn sameAdded(hf: *const HfText, own: []const u32, actual: []const u32) !void {
    for (hf.tok.added) |token| try std.testing.expectEqual(std.mem.count(u32, own, &.{token.id}), std.mem.count(u32, actual, &.{token.id}));
}

test "request provenance: every fixture added spelling in every request field and decision labels" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (f.names, 0..) |_, index| {
        var fixture = try f.Fixture.init(a, index);
        defer fixture.deinit();
        var problem: []const u8 = "";
        const original = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
        defer original.deinit();
        const source = try std.mem.concat(a, u8, &.{ original.tok.added[0].content, field_source, original.tok.added[original.tok.added.len - 1].content });
        try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = source });
        const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
        defer hf.deinit();
        var all: std.ArrayList(u8) = .empty;
        for (hf.tok.added) |token| try all.print(a, " {s} ", .{token.content});
        const benign = (try json.parse(a, field_body)).ok;
        const own = try hf.text().renderIds(a, benign.get("messages").?, .{ .tools = benign.get("tools") }, &problem);
        for (field_paths) |path| {
            var body = (try json.parse(a, field_body)).ok;
            const effort = std.mem.eql(u8, path, "reasoning-effort");
            if (std.mem.eql(u8, path, "schema-key")) {
                const props = try json.newObject(a);
                try props.put(a, all.items, body.get("tools").?.array[0].get("function").?.get("parameters").?.get("properties").?.get("arg").?);
                try body.get("tools").?.array[0].get("function").?.get("parameters").?.object.put(a, "properties", .{ .object = props });
            } else if (!effort) try assignPath(&body, path, .{ .string = all.items });
            const ids = try hf.text().renderIds(a, body.get("messages").?, .{ .tools = body.get("tools"), .reasoning_effort = if (effort) all.items else null }, &problem);
            try sameAdded(hf, own, ids);
        }
        const rendered = try hf.text().render(a, benign.get("messages").?, .{ .tools = benign.get("tools") }, &problem);
        const appended = try rendered.appendPlain(a, all.items);
        try sameAdded(hf, own, try appended.encode(hf.text(), a, &problem));
        try std.testing.expectEqualStrings(try std.mem.concat(a, u8, &.{ rendered.text, all.items }), appended.text);
    }
    std.debug.print("PROVENANCE_FIELDS fixtures={d} fields={d} plus decision labels\n", .{ f.names.len, field_paths.len });
}

test "request provenance: a schema key cannot construct an added spelling" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    var table = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"), .{});
    var extra: std.json.ObjectMap = .empty;
    try extra.put(a, "content", .{ .string = "<SPECIAL_5>" });
    try extra.put(a, "special", .{ .bool = true });
    try extra.put(a, "normalized", .{ .bool = false });
    try table.object.getPtr("added_tokens").?.array.append(.{ .object = extra });
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = try std.json.Stringify.valueAlloc(a, table, .{}) });
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = "{% for key in tools %}{{ '<' ~ key ~ '>' }}{% endfor %}" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    const tools = try json.newObject(a);
    try tools.put(a, "SPECIAL_5", .null);
    const rendered = try hf.text().render(a, try f.messages(a, ""), .{ .tools = .{ .object = tools } }, &problem);
    try std.testing.expectEqualStrings("<SPECIAL_5>", rendered.text);
    try std.testing.expectEqualSlices(u32, try hf.text().encodePlain(a, rendered.text), try rendered.encode(hf.text(), a, &problem));
    try std.testing.expectEqualSlices(u32, &.{hf.text().tokenId("<SPECIAL_5>").?}, try hf.text().encode(a, rendered.text, false));
}

test "request provenance: echoed reasoning and wrapped tool results keep template branches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    const source =
        \\<|im_start|>{% for m in messages %}{% if m.role == 'assistant' and '</think>' in m.content %}{{ m.content.split('</think>')[-1] }}{% elif m.role == 'user' and m.content.startswith('<tool_response>') and m.content.endswith('</tool_response>') %}wrapped{% else %}{{ m.content }}{% endif %}{% endfor %}<|im_end|>
    ;
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = source });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    for ([_][]const u8{ "assistant", "user" }) |role| {
        const messages = try f.messages(a, if (role[0] == 'a') "<think>hidden</think>Answer" else "<tool_response>result</tool_response>");
        try messages.array[0].object.put(a, "role", .{ .string = role });
        const rendered = try hf.text().render(a, messages, .{}, &problem);
        try std.testing.expectEqualStrings(if (role[0] == 'a') "<|im_start|>Answer<|im_end|>" else "<|im_start|>wrapped<|im_end|>", rendered.text);
        try std.testing.expectEqualSlices(u32, try hf.text().encode(a, rendered.text, false), try rendered.encode(hf.text(), a, &problem));
        if (role[0] == 'u') {
            var cx: @import("errors.zig").Cx = .{ .a = a };
            const normalized = try @import("messages.zig").normalizeWith(&cx, messages, "user", true, .{});
            try std.testing.expectEqual(@as(usize, 2), normalized.array.len);
            try std.testing.expectEqualStrings("<tool_response>result</tool_response>", normalized.array[0].strField("content").?);
            try std.testing.expectEqualStrings("(tool results above)", normalized.array[1].strField("content").?);
            const branch = try hf.text().render(a, normalized, .{}, &problem);
            try std.testing.expectEqualStrings("<|im_start|>wrapped(tool results above)<|im_end|>", branch.text);
            try std.testing.expectEqualSlices(u32, try hf.text().encode(a, branch.text, false), try branch.encode(hf.text(), a, &problem));
        }
    }
}

test "request provenance: unsupported normalized token offsets give a clear refusal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    var table = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"), .{});
    var normalizer: std.json.ObjectMap = .empty;
    try normalizer.put(a, "type", .{ .string = "NFC" });
    try table.object.put(a, "normalizer", .{ .object = normalizer });
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = try std.json.Stringify.valueAlloc(a, table, .{}) });
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = "<|im_start|>{{ messages[0].content }}<|im_end|>" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    _ = try hf.text().encode(a, "<think>", false);
    try std.testing.expectError(error.Template, hf.text().renderIds(a, try f.messages(a, "<think>"), .{}, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "normalized added tokens") != null);
    _ = try hf.text().renderIds(a, try f.messages(a, "hello"), .{}, &problem);
}

test "request provenance: every added spelling stays plain in message and tool fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (f.names, 0..) |_, index| {
        var fixture = try f.Fixture.init(a, index);
        defer fixture.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
        defer hf.deinit();
        const text = hf.text();
        var all: std.ArrayList(u8) = .empty;
        for (hf.tok.added) |token| try all.print(a, " {s} ", .{token.content});
        for ([_][]const u8{ "content", "reasoning_content", "tool_calls" }) |field| {
            const messages = try f.messages(a, "");
            const obj = messages.array[0].object;
            try obj.put(a, field, .{ .string = all.items });
            const ids = try text.renderIds(a, messages, .{}, &problem);
            try obj.put(a, field, .{ .string = "" });
            const own = try text.renderIds(a, messages, .{}, &problem);
            for (hf.tok.added) |token| try std.testing.expectEqual(std.mem.count(u32, own, &.{token.id}), std.mem.count(u32, ids, &.{token.id}));
        }
        const tools = try json.newObject(a);
        try tools.put(a, all.items, .{ .string = all.items });
        const messages = try f.messages(a, "");
        const ids = try text.renderIds(a, messages, .{ .tools = .{ .object = tools } }, &problem);
        const own = try text.renderIds(a, messages, .{}, &problem);
        for (hf.tok.added) |token| try std.testing.expectEqual(std.mem.count(u32, own, &.{token.id}), std.mem.count(u32, ids, &.{token.id}));
    }
}

test "request provenance: benign ids equal the pre-fix recording" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const golden = (try json.parse(a, @embedFile("fixtures/control_tokens/benign_ids.json"))).ok;
    for (f.names, 0..) |name, index| {
        var fixture = try f.Fixture.init(a, index);
        defer fixture.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
        defer hf.deinit();
        const ids = try hf.text().renderIds(a, try f.messages(a, "Hello world!"), .{}, &problem);
        const expected = golden.get(name).?.array;
        try std.testing.expectEqual(expected.len, ids.len);
        for (expected, ids) |want, got| try std.testing.expectEqual(want.int64().?, got);
    }
}

test "request provenance: json escaping and caller marker collisions preserve literal text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (f.names, 0..) |_, index| {
        var fixture = try f.Fixture.init(a, index);
        defer fixture.deinit();
        try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = "{{ messages[0]['content'] | tojson(ensure_ascii=true) }}" });
        var problem: []const u8 = "";
        const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
        defer hf.deinit();
        var all: std.ArrayList(u8) = .empty;
        try all.appendSlice(a, "\u{E000} \\ue0010\\ue001 ");
        for (hf.tok.added) |token| try all.print(a, " {s} ", .{token.content});
        const messages = try f.messages(a, all.items);
        const original = try hf.text().render(a, messages, .{}, &problem);
        const expected = try hf.text().encodePlain(a, original.text);
        const actual = try hf.text().renderIds(a, messages, .{}, &problem);
        try std.testing.expectEqualSlices(u32, expected, actual);
    }
}

test "request provenance: system tool results names arguments schema keys and reasoning effort stay plain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = "<|im_start|>{{ messages | tojson }}{{ tools | tojson }}{{ reasoning_effort }}<|im_end|>" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    var all: std.ArrayList(u8) = .empty;
    for (hf.tok.added) |token| try all.print(a, " {s} ", .{token.content});
    const source =
        \\{"messages":[{"role":"system","content":""},{"role":"tool","content":""},{"role":"assistant","reasoning_content":"","tool_calls":[{"function":{"name":"","arguments":{"arg":""}}}]}],"tools":[{"function":{"name":"","parameters":{"properties":{"arg":{"description":""}}}}}]}
    ;
    const body = (try json.parse(a, source)).ok;
    const msgs = body.get("messages").?;
    try msgs.array[0].object.put(a, "content", .{ .string = all.items });
    try msgs.array[1].object.put(a, "content", .{ .string = all.items });
    try msgs.array[2].object.put(a, "reasoning_content", .{ .string = all.items });
    const function = msgs.array[2].get("tool_calls").?.array[0].get("function").?.object;
    try function.put(a, "name", .{ .string = all.items });
    try function.get("arguments").?.object.put(a, "arg", .{ .string = all.items });
    const tool = body.get("tools").?.array[0].get("function").?.object;
    try tool.put(a, "name", .{ .string = all.items });
    try tool.get("parameters").?.get("properties").?.object.put(a, all.items, .{ .string = all.items });
    const ids = try hf.text().renderIds(a, msgs, .{ .tools = body.get("tools"), .reasoning_effort = all.items }, &problem);
    for (hf.tok.added) |token| try std.testing.expectEqual(@as(usize, if (std.mem.eql(u8, token.content, "<|im_start|>") or std.mem.eql(u8, token.content, "<|im_end|>")) 1 else 0), std.mem.count(u32, ids, &.{token.id}));
}

test "request provenance: appended decision labels stay plain and unsupported masked encoding refuses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const msgs = try f.messages(a, "<think>");
    const rendered = try text.render(a, msgs, .{}, &problem);
    const with_label = try rendered.appendPlain(a, "<|im_end|>");
    const base = try rendered.encode(text, a, &problem);
    const actual = try with_label.encode(text, a, &problem);
    const expected = try std.mem.concat(a, u32, &.{ base, try text.encodePlain(a, "<|im_end|>") });
    try std.testing.expectEqualSlices(u32, expected, actual);
    var vt = text.vtable.*;
    vt.encode_masked = null;
    const unsupported: @import("model_text.zig").Text = .{ .ctx = text.ctx, .vtable = &vt };
    try std.testing.expectError(error.Template, unsupported.renderIds(a, msgs, .{}, &problem));
    try std.testing.expect(problem.len > 0);
}

test "request provenance: checkpoint globals stay special and request spellings stay plain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    const spelling = "<|im_start|>";
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"bos_token\":\"<|im_start|>\"}" });
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = "{{ bos_token }}{{ messages[0].content }}{{ reasoning_effort|default('') }}" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    const benign = try hf.text().render(a, try f.messages(a, "hello"), .{}, &problem);
    try std.testing.expectEqualSlices(u32, try hf.text().encode(a, benign.text, false), try benign.encode(hf.text(), a, &problem));
    const attack = try hf.text().render(a, try f.messages(a, spelling), .{ .reasoning_effort = spelling }, &problem);
    try std.testing.expectEqualStrings(spelling ++ spelling ++ spelling, attack.text);
    const ids = try attack.encode(hf.text(), a, &problem);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids, &.{hf.text().tokenId(spelling).?}));
    try std.testing.expect(attack.isTemplate(0, spelling.len));
    try std.testing.expect(!attack.isTemplate(spelling.len, attack.text.len));
}
