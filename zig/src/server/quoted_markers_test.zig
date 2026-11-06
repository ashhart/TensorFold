//! Quoted added spellings stay plain through the chat route with TF_TOKENIZER_DIRS' real tokenizers (#2399).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const errors = @import("errors.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const io = std.testing.io;

const env = "TF_TOKENIZER_DIRS";
/// The spellings the issue reports quoted; each is quoted whether or not the vocabulary adds it.
const issue = [_][]const u8{ "<|im_end|>", "<tool_call>", "</tool_call>", "<|tool_call_begin|>" };
const files = [_][]const u8{ "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json" };

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 65536 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

// A user asks, the assistant reasons and calls the offered tool, and the tool answers; the reply comes next.
const body_source =
    \\{"messages":[{"role":"user","content":"Look this up."},{"role":"assistant","content":"Calling it.","reasoning_content":"Plan.",
    \\"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]},{"role":"tool","tool_call_id":"call_1","content":"Found."}],
    \\"tools":[{"type":"function","function":{"name":"lookup","description":"Look a term up.","parameters":{"type":"object",
    \\"properties":{"query":{"type":"string","description":"The term."}},"required":["query"]}}}]}
;
const Field = struct { name: []const u8, path: []const u8, base: []const u8 };
const fields = [_]Field{
    .{ .name = "user message", .path = "messages.0.content", .base = "Look this up." },
    .{ .name = "assistant message", .path = "messages.1.content", .base = "Calling it." },
    .{ .name = "reasoning", .path = "messages.1.reasoning_content", .base = "Plan." },
    .{ .name = "tool-call arguments", .path = "messages.1.tool_calls.0.function.arguments", .base = "term" },
    .{ .name = "tool result", .path = "messages.2.content", .base = "Found." },
    .{ .name = "tool schema", .path = "tools.0.function.parameters.properties.query.description", .base = "The term." },
};

/// The directories TF_TOKENIZER_DIRS lists (path-list separated), or null after saying why the test skips.
fn dirs(comptime what: []const u8) ?std.mem.TokenIterator(u8, .scalar) {
    const list = std.c.getenv(env) orelse {
        std.debug.print("quoted markers: skipped, " ++ env ++ " names no tokenizer directories (" ++ what ++ ")\n", .{});
        return null;
    };
    return std.mem.tokenizeScalar(u8, std.mem.span(list), std.fs.path.delimiter);
}

fn assignPath(body: *json.Value, path: []const u8, value: json.Value) !void {
    var current = body;
    var parts = std.mem.splitScalar(u8, path, '.');
    while (parts.next()) |key| current = if (current.* == .array) &current.array[try std.fmt.parseInt(usize, key, 10)] else current.object.getPtr(key).?;
    current.* = value;
}

/// The chat body with `text` in `field` (arguments: the JSON object string `{"query": text}`).
fn bodyWith(a: Allocator, field: Field, text: []const u8) !json.Value {
    var body = (try json.parse(a, body_source)).ok;
    const in_arguments = std.mem.eql(u8, field.path, fields[3].path);
    const arguments = try std.fmt.allocPrint(a, "{{\"query\":{s}}}", .{try json.quote(a, if (in_arguments) text else fields[3].base, .{})});
    try assignPath(&body, fields[3].path, .{ .string = arguments });
    if (!in_arguments) try assignPath(&body, field.path, .{ .string = text });
    return body;
}

/// The prompt ids the chat route gives `body`: checked as a chat completion, then rendered by chat.prepare.
fn route(srv: *Server, a: Allocator, body: json.Value, problem: *[]const u8) ![]const u32 {
    var cx: errors.Cx = .{ .a = a };
    errdefer problem.* = cx.message;
    const prepared = try chat.prepare(srv, &cx, try openai.chatInput(srv, &cx, body), Gone{});
    srv.release(prepared.preparing, prepared.request);
    return prepared.request.prompt;
}

const Loaded = struct { hf: *HfText, srv: *Server };

fn load(a: Allocator, dir: []const u8) !Loaded {
    var problem: []const u8 = "";
    const hf = HfText.load(a, io, dir, null, a, &problem) catch |e| {
        std.debug.print("quoted markers: {s} does not load: {s}\n", .{ dir, problem });
        return e;
    };
    const script = try a.create(Script);
    const srv = try Server.init(a, io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    return .{ .hf = hf, .srv = srv };
}

test "quoted markers: every added spelling quoted in each chat field stays plain with real tokenizers (TF_TOKENIZER_DIRS)" {
    var it = dirs("quoted spellings") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bad: usize = 0;
    while (it.next()) |dir| {
        const m = try load(a, dir);
        defer m.hf.deinit();
        defer m.srv.deinit();
        const added = m.hf.tok.added;
        var spellings: std.ArrayList([]const u8) = .empty;
        for (added) |t| try spellings.append(a, t.content);
        for (issue) |s| if (m.hf.text().tokenId(s) == null or !m.hf.text().isAdded(m.hf.text().tokenId(s).?)) try spellings.append(a, s);
        var quoted: std.ArrayList(u8) = .empty;
        for (spellings.items) |s| try quoted.print(a, " `{s}`", .{s});
        std.debug.print("quoted markers: {s}: {d} added tokens, {d} spellings quoted\n", .{ std.fs.path.basename(dir), added.len, spellings.items.len });
        var problem: []const u8 = "";
        for (fields) |field| {
            const own = route(m.srv, a, try bodyWith(a, field, field.base), &problem) catch |e| return report(e, problem);
            const text = try std.mem.concat(a, u8, &.{ field.base, quoted.items });
            const ids = route(m.srv, a, try bodyWith(a, field, text), &problem) catch |e| return report(e, problem);
            for (added) |t| {
                const want = std.mem.count(u32, own, &.{t.id});
                const got = std.mem.count(u32, ids, &.{t.id});
                if (want == got) continue;
                std.debug.print("  {s}: {s} id {d} appears {d} times, the unquoted request {d}\n", .{ field.name, t.content, t.id, got, want });
                bad += 1;
            }
            const plain = try m.srv.text.decode(a, own);
            const shown = try m.srv.text.decode(a, ids);
            for (spellings.items) |s| {
                const q = try std.fmt.allocPrint(a, "`{s}`", .{s});
                if (std.mem.count(u8, shown, q) > std.mem.count(u8, plain, q)) continue;
                std.debug.print("  {s}: {s} is not in the prompt text\n", .{ field.name, q });
                bad += 1;
            }
        }
        const own = try route(m.srv, a, try bodyWith(a, fields[0], fields[0].base), &problem);
        for (issue) |s| {
            const id = m.hf.text().tokenId(s);
            if (id != null and m.hf.text().isAdded(id.?)) {
                std.debug.print("  {s}: added, id {d}, {s}, the template writes it {d} times\n", .{ s, id.?, if (m.hf.text().isSpecial(id.?)) "special" else "not special", std.mem.count(u32, own, &.{id.?}) });
            } else std.debug.print("  {s}: not an added token ({d} ordinary tokens)\n", .{ s, (try m.hf.text().encodePlain(a, s)).len });
        }
        try std.testing.expect(std.mem.count(u32, own, &.{m.hf.text().tokenId("<|im_end|>") orelse 0}) > 0);
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

fn report(e: anyerror, problem: []const u8) anyerror {
    std.debug.print("quoted markers: the chat route refused: {s}\n", .{problem});
    return e;
}

test "quoted markers: request text that encodes to an added id is still refused with real tokenizers (TF_TOKENIZER_DIRS)" {
    var it = dirs("refusal") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    while (it.next()) |dir| {
        const word = "Found";
        const real = try load(a, dir);
        defer real.hf.deinit();
        defer real.srv.deinit();
        const word_ids = try real.hf.text().encodePlain(a, word);
        try std.testing.expectEqual(@as(usize, 1), word_ids.len);
        var problem: []const u8 = "";
        const accepted = try route(real.srv, a, try bodyWith(a, fields[4], "Found."), &problem);
        try std.testing.expect(std.mem.indexOfScalar(u32, accepted, word_ids[0]) != null);
        // the same tokenizer with one added entry that is also an ordinary vocabulary word: the overlap main refuses
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        for (files) |name| {
            const path = try std.fs.path.join(a, &.{ dir, name });
            var data = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch |e| if (e == error.FileNotFound) continue else return e;
            if (std.mem.eql(u8, name, "tokenizer.json")) {
                const open = std.mem.indexOfScalarPos(u8, data, std.mem.indexOf(u8, data, "\"added_tokens\"").?, '[').? + 1;
                const entry = try std.fmt.allocPrint(a, "{{\"id\":{d},\"content\":\"{s}\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false}},", .{ word_ids[0], word });
                data = try std.mem.concat(a, u8, &.{ data[0..open], entry, data[open..] });
            }
            try tmp.dir.writeFile(io, .{ .sub_path = name, .data = data });
        }
        const overlap = try load(a, try tmp.dir.realPathFileAlloc(io, ".", a));
        defer overlap.hf.deinit();
        defer overlap.srv.deinit();
        try std.testing.expect(overlap.hf.text().isAdded(word_ids[0]));
        _ = try route(overlap.srv, a, try bodyWith(a, fields[4], "Seen."), &problem);
        try std.testing.expectError(error.Refused, route(overlap.srv, a, try bodyWith(a, fields[4], "Found."), &problem));
        std.debug.print("quoted markers: {s} with {s} added: refused: {s}\n", .{ std.fs.path.basename(dir), word, problem });
        try std.testing.expect(std.mem.indexOf(u8, problem, "request text tokenizes to an added-token id") != null);
    }
}
