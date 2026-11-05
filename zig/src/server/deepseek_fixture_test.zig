//! Licensed encoder-output cases through the actual native renderer and server prompt path.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const mt = @import("model_text.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const aerr = std.mem.Allocator.Error;

pub fn cases(a: std.mem.Allocator) ![]u8 {
    var input: std.Io.Reader = .fixed(@embedFile("fixtures/deepseek_v41_encoder/cases.jsonl.gz"));
    var gz: std.compress.flate.Decompress = .init(&input, .gzip, &.{});
    return gz.reader.allocRemaining(a, .limited(2 << 20));
}

const Probe = struct {
    hf: *HfText,
    rendered: mt.Rendered = .{ .text = "" },

    fn text(p: *Probe) mt.Text {
        return .{ .native_deepseek = p.hf.text().native_deepseek, .ctx = p, .vtable = &.{ .encode = encode, .encode_masked = masked, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: bool) mt.Error![]u32 {
        const out = try a.alloc(u32, bytes.len);
        for (out, bytes) |*id, b| id.* = b;
        return out;
    }
    fn masked(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, spans: []const mt.Span, _: *[]const u8) mt.Error![]u32 {
        var end: usize = 0;
        for (spans) |s| {
            std.debug.assert(s[0] >= end and s[1] >= s[0] and s[1] <= bytes.len);
            end = s[1];
        }
        return encode(ctx, a, bytes, false);
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) aerr![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*b, id| b.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) aerr![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn source(ctx: *anyopaque) []const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return p.hf.text().templateSource();
    }
    fn render(ctx: *anyopaque, a: std.mem.Allocator, msgs: json.Value, opts: mt.RenderOptions, problem: *[]const u8) mt.Error!mt.Rendered {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.rendered = try p.hf.text().render(a, msgs, opts, problem);
        return p.rendered;
    }
};

const Engine = struct {
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 20 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn engine(ctx: *anyopaque) api.Engine {
        return .{ .ctx = ctx, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
};

fn body(a: std.mem.Allocator, c: json.Value) !json.Value {
    const out = try json.newObject(a);
    try out.put(a, "messages", c.get("messages").?);
    try out.put(a, "tools", c.get("tools").?);
    try out.put(a, "thinking", c.get("cc_thinking") orelse c.get("thinking").?);
    if (c.field("cc_effort") orelse c.field("effort_field") orelse compatibilityEffort(c)) |e| try out.put(a, "reasoning_effort", e);
    if (c.field("response_format")) |rf| try out.put(a, "response_format", rf);
    return .{ .object = out };
}

test "DeepSeek chat: licensed encoder fixture through the server render path" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try @import("hf_eos_fixture.zig").Fixture.init(a, std.testing.io, null);
    defer tmp.deinit();
    const mini = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    try tmp.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.json", .data = mini });
    try tmp.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    var probe: Probe = .{ .hf = hf };
    const srv = try Server.init(a, std.testing.io, Engine.engine(&probe), probe.text(), .{ .served_name = "fixture", .model_ids = &.{"fixture"}, .enable_thinking = true, .use_drafts = false }, null);
    defer srv.deinit();
    var total: usize = 0;
    var accepted: usize = 0;
    var equal: usize = 0;
    var refused: usize = 0;
    var rejected: usize = 0;
    var unexpected: usize = 0;
    var compatibility: usize = 0;
    var qualified: usize = 0;
    var it = std.mem.splitScalar(u8, try cases(a), '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const c = (try json.parse(a, line)).ok;
        total += 1;
        const ok = c.get("chat_ok").?.bool;
        if (ok) {
            accepted += 1;
            try checkEncoder(a, c);
        }
        if (c.get("effort_ok") != null) try checkEffort(srv, &probe, a, c);
        var cx: Cx = .{ .a = a };
        const input = @import("openai.zig").chatInput(srv, &cx, try body(a, c)) catch {
            if (ok) std.debug.print("FIXTURE case={d} refused={s}\n", .{ c.get("id").?.int64().?, cx.message });
            if (ok) {
                try std.testing.expect(keptRefusal(c.get("id").?.int64().?, cx.message));
                refused += 1;
            } else rejected += 1;
            continue;
        };
        const thinking = if (input.fields.get("enable_thinking")) |b| b.bool else true;
        const effort = srv.effortFor(if (input.fields.get("reasoning_effort")) |e| e.string else null);
        _ = @import("prompt.zig").renderIds(srv, &cx, input.messages, input.tools, thinking, effort, true) catch {
            std.debug.print("FIXTURE case={d} render_refused={s}\n", .{ c.get("id").?.int64().?, cx.message });
            if (ok) {
                try std.testing.expect(keptRefusal(c.get("id").?.int64().?, cx.message));
                refused += 1;
            } else rejected += 1;
            continue;
        };
        if (!ok and compatibilityCase(c)) {
            try std.testing.expectEqualStrings(c.get("prompt").?.string, probe.rendered.text);
            try checkEncoder(a, c);
            compatibility += 1;
            continue;
        }
        if (!ok) {
            std.debug.print("UNEXPECTED_ACCEPT {d}\n", .{c.get("id").?.int64().?});
            unexpected += 1;
            continue;
        }
        if (try checkPrompt(a, c, probe.rendered.text)) equal += 1 else qualified += 1;
    }
    std.debug.print("FIXTURE total={d} accepted={d} equal={d} qualified={d} refused={d}\n", .{ total, accepted, equal, qualified, refused });
    try extraRequests(srv, &probe, a);
    try std.testing.expectEqual(@as(usize, 328), total);
    try std.testing.expectEqual(@as(usize, 78), accepted);
    try std.testing.expectEqual(@as(usize, 50), equal);
    try std.testing.expectEqual(@as(usize, 3), qualified);
    try std.testing.expectEqual(@as(usize, 25), refused);
    try std.testing.expectEqual(@as(usize, 242), rejected);
    try std.testing.expectEqual(@as(usize, 0), unexpected);
    try std.testing.expectEqual(@as(usize, 8), compatibility);
}

// These rejected API cases record budgets 50 and 100; replay their prompts through the equivalent named client efforts.
fn compatibilityEffort(c: json.Value) ?json.Value {
    return switch (c.get("id").?.int64().?) {
        59 => .{ .string = "low" },
        271 => .{ .string = "max" },
        else => null,
    };
}

// Existing client compatibility overrides these eight fixture API refusals; their prompt bytes remain exact.
fn compatibilityCase(c: json.Value) bool {
    const id = c.get("id").?.int64().?;
    switch (id) {
        25, 59, 61, 271 => {
            for (c.get("messages").?.array) |m| if (std.mem.eql(u8, m.strField("role") orelse "", "developer")) return true;
        },
        49, 153, 191, 237 => {
            for (c.get("tools").?.array) |tool| if (tool.get("function").?.get("strict")) |strict| if (strict == .bool and strict.bool) return true;
        },
        else => {},
    }
    return false;
}

// Raw JSON schemas are not OpenAI response_format envelopes (#1698 owns that API).

// Image placeholders must not claim that this text-only build can see an image.
fn keptRefusal(id: i64, message: []const u8) bool {
    switch (id) {
        13, 26, 47, 63, 78, 84, 96, 119, 135, 139, 143, 161, 174, 190, 203, 211, 216, 229, 230, 235, 242, 261, 275 => return std.mem.startsWith(u8, message, "response_format type must be text, json_object or json_schema"),
        15, 146 => return std.mem.eql(u8, message, "this server accepts text parts only; image, audio and video inputs are unsupported"),
        else => return false,
    }
}

fn image(_: ?*anyopaque, _: json.Value) []const u8 {
    return "<｜image｜>";
}

// Three supplied accepted records drop namespaces; keep their original bytes and pin the qualified policy separately.
fn checkPrompt(a: std.mem.Allocator, c: json.Value, got: []const u8) !bool {
    const id = c.get("id").?.int64().?;
    const recorded = c.get("prompt").?.string;
    var it = std.mem.splitScalar(u8, @embedFile("fixtures/deepseek_v41_encoder/namespaces.jsonl"), '\n');
    var count: usize = 0;
    var match: ?json.Value = null;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const row = (try json.parseText(a, line)).ok;
        const row_id = row.get("id").?.int64().?;
        try std.testing.expect(row_id == 6 or row_id == 67 or row_id == 97);
        if (row_id == id) {
            try std.testing.expect(match == null);
            match = row;
        }
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
    if (match) |row| {
        try checkNamespacePin(a, row, recorded);
        try std.testing.expect(row.get("reason").?.string.len > 0);
        const expected = row.get("expected_qualified").?.string;
        try std.testing.expect(!std.mem.eql(u8, expected, recorded));
        try std.testing.expectEqualStrings(expected, got);
        return false;
    }
    try std.testing.expectEqualStrings(recorded, got);
    return true;
}

fn checkNamespacePin(a: std.mem.Allocator, row: json.Value, recorded: []const u8) !void {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(recorded, &hash, .{});
    const pinned = row.get("recorded_prompt_sha256").?.string;
    try std.testing.expectEqualStrings(pinned, &std.fmt.bytesToHex(hash, .lower));
    const names: [2][]const u8 = switch (row.get("id").?.int64().?) {
        6 => .{ "\"name\": \"git::commit\", \"description\": \"Git operations.\\nCommit staged changes\"", "\"name\": \"commit\", \"description\": \"Commit staged changes\"" },
        67, 97 => .{ "\"name\": \"fs::read_file\"", "\"name\": \"read_file\"" },
        else => return error.TestUnexpectedResult,
    };
    const qualified = row.get("expected_qualified").?.string;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, qualified, names[0]));
    const undone = try std.mem.replaceOwned(u8, a, qualified, names[0], names[1]);
    defer a.free(undone);
    std.crypto.hash.sha2.Sha256.hash(undone, &hash, .{});
    try std.testing.expectEqualStrings(pinned, &std.fmt.bytesToHex(hash, .lower));
}

test "DeepSeek chat: namespace pins reject unrelated prompt drift" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var it = std.mem.splitScalar(u8, @embedFile("fixtures/deepseek_v41_encoder/namespaces.jsonl"), '\n');
    const input = try cases(a);
    var checked: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const row = (try json.parseText(a, line)).ok;
        var originals = std.mem.splitScalar(u8, input, '\n');
        var recorded: ?[]const u8 = null;
        while (originals.next()) |original| {
            if (original.len == 0) continue;
            const c = (try json.parseText(a, original)).ok;
            if (c.get("id").?.int64().? == row.get("id").?.int64().?) {
                recorded = c.get("prompt").?.string;
                break;
            }
        }
        try std.testing.expect(recorded != null);
        try checkNamespacePin(a, row, recorded.?);
        const expected = row.get("expected_qualified").?.string;
        try row.object.put(a, "expected_qualified", .{ .string = try std.fmt.allocPrint(a, "{s}unrelated drift", .{expected}) });
        try std.testing.expectError(error.TestExpectedEqual, checkNamespacePin(a, row, recorded.?));
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), checked);
}

fn checkEncoder(a: std.mem.Allocator, c: json.Value) !void {
    const ds = @import("dsv41_chat");
    const e = c.field("cc_effort") orelse compatibilityEffort(c);
    var problem: ds.Problem = .{};
    const got = try ds.render(a, c.get("messages").?, .{
        .tools = if (c.field("tools")) |t| t.array else &.{},
        .thinking = (c.get("cc_thinking") orelse c.get("thinking").?).bool,
        .budget = if (e) |v| (ds.effortOf(v.string).? orelse ds.default_budget) else ds.default_budget,
        .response_format = c.field("response_format"),
        .image = .{ .text = image },
    }, &problem);
    _ = try checkPrompt(a, c, got);
}

fn checkEffort(srv: *Server, p: *Probe, a: std.mem.Allocator, c: json.Value) !void {
    const saved_default = srv.config.enable_thinking;
    defer srv.config.enable_thinking = saved_default;
    srv.config.enable_thinking = c.get("default_thinking").?.bool;
    const b = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")).ok;
    try b.object.put(a, "reasoning_effort", c.get("effort_field").?);
    var cx: Cx = .{ .a = a };
    const input = @import("openai.zig").chatInput(srv, &cx, b) catch |err| {
        try std.testing.expect(!c.get("effort_ok").?.bool);
        try std.testing.expectEqual(error.Refused, err);
        return;
    };
    try std.testing.expect(c.get("effort_ok").?.bool);
    const thinking = input.fields.get("enable_thinking").?.bool;
    try std.testing.expectEqual(c.get("effort_thinking").?.bool, thinking);
    _ = try @import("prompt.zig").renderIds(srv, &cx, input.messages, input.tools, thinking, input.fields.get("reasoning_effort").?.string, true);
    if (c.field("effort_budget")) |budget| {
        const prefix = try std.fmt.allocPrint(a, "Reasoning Effort: {d} (range", .{budget.int64().?});
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, prefix) != null);
        try std.testing.expect(std.mem.endsWith(u8, p.rendered.text, "<｜Assistant｜><think>"));
    } else try std.testing.expect(std.mem.endsWith(u8, p.rendered.text, "<｜Assistant｜></think>"));
    // No fixture assigns ultra a budget. Preserve main's named refusal instead of guessing one.
    try b.object.put(a, "reasoning_effort", .{ .string = "ultra" });
    try std.testing.expectError(error.Refused, @import("openai.zig").chatInput(srv, &cx, b));
}

fn extraRequests(srv: *Server, p: *Probe, a: std.mem.Allocator) !void {
    const raw = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"\"},{\"type\":\"text\",\"text\":\"hi\"}]},{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"5\"}}]}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{},\"strict\":false,\"extra\":1}}]}";
    for ([_][]const u8{ "5", "[1,2]", "null", "broken" }) |arg| {
        const b = (try json.parse(a, raw)).ok;
        try b.get("messages").?.array[1].get("tool_calls").?.array[0].get("function").?.object.put(a, "arguments", .{ .string = arg });
        var cx: Cx = .{ .a = a };
        const input = try @import("openai.zig").chatInput(srv, &cx, b);
        _ = try @import("prompt.zig").renderIds(srv, &cx, input.messages, input.tools, true, null, true);
        const parameter = try std.fmt.allocPrint(a, "parameter name=\"arguments\" string=\"true\">{s}</", .{arg});
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, parameter) != null);
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, "_invalid_arguments") == null);
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, "{\"name\": \"f\", \"description\": \"\", \"parameters\": {}}") != null);
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, "\"strict\"") == null);
        try std.testing.expect(std.mem.indexOf(u8, p.rendered.text, "<｜User｜>\n\nhi") != null);
    }
}
