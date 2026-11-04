//! deepseek-recipe's DeepSeek-V4.1 prompt text (MIT, fixtures/dsrecipe) against the server's own render path.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const messages = @import("messages.zig");
const model_text = @import("model_text.zig");
const tool_specs = @import("tool_specs.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const dsv41_chat = @import("dsv41_chat");
const dsv41 = if (@hasDecl(dsv41_chat, "template")) dsv41_chat.template else dsv41_chat;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const t = std.testing;

const fixture_dir = "zig/src/families/deepseek_v41/fixtures/";
pub const cases_path = fixture_dir ++ "dsrecipe/cases.jsonl.gz";
pub const case_count = 328;

/// A rendering's text, or the refusal message a client would read.
pub const Outcome = union(enum) { text: []const u8, refused: []const u8 };

/// The fixture's lines, gunzipped into `a`.
pub fn readCases(a: Allocator) ![]const u8 {
    const gz = try std.Io.Dir.cwd().readFileAlloc(t.io, cases_path, a, .limited(4 << 20));
    var in: std.Io.Reader = .fixed(gz);
    var d: std.compress.flate.Decompress = .init(&in, .gzip, &.{});
    var out: std.Io.Writer.Allocating = .init(a);
    _ = try d.reader.streamRemaining(&out.writer);
    return out.written();
}

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 20 };
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

/// A served DeepSeek-V4.1 checkpoint without Jinja: HfText over the committed mini tokenizer, and its Server.
pub const Served = struct {
    tmp: t.TmpDir,
    hf: *HfText,
    srv: *Server,
    script: Script = .{},

    pub fn init(s: *Served, a: Allocator) !void {
        s.tmp = t.tmpDir(.{});
        errdefer s.tmp.cleanup();
        const tok = try std.Io.Dir.cwd().readFileAlloc(t.io, fixture_dir ++ "serve/mini_tokenizer.json", a, .limited(1 << 20));
        try s.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = tok });
        try s.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
        const path = try s.tmp.dir.realPathFileAlloc(t.io, ".", a);
        var problem: []const u8 = "";
        s.hf = try HfText.load(a, t.io, path, null, a, &problem);
        s.script = .{};
        s.srv = try Server.init(a, t.io, s.script.engine(), s.hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    }

    pub fn deinit(s: *Served) void {
        s.srv.deinit();
        s.hf.deinit();
        s.tmp.cleanup();
    }
};

/// The effort a request names for a case's budget; a number past low / high / max reaches only a server default.
fn effortName(budget: Value) []const u8 {
    const text = if (budget == .int) budget.int else if (budget == .string) budget.string else "75";
    if (std.mem.eql(u8, text, "50")) return "low";
    if (std.mem.eql(u8, text, "75")) return "high";
    if (std.mem.eql(u8, text, "100")) return "max";
    return text;
}

/// The chat request a case stands for, with the server defaults it assumes set on `srv`.
fn requestOf(a: Allocator, srv: *Server, c: Value) !Value {
    const body = try json.newObject(a);
    try body.put(a, "messages", c.get("messages").?);
    if (c.get("tools")) |tools| if (tools != .null) try body.put(a, "tools", tools);
    if (c.get("response_format")) |schema| if (schema != .null) {
        const inner = try json.newObject(a);
        try inner.put(a, "name", .{ .string = "case" });
        try inner.put(a, "schema", schema);
        const rf = try json.newObject(a);
        try rf.put(a, "type", .{ .string = "json_schema" });
        try rf.put(a, "json_schema", .{ .object = inner });
        try body.put(a, "response_format", .{ .object = rf });
    };
    srv.config.enable_thinking = true;
    srv.config.reasoning_effort = null;
    if (c.get("effort_field")) |effort| {
        try body.put(a, "reasoning_effort", effort);
        if (c.get("default_thinking")) |d| if (d == .bool) {
            srv.config.enable_thinking = d.bool;
        };
    } else {
        const kwargs = try json.newObject(a);
        try kwargs.put(a, "enable_thinking", c.get("thinking").?);
        try body.put(a, "chat_template_kwargs", .{ .object = kwargs });
        srv.config.reasoning_effort = effortName(c.get("budget").?);
    }
    return .{ .object = body };
}

/// The prompt text as /tokenize and the chat route render it: their field reading, then prompt.renderIds before encode.
pub fn serverPath(a: Allocator, srv: *Server, c: Value) !Outcome {
    var cx: errors.Cx = .{ .a = a };
    const body = try requestOf(a, srv, c);
    var media = srv.media();
    if (c.get("image_text").? != .null) media.images = true; // a vision-serving build; the text is DeepSeek's
    const generation = c.get("generation").?.bool;
    const refused = struct {
        fn of(x: *errors.Cx, e: errors.Refused) errors.Refused!Outcome {
            return if (e == error.Refused) .{ .refused = x.message } else e;
        }
    }.of;
    const msgs = messages.normalizeWith(&cx, body.get("messages"), "system", srv.needs_user_after_tool, media) catch |e| return refused(&cx, e);
    const tools = tool_specs.active(&cx, body.get("tools"), body.get("tool_choice")) catch |e| return refused(&cx, e);
    const thinking = fields.thinkingFields(&cx, body, srv.effort_levels) catch |e| return refused(&cx, e);
    const on = thinking.enable orelse srv.config.enable_thinking;
    cx.clear_thinking = thinking.clear;
    if (@hasField(errors.Cx, "response_format")) cx.response_format = body.get("response_format");
    const effort = srv.effortFor(thinking.effort);
    const normalized = messages.toolArguments(&cx, messages.normalizeWith(&cx, msgs, srv.late_system, srv.needs_user_after_tool, media) catch |e| return refused(&cx, e)) catch |e| return refused(&cx, e);
    var options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = tools } else null,
        .add_generation_prompt = generation,
        .enable_thinking = on,
        .reasoning_effort = if (on) effort else null,
        .clear_thinking = cx.clear_thinking,
    };
    if (@hasField(model_text.RenderOptions, "response_format")) options.response_format = cx.response_format;
    if (@hasField(model_text.RenderOptions, "images")) options.images = media.images;
    var problem: []const u8 = "";
    const rendered = srv.text.render(a, normalized, options, &problem) catch |e| switch (e) {
        error.Template => return .{ .refused = problem },
        else => |x| return x,
    };
    return .{ .text = rendered.text };
}

fn imageText(ctx: ?*anyopaque, _: Value) []const u8 {
    const s: *const []const u8 = @ptrCast(@alignCast(ctx.?));
    return s.*;
}

/// The family encoder alone with the case's own flags, as golden_test.zig feeds it.
pub fn encoderPath(a: Allocator, c: Value) !Outcome {
    var image_text: []const u8 = "";
    var o: dsv41.Options = .{
        .thinking = c.get("thinking").?.bool,
        .budget = dsv41.budgetOf(c.get("budget").?) orelse 0,
        .drop_thinking = c.get("drop_thinking").?.bool,
        .add_generation_prompt = c.get("generation").?.bool,
    };
    if (c.get("effort_field")) |field| {
        const text = if (field == .string) field.string else if (field == .int) field.int else "";
        const level = dsv41.effortOf(text) orelse return .{ .refused = "reasoning_effort refused" };
        o.thinking = level != null;
        o.budget = level orelse dsv41.default_budget;
    }
    if (c.get("tools")) |tools| if (tools == .array) {
        o.tools = tools.array;
    };
    if (c.get("response_format")) |rf| if (rf != .null) {
        o.response_format = rf;
    };
    if (c.get("image_text")) |it| if (it == .string) {
        image_text = it.string;
        o.image = .{ .ctx = @ptrCast(&image_text), .text = imageText };
    };
    var problem: dsv41.Problem = .{};
    const text = dsv41.render(a, c.get("messages").?, o, &problem) catch |e| switch (e) {
        error.Refused => return .{ .refused = problem.message },
        else => |x| return x,
    };
    return .{ .text = text };
}

/// Equal text, or a refusal where deepseek-recipe gives no prompt.
pub fn agrees(want: Value, got: Outcome) bool {
    return switch (got) {
        .text => |g| want == .string and std.mem.eql(u8, g, want.string),
        .refused => want == .null,
    };
}

pub const Layer = enum { server, encoder };

/// Every case through `layer`; each result appended to `report` as a JSON line; the count that differ.
fn run(layer: Layer, report: ?*std.Io.Writer) !usize {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const lines = try readCases(arena.allocator());
    var served: Served = undefined;
    try served.init(arena.allocator());
    defer served.deinit();
    var case_arena = std.heap.ArenaAllocator.init(t.allocator);
    defer case_arena.deinit();
    var it = std.mem.splitScalar(u8, lines, '\n');
    var cases: usize = 0;
    var bad: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = case_arena.reset(.retain_capacity);
        const a = case_arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        const got = switch (layer) {
            .server => try serverPath(a, served.srv, c),
            .encoder => try encoderPath(a, c),
        };
        const want = c.get("prompt").?;
        cases += 1;
        if (report) |w| try w.print("{f}\n", .{std.json.fmt(.{ .layer = @tagName(layer), .id = cases - 1, .text = if (got == .text) got.text else null, .refused = if (got == .refused) got.refused else null }, .{})});
        if (agrees(want, got)) continue;
        bad += 1;
        if (bad > 3) continue;
        switch (got) {
            .refused => |why| std.debug.print("{s} case {d}: refused ({s}) where deepseek-recipe renders\n", .{ @tagName(layer), cases - 1, why }),
            .text => |g| if (want == .string) {
                const k = std.mem.indexOfDiff(u8, g, want.string).?;
                std.debug.print("{s} case {d} differs at byte {d}:\n  want {f}\n  got  {f}\n", .{ @tagName(layer), cases - 1, k, std.json.fmt(want.string[k..@min(want.string.len, k + 120)], .{}), std.json.fmt(g[k..@min(g.len, k + 120)], .{}) });
            } else std.debug.print("{s} case {d}: renders where deepseek-recipe refuses\n", .{ @tagName(layer), cases - 1 }),
        }
    }
    try t.expectEqual(@as(usize, case_count), cases);
    std.debug.print("dsrecipe {s}: {d}/{d} equal to deepseek-recipe's prompt text\n", .{ @tagName(layer), cases - bad, cases });
    return bad;
}

/// Runs `layer`, writing every result to TF_DSRECIPE_REPORT's file (one per layer) when it is set.
fn check(layer: Layer) !void {
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    const dir = t.environ.getPosix("TF_DSRECIPE_REPORT");
    const bad = try run(layer, if (dir != null) &out.writer else null);
    if (dir) |d| {
        const path = try std.fmt.allocPrint(t.allocator, "{s}/{s}.jsonl", .{ d, @tagName(layer) });
        defer t.allocator.free(path);
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = out.written() });
    }
    try t.expectEqual(@as(usize, 0), bad);
}

test "DeepSeek recipe: the server's chat render path gives deepseek-recipe's prompt text for every case" {
    try check(.server);
}

test "DeepSeek recipe: the family encoder with each case's own flags gives deepseek-recipe's prompt text" {
    try check(.encoder);
}

test "DeepSeek recipe: the fixture is whole and a case the recipe refuses has no prompt" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const lines = try readCases(arena.allocator());
    var it = std.mem.splitScalar(u8, lines, '\n');
    var n: usize = 0;
    var refused: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const c = (try json.parseText(arena.allocator(), line)).ok;
        try t.expectEqual(@as(i64, @intCast(n)), c.get("id").?.int64().?);
        if (c.get("prompt").? == .null) refused += 1;
        n += 1;
    }
    try t.expectEqual(@as(usize, case_count), n);
    try t.expectEqual(@as(usize, 12), refused);
    try t.expect(agrees(.null, .{ .refused = "x" }) and !agrees(.{ .string = "x" }, .{ .refused = "x" }));
}
