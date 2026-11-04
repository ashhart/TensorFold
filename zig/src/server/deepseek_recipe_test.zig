//! deepseek-recipe's DeepSeek-V4.1 prompt text (MIT, fixtures/dsrecipe) against the server's own render path.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const messages = @import("messages.zig");
const model_text = @import("model_text.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const kept = @import("deepseek_recipe_kept.zig").kept;
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

/// The effort name for a case's budget, as DeepSeek's harness sent it (`cc_effort`): high for any other number.
fn effortName(budget: Value) []const u8 {
    const text = if (budget == .int) budget.int else if (budget == .string) budget.string else "";
    if (std.mem.eql(u8, text, "50")) return "low";
    if (std.mem.eql(u8, text, "100")) return "max";
    return "high";
}

/// The chat request a case stands for: the one DeepSeek's endpoint took (chat_ok), else the case's flags by name.
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
        return .{ .object = body };
    }
    const chat_ok = c.get("chat_ok").? == .bool and c.get("chat_ok").?.bool;
    const on = if (chat_ok) c.get("cc_thinking").? else c.get("thinking").?;
    const kwargs = try json.newObject(a);
    try kwargs.put(a, "enable_thinking", on);
    try body.put(a, "chat_template_kwargs", .{ .object = kwargs });
    if (on.bool) try body.put(a, "reasoning_effort", if (chat_ok) c.get("cc_effort").? else .{ .string = effortName(c.get("budget").?) });
    return .{ .object = body };
}

/// The prompt text /tokenize renders for the case (a chat request's always ends in the generation prompt).
pub fn serverPath(a: Allocator, srv: *Server, c: Value) !Outcome {
    var cx: errors.Cx = .{ .a = a };
    const body = try requestOf(a, srv, c);
    var media = srv.media();
    if (c.get("image_text").? != .null) media.images = true; // a vision-serving build; the text is DeepSeek's
    const chat_ok = c.get("chat_ok").? == .bool and c.get("chat_ok").?.bool;
    const generation = chat_ok or c.get("generation").?.bool;
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
    const normalized = prompt.templateMessages(srv, &cx, msgs, media) catch |e| return refused(&cx, e);
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

/// Equal text, or a refusal where deepseek-recipe gives no prompt.
pub fn agrees(want: Value, got: Outcome) bool {
    return switch (got) {
        .text => |g| want == .string and std.mem.eql(u8, g, want.string),
        .refused => want == .null,
    };
}

/// The kept class listing case `id`, or null.
fn keptFor(id: usize) ?usize {
    for (kept, 0..) |k, i| for (k.ids) |x| if (x == id) return i;
    return null;
}

fn className(i: usize) []const u8 {
    for (kept, 0..) |k, j| if (j == i) return k.class;
    unreachable;
}

fn waits(i: usize) ?u16 {
    for (kept, 0..) |k, j| if (j == i) return k.waits;
    unreachable;
}

fn show(id: usize, want: Value, got: Outcome) void {
    switch (got) {
        .refused => |why| std.debug.print("case {d}: refused ({s}) where deepseek-recipe renders\n", .{ id, why }),
        .text => |g| if (want == .string) {
            const k = std.mem.indexOfDiff(u8, g, want.string) orelse g.len;
            std.debug.print("case {d} differs at byte {d}:\n  want {f}\n  got  {f}\n", .{ id, k, std.json.fmt(want.string[k..@min(want.string.len, k + 120)], .{}), std.json.fmt(g[k..@min(g.len, k + 120)], .{}) });
        } else std.debug.print("case {d}: renders where deepseek-recipe refuses\n", .{id}),
    }
}

/// Every case through the server path, each result appended to `report`; the count of unlisted and stale cases.
fn run(report: ?*std.Io.Writer) !usize {
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
    var equal: usize = 0;
    var bad: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = case_arena.reset(.retain_capacity);
        const a = case_arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        const got = try serverPath(a, served.srv, c);
        const want = c.get("prompt").?;
        const id = cases;
        cases += 1;
        if (report) |w| try w.print("{f}\n", .{std.json.fmt(.{ .id = id, .text = if (got == .text) got.text else null, .refused = if (got == .refused) got.refused else null }, .{})});
        const listed = keptFor(id);
        const chat_ok = c.get("chat_ok").? == .bool and c.get("chat_ok").?.bool;
        if (agrees(want, got)) {
            equal += 1;
            if (listed) |k| {
                bad += 1;
                std.debug.print("case {d} now agrees: take it off kept class {s}\n", .{ id, className(k) });
            }
        } else if (listed) |k| {
            if (chat_ok and waits(k) == null) {
                bad += 1;
                std.debug.print("case {d}: DeepSeek's endpoint takes it, so class {s} cannot keep it\n", .{ id, className(k) });
            }
        } else {
            bad += 1;
            if (bad <= 5) show(id, want, got);
        }
    }
    try t.expectEqual(@as(usize, case_count), cases);
    std.debug.print("dsrecipe server: {d}/{d} equal to deepseek-recipe's prompt text, {d} kept, {d} unexplained\n", .{ equal, cases, cases - equal, bad });
    return bad;
}

test "DeepSeek recipe: the server's chat render path gives deepseek-recipe's prompt text or a kept difference" {
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    const dir = t.environ.getPosix("TF_DSRECIPE_REPORT");
    const bad = try run(if (dir != null) &out.writer else null);
    if (dir) |d| {
        const path = try std.fmt.allocPrint(t.allocator, "{s}/server.jsonl", .{d});
        defer t.allocator.free(path);
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = out.written() });
    }
    try t.expectEqual(@as(usize, 0), bad);
}

/// The kept class listing effort name `name`, or null.
fn keptEffort(name: []const u8) ?[]const u8 {
    for (kept) |k| for (k.efforts) |e| if (std.mem.eql(u8, e, name)) return k.class;
    return null;
}

/// A one-message request with `effort` and the server's default thinking `default`, through the server path.
fn effortProbe(a: Allocator, srv: *Server, effort: Value, default: Value) !Outcome {
    const probe = (try json.parseText(a, "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"tools\": null, \"response_format\": null, \"image_text\": null, \"chat_ok\": true}")).ok;
    try probe.object.put(a, "effort_field", effort);
    try probe.object.put(a, "default_thinking", default);
    return serverPath(a, srv, probe);
}

test "DeepSeek recipe: every reasoning_effort DeepSeek's endpoint takes resolves as its records say, or is kept" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var served: Served = undefined;
    try served.init(a);
    defer served.deinit();
    const lines = try readCases(a);
    var it = std.mem.splitScalar(u8, lines, '\n');
    var checked: usize = 0;
    var bad: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const c = (try json.parseText(a, line)).ok;
        const ok = c.get("effort_ok") orelse continue;
        if (!ok.bool) continue;
        const name = c.get("effort_field").?.string;
        const got = try effortProbe(a, served.srv, c.get("effort_field").?, c.get("default_thinking").?);
        const want: ?[]const u8 = if (c.get("effort_thinking").?.bool) try std.fmt.allocPrint(a, "Reasoning Effort: {d} (range", .{c.get("effort_budget").?.int64().?}) else null;
        const text = if (got == .text) got.text else "";
        const right = if (want) |w| std.mem.indexOf(u8, text, w) != null else got == .text and std.mem.indexOf(u8, text, "Reasoning Effort:") == null;
        checked += 1;
        if (right == (keptEffort(name) == null)) continue;
        bad += 1;
        if (right) std.debug.print("reasoning_effort {s} now resolves as DeepSeek's: take it off kept class {s}\n", .{ name, keptEffort(name).? });
        if (!right) std.debug.print("reasoning_effort {s} (thinking by default {}): want {s}, got {f}\n", .{ name, c.get("default_thinking").?.bool, want orelse "thinking off", std.json.fmt(text[0..@min(text.len, 120)], .{}) });
    }
    try t.expectEqual(@as(usize, 14), checked);
    // a kept name no record covers stays refused
    const ultra = try effortProbe(a, served.srv, .{ .string = "ultra" }, .{ .bool = true });
    try t.expect(keptEffort("ultra") == null or ultra == .refused);
    try t.expectEqual(@as(usize, 0), bad);
}

test "DeepSeek recipe: the fixture is whole, a case the recipe refuses has no prompt, and each kept case once" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const lines = try readCases(arena.allocator());
    var it = std.mem.splitScalar(u8, lines, '\n');
    var n: usize = 0;
    var refused: usize = 0;
    var chat: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const c = (try json.parseText(arena.allocator(), line)).ok;
        try t.expectEqual(@as(i64, @intCast(n)), c.get("id").?.int64().?);
        if (c.get("prompt").? == .null) refused += 1;
        if (c.get("chat_ok").? == .bool and c.get("chat_ok").?.bool) chat += 1;
        n += 1;
    }
    try t.expectEqual(@as(usize, case_count), n);
    try t.expectEqual(@as(usize, 12), refused);
    try t.expectEqual(@as(usize, 78), chat);
    try t.expect(agrees(.null, .{ .refused = "x" }) and !agrees(.{ .string = "x" }, .{ .refused = "x" }));
    var seen: std.StaticBitSet(case_count) = .empty;
    for (kept) |k| for (k.ids) |id| {
        try t.expect(id < case_count and !seen.isSet(id));
        seen.set(id);
    };
}
