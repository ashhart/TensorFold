//! --tool-think-close: a required or named call closes its think block in time for the call; off, nothing changes.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const parse = @import("tool_parse.zig");
const fx = @import("tool_required_test.zig");
const a = std.testing.allocator;
const close = [_]u32{ 97, 98 };

fn thinkRun(want: []const u32, r: api.Request, seed: u64) !struct { tokens: []u32, stop: bool } {
    var seq: lanes.grammar.Sequence = .{ .want = want, .stop = 96, .gpa = a };
    defer seq.log.deinit(a);
    var constraint: lanes.grammar.Constraint = .init(seq.rules(), 4, 98);
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 0, .hidden_rows = true, .streams_exact = true, .batch_rows = 8, .max_streams = 2, .draft_streams = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "think", .prompt = &.{ 3, 1, 4, 1 }, .max_new = r.max_tokens, .sampling = .{ .seed = seed, .temperature = 1 }, .drafts = false, .grammar = &constraint, .think_budget = r.think_budget, .think_close = r.think_close, .think_end = if (r.think_end) |e| e else -1, .think_open = true });
    defer s.deinit(a);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .tokens = try a.dupe(u32, s.emitted()), .stop = s.reason == .stop };
}

/// A thinking server with the close markers on the fake engine, --tool-think-close as given; the caller deinits it.
fn serve(text: *fx.Text, stub: *fx.Stub, compiler: *api.grammar.Compiler, on: bool) !*Server {
    const srv = try Server.init(a, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .default_max_tokens = 300, .enable_thinking = true, .use_drafts = false, .tool_think_close = on }, null);
    srv.info.structures = @import("qwen27_masks").supported;
    srv.info.thinking_budget = true;
    srv.think_close = &close;
    srv.think_close_end = 98;
    srv.grammars = compiler;
    return srv;
}

fn choiceFor(response: bool, kind: []const u8) []const u8 {
    if (!std.mem.eql(u8, kind, "named")) return kind;
    return if (response) "{\"type\":\"function\",\"name\":\"now\"}" else "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}";
}

test "required tools: with thinking on and --tool-think-close, a required or named call closes a think block that never ends and leaves room for the call" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: fx.Stub = .{};
    var compile: fx.Compile = .{};
    var compiler = compile.compiler();
    for ([_]bool{ false, true }) |response| for ([_][]const u8{ "\"required\"", "named", "\"auto\"" }) |kind| {
        var text: fx.Text = .{};
        const srv = try serve(&text, &stub, &compiler, true);
        defer srv.deinit();
        var cx: errors.Cx = .{ .a = g };
        const named = std.mem.eql(u8, kind, "named");
        const input = try openai.chatInput(srv, &cx, try fx.body(&cx, response, choiceFor(response, kind)));
        const required = if (input.fields.get("tool_call_required")) |v| v.truthy() else false;
        const prepared = try @import("chat.zig").prepare(srv, &cx, input, @import("rerank_test.zig").Present{});
        defer srv.release(prepared.preparing, prepared.request);
        const request = prepared.request;
        try std.testing.expectEqual(!std.mem.eql(u8, kind, "\"auto\""), required);
        if (!required) {
            try std.testing.expect(request.think_budget == 0 and request.think_end == null);
            continue;
        }
        try std.testing.expectEqual(@as(u32, 300 - 256), request.think_budget);
        const want = try fx.Text.encode(undefined, g, try @import("tool_required_witness.zig").text(g, try fx.value(g, request.structure.?.text)), false);
        for ([_]u64{ 1235, 1236 }) |seed| {
            const before = try thinkRun(want, .{ .prompt = request.prompt, .max_tokens = request.max_tokens }, seed);
            defer a.free(before.tokens);
            try std.testing.expect(!before.stop and before.tokens.len == 300);
            const ran = try thinkRun(want, request, seed);
            defer a.free(ran.tokens);
            const end = std.mem.indexOfScalar(u32, ran.tokens, 98).?;
            try std.testing.expect(ran.stop and end == 300 - 256);
            const result = try parse.parse(g, try fx.Text.decode(undefined, g, ran.tokens[end + 1 ..]), input.tools, 1);
            try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
            try std.testing.expectEqualStrings(if (named) "now" else "lookup", result.calls.?[0].get("function").?.strField("name").?);
        }
    };
}

test "required tools: without --tool-think-close, a call whose thinking ends past the cap and before max_tokens is emitted as before" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var stub: fx.Stub = .{};
    var compile: fx.Compile = .{};
    var compiler = compile.compiler();
    // thinking that ends at 200 of 300 tokens, past the flag's cap of 44, then calls (a request budget stands in)
    for ([_]bool{ false, true }) |response| for ([_][]const u8{ "\"required\"", "named" }) |kind| for ([_]?i64{ null, 200 }) |own| for ([_]bool{ false, true }) |on| {
        var text: fx.Text = .{};
        const srv = try serve(&text, &stub, &compiler, on);
        defer srv.deinit();
        var cx: errors.Cx = .{ .a = g };
        var input = try openai.chatInput(srv, &cx, try fx.body(&cx, response, choiceFor(response, kind)));
        if (own) |n| try input.fields.object.put(g, "thinking_budget", .{ .int = try std.fmt.allocPrint(g, "{d}", .{n}) });
        const prepared = try @import("chat.zig").prepare(srv, &cx, input, @import("rerank_test.zig").Present{});
        defer srv.release(prepared.preparing, prepared.request);
        const request = prepared.request;
        // main's request: the client's budget alone, and a close only when the client asked for one
        const was: api.Request = .{ .prompt = request.prompt, .max_tokens = request.max_tokens, .think_budget = if (own) |n| @intCast(n) else 0, .think_close = if (own != null) &close else &.{}, .think_end = if (own != null) 98 else null };
        if (on) {
            try std.testing.expectEqual(@as(u32, 300 - 256), request.think_budget);
            continue;
        }
        try std.testing.expectEqual(was.think_budget, request.think_budget);
        try std.testing.expectEqual(was.think_end, request.think_end);
        try std.testing.expectEqualSlices(u32, was.think_close, request.think_close);
        const want = try fx.Text.encode(undefined, g, try @import("tool_required_witness.zig").text(g, try fx.value(g, request.structure.?.text)), false);
        for ([_]u64{ 1235, 1236 }) |seed| {
            const before = try thinkRun(want, was, seed);
            defer a.free(before.tokens);
            const ran = try thinkRun(want, request, seed);
            defer a.free(ran.tokens);
            try std.testing.expectEqualSlices(u32, before.tokens, ran.tokens);
            if (own == null) continue;
            const end = std.mem.indexOfScalar(u32, ran.tokens, 98).?;
            try std.testing.expect(ran.stop and end == 200);
            const result = try parse.parse(g, try fx.Text.decode(undefined, g, ran.tokens[end + 1 ..]), input.tools, 1);
            try std.testing.expectEqual(@as(usize, 1), result.calls.?.len);
        }
    };
}

test "required tools: --tool-think-close parses as a native serve flag, off by default" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var usage: @import("cli.zig").Usage = .{};
    try std.testing.expect((try @import("cli.zig").parse(arena.allocator(), &.{ "model", "--tool-think-close" }, &usage)).tool_think_close);
    try std.testing.expect(!(try @import("cli.zig").parse(arena.allocator(), &.{"model"}, &usage)).tool_think_close);
    try std.testing.expect(!(@import("server.zig").Config{ .served_name = "m", .model_ids = &.{"m"} }).tool_think_close);
}
