const std = @import("std");
const render = @import("responses_render.zig");
const responses = @import("responses_translate.zig");
const store_mod = @import("responses_store.zig");
const errors = @import("errors.zig");
const json = @import("json");
const server_mod = @import("server.zig");

const t = std.testing;

const Fixture = struct {
    text: @import("stream_preflight_test.zig").TestText = .{},
    srv: *server_mod.Server,
    fn init() !*Fixture {
        const fixture = try t.allocator.create(Fixture);
        errdefer t.allocator.destroy(fixture);
        fixture.* = .{ .srv = undefined };
        fixture.srv = try server_mod.Server.init(t.allocator, t.io, fixture.engine(), fixture.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
        return fixture;
    }

    fn deinit(f: *Fixture) void {
        f.srv.deinit();
        t.allocator.destroy(f);
    }

    fn engine(f: *Fixture) @import("engine_api").Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) @import("engine_api").Info {
        return .{ .context_window = 4096 };
    }
    fn submit(_: *anyopaque, _: @import("engine_api").Id, _: *const @import("engine_api").Request, _: @import("engine_api").Sink) @import("engine_api").SubmitError!void {}
    fn cancel(_: *anyopaque, _: @import("engine_api").Id) void {}
    fn status(_: *anyopaque, out: *@import("engine_api").Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?@import("engine_api").Memory {
        return null;
    }
};

fn identity(srv: *server_mod.Server) render.Identity {
    return .{ .model = @splat(1), .tokenizer = @splat(2), .template = render.templateIdentity(srv), .media = @splat(4) };
}

const NoHistory = struct {
    pub fn conversation(_: @This(), cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no earlier turn");
    }
};

test "Responses render receipt: inline and saved-parent turns emit identical token IDs" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };

    const inline_body = (try json.parse(a, "{\"input\":[{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"user\",\"content\":\"next\"}]}")).ok;
    const inline_turn = try responses.translate(&cx, inline_body, NoHistory{});
    const inline_receipt = try render.render(fixture.srv, &cx, inline_turn.chat, identity(fixture.srv));

    var parent_messages = [_]json.Value{(try json.parse(a, "{\"role\":\"user\",\"content\":\"first\"}")).ok};
    try fixture.srv.store.put(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_parent\",\"output\":[]}")).ok, &parent_messages, null);
    const saved_body = (try json.parse(a, "{\"input\":\"next\",\"previous_response_id\":\"resp_parent\"}")).ok;
    const history: store_mod.History = .{ .store = &fixture.srv.store, .io = t.io, .owner = null };
    const saved_turn = try responses.translate(&cx, saved_body, history);
    const saved_receipt = try render.render(fixture.srv, &cx, saved_turn.chat, identity(fixture.srv));

    try t.expectEqualSlices(u32, inline_receipt.ids, saved_receipt.ids);
    try t.expectEqualStrings(&inline_receipt.token_sha256, &saved_receipt.token_sha256);
    try t.expectEqual(inline_receipt.render_options, saved_receipt.render_options);
    try t.expectEqual(@as(u32, 1), inline_receipt.render_calls);
    try t.expectEqual(@as(u32, 1), saved_receipt.render_calls);
}

test "Responses render receipt: custom call and result replay produce identical prompt IDs" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const tools = ",\"tools\":[{\"type\":\"custom\",\"name\":\"freeform\",\"format\":{\"type\":\"text\"}}]";
    const call = "{\"type\":\"custom_tool_call\",\"id\":\"ctc_client\",\"call_id\":\"custom-1\",\"name\":\"freeform\",\"input\":\" raw\\n{\\\"partial\"}";
    const result = "{\"type\":\"custom_tool_call_output\",\"call_id\":\"custom-1\",\"output\":\"kept\"}";
    const inline_body = (try json.parse(a, "{\"input\":[" ++ call ++ "," ++ result ++ "]" ++ tools ++ "}")).ok;
    const inline_turn = try responses.translate(&cx, inline_body, NoHistory{});
    const inline_receipt = try render.render(fixture.srv, &cx, inline_turn.chat, identity(fixture.srv));

    var none: [0]json.Value = .{};
    try fixture.srv.store.put(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_custom\",\"output\":[" ++ call ++ "]}")).ok, &none, null);
    const saved_body = (try json.parse(a, "{\"input\":[" ++ result ++ "],\"previous_response_id\":\"resp_custom\"" ++ tools ++ "}")).ok;
    const history: store_mod.History = .{ .store = &fixture.srv.store, .io = t.io, .owner = null };
    const saved_turn = try responses.translate(&cx, saved_body, history);
    const saved_receipt = try render.render(fixture.srv, &cx, saved_turn.chat, identity(fixture.srv));
    try t.expectEqualSlices(u32, inline_receipt.ids, saved_receipt.ids);
    try t.expectEqualStrings(&inline_receipt.token_sha256, &saved_receipt.token_sha256);
    try t.expectEqual(inline_receipt.render_options, saved_receipt.render_options);
}

test "Responses render receipt: worker mismatch is explicit and prefill delay uses observed timestamps" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}")).ok;
    const receipt = try render.render(fixture.srv, &cx, body, identity(fixture.srv));
    const worker: render.Worker = .{ .identity = receipt.identity, .render_options = receipt.render_options };
    try receipt.admit(worker);
    try t.expectEqual(@as(u64, 25), try receipt.inputToPrefillNs(receipt.started_ns + 25));
    try t.expectError(error.InvalidPrefillTimestamp, receipt.inputToPrefillNs(receipt.started_ns - 1));

    inline for (.{ "model", "tokenizer", "template", "media" }) |field| {
        var mismatched = worker;
        @field(mismatched.identity, field)[0] +%= 1;
        try t.expectError(error.WorkerIdentityMismatch, receipt.admit(mismatched));
    }
    var options_changed = worker;
    options_changed.render_options[0] +%= 1;
    try t.expectError(error.WorkerRenderOptionsMismatch, receipt.admit(options_changed));

    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    try receipt.describe(&out.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
    defer parsed.deinit();
    try t.expectEqual(@as(i64, 1), parsed.value.object.get("render_calls").?.integer);
    try t.expectEqual(@as(i64, @intCast(receipt.ids.len)), parsed.value.object.get("prompt_tokens").?.integer);
    try t.expectEqualStrings(&receipt.token_sha256, parsed.value.object.get("token_sha256").?.string);
}

test "Responses render receipt: options digest changes with render controls" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = (try json.parse(a, "{\"messages\":[],\"enable_thinking\":false}")).ok;
    const thinking = (try json.parse(a, "{\"messages\":[],\"enable_thinking\":true}")).ok;
    const first = try render.optionDigest(a, plain);
    const same = try render.optionDigest(a, plain);
    const changed = try render.optionDigest(a, thinking);
    try t.expectEqual(first, same);
    try t.expect(!std.mem.eql(u8, &first, &changed));
}
