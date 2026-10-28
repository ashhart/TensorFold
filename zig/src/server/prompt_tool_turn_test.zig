//! A template closing a tool reply's turn only when no assistant message follows: the mark stays the next turn's start.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const f = @import("control_tokens_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const errors = @import("errors.zig");
const prompt_mod = @import("prompt.zig");
const Server = @import("server.zig").Server;

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
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

// A tool reply is closed with the turn end only as the last message; an assistant message after it continues the turn.
const source =
    \\{% for m in messages %}{% if m.role != 'assistant' or loop.index0 == 0 or messages[loop.index0 - 1].role != 'tool' %}<|turn>{{ m.role }}
    \\{% endif %}{{ m.content }}{% if m.role != 'tool' or loop.last %}<turn|>
    \\{% endif %}{% endfor %}{% if add_generation_prompt %}<|turn>model
    \\{% endif %}
;

fn message(a: std.mem.Allocator, role: []const u8, content: []const u8) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

test "tool turn prompt: the history mark after a tool reply is a prefix of the next turn" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    try fixture.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chat_template.jinja", .data = source });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, std.testing.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{};
    const srv = try Server.init(a, std.testing.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const turn = [_]json.Value{ try message(a, "user", "ask"), try message(a, "assistant", "call"), try message(a, "tool", "result") };
    const next = turn ++ [_]json.Value{ try message(a, "assistant", "answer"), try message(a, "user", "again") };
    const first = try prompt_mod.prepare(srv, &cx, .{ .messages = .{ .array = try a.dupe(json.Value, &turn) }, .fields = .null }, false, null);
    const second = try prompt_mod.prepare(srv, &cx, .{ .messages = .{ .array = try a.dupe(json.Value, &next) }, .fields = .null }, false, null);
    try std.testing.expect(first.history_len > 0);
    try std.testing.expectEqualSlices(u32, first.ids[0..first.history_len], second.ids[0..first.history_len]);
}
