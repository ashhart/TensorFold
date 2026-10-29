//! Host checks for auto compaction when the 16384 reserve is larger than the window.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const compact = @import("compact.zig");
const server_mod = @import("server.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const io = std.testing.io;

test "auto note budget fits a window under the reserve floor" {
    try std.testing.expectEqual(@as(i64, 16384), compact.reserveTokens(1000, .auto));
    try std.testing.expectEqual(@as(i64, 982), compact.noteBudgetTokens(8192, .auto));
    try std.testing.expectEqual(@as(i64, 8192), compact.noteBudgetTokens(200000, .auto));
}

test "auto under the reserve floor leaves a short request and compacts past fifteen percent" {
    var eng: Eng = .{ .window = 8192 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, 8);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const light = try go(srv, a, &.{try userMsg(a, "hello")}, 8, &cx);
    try std.testing.expect(light.runtime.get("compaction") == null);
    try std.testing.expectEqual(@as(usize, 1), eng.prompts.items.len);
    const msgs = [_]Value{ try userMsg(a, try pads(a, "BIG-", 7000)), try userMsg(a, "TAIL") };
    const reply = try go(srv, a, &msgs, 8, &cx);
    try std.testing.expect(reply.runtime.get("compaction") != null);
    try std.testing.expect(seenFrom(&eng, 1, "BIG-"));
}

const Stay = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{
            .encode = encode,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .vocab_size = vocabSize,
            .eos_ids = eosIds,
            .render = render,
            .template_source = templateSource,
        } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var buf: std.ArrayList(u8) = .empty;
        if (messages == .array) for (messages.array) |m| {
            const role = m.get("role") orelse Value{ .string = "" };
            const content = m.get("content") orelse Value{ .string = "" };
            try buf.appendSlice(a, if (role == .string) role.string else "");
            try buf.append(a, ':');
            try buf.appendSlice(a, if (content == .string) content.string else "");
            try buf.append(a, '\n');
        };
        if (options.add_generation_prompt) try buf.append(a, '>');
        return .{ .text = buf.items };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

const Eng = struct {
    window: u32,
    prompts: std.ArrayList([]u8) = .empty,

    fn engine(e: *Eng) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Eng = @ptrCast(@alignCast(ctx));
        return .{ .context_window = e.window, .name = "fake" };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Eng = @ptrCast(@alignCast(ctx));
        const buf = gpa.alloc(u8, request.prompt.len) catch return error.Busy;
        for (request.prompt, buf) |tok, *byte| byte.* = @intCast(tok);
        e.prompts.append(gpa, buf) catch {};
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'O', 'K' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn freePrompts(e: *Eng) void {
    for (e.prompts.items) |p| gpa.free(p);
    e.prompts.deinit(gpa);
}

fn start(eng: *Eng, text: *Text, keep: ?u32) !*server_mod.Server {
    return server_mod.Server.init(gpa, io, eng.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = true,
        .default_max_tokens = 32,
        .compact_at = .auto,
        .compact_keep = keep,
    }, null);
}

fn userMsg(a: Allocator, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "user" });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

fn pads(a: Allocator, head: []const u8, n: usize) ![]u8 {
    const s = try a.alloc(u8, head.len + n);
    @memcpy(s[0..head.len], head);
    @memset(s[head.len..], 'x');
    return s;
}

fn go(srv: *server_mod.Server, a: Allocator, msgs: []const Value, max: i64, cx: *errors.Cx) !chat.Reply {
    const owned = try a.alloc(Value, msgs.len);
    @memcpy(owned, msgs);
    return chat.run(srv, cx, .{ .messages = .{ .array = owned }, .fields = .{ .object = try json.newObject(a) }, .max_tokens = max, .temperature = 0 }, null, Stay{});
}

fn seenFrom(e: *Eng, from: usize, needle: []const u8) bool {
    for (e.prompts.items[from..]) |p| if (std.mem.indexOf(u8, p, needle) != null) return true;
    return false;
}
