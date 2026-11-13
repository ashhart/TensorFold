//! GLM's opted-in warm pass keeps only the next turn's own cuts: main's starts 256 apart, chunked alike below each.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fixture = @import("control_tokens_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const chat = @import("chat.zig");
const openai = @import("openai.zig");
const warm = @import("warm.zig");
const t = std.testing;

const Capture = struct {
    a: std.mem.Allocator,
    prompt: []const u32 = &.{},
    chunks: []const u32 = &.{},
    fn engine(c: *@This()) api.Engine {
        return .{ .ctx = c, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info { // GLM opted in: its 2,048-row grid, request_min_chunk 256, warm turns
        return .{ .context_window = 65536, .prefill_step = 2048, .prefill_min_chunk = 256, .prompt_cache = true, .warm_turns = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        c.prompt = c.a.dupe(u32, r.prompt) catch return error.Busy;
        c.chunks = c.a.dupe(u32, r.chunks) catch return error.Busy;
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

const Never = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

fn message(a: std.mem.Allocator, role: []const u8, content: []const u8) !json.Value {
    const out = try json.newObject(a);
    try out.put(a, "role", .{ .string = role });
    try out.put(a, "content", .{ .string = content });
    return .{ .object = out };
}

test "turn starts: GLM's warm pass cuts where the next turn cuts, so a resumed turn chunks as a cold one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files = try fixture.Fixture.init(a, 0);
    defer files.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, files.path, null, a, &problem);
    defer hf.deinit();
    var c: Capture = .{ .a = a };
    const srv = try Server.init(a, t.io, c.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var messages: std.ArrayList(json.Value) = .empty;
    try messages.append(a, try message(a, "user", "Read this and answer."));
    const lengths = [_]usize{ 40, 300, 2600, 5000, 120 }; // short and long replies, past the grid and below the minimum
    var gained: usize = 0;
    var last_len: usize = 0;
    for (lengths, 0..) |n, turn| {
        const body = try json.newObject(a);
        try body.put(a, "messages", .{ .array = try a.dupe(json.Value, messages.items) });
        var cx: errors.Cx = .{ .a = a };
        const input = try openai.chatInput(srv, &cx, .{ .object = body });
        const prepared = try chat.prepare(srv, &cx, input, Never{});
        defer srv.release(prepared.preparing, prepared.request);
        const r = prepared.request;
        if (turn > 0) { // the last warm pass: a prefix of this prompt, its starts this turn's below its end
            try t.expect(c.prompt.len < r.prompt.len and std.mem.eql(u32, c.prompt, r.prompt[0..c.prompt.len]));
            var below: usize = 0;
            while (below < r.chunks.len and r.chunks[below] < c.prompt.len) below += 1;
            try t.expectEqualSlices(u32, r.chunks[0..below], c.chunks);
            if (c.chunks.len > 0 and c.chunks[c.chunks.len - 1] > last_len) gained += 1; // a cut inside the reply
        }
        // main's spacing, opted in: starts at least 256 apart, or a 2,048-row grid step
        if (r.chunks.len > 1) for (1..r.chunks.len) |i| try t.expect(r.chunks[i] - r.chunks[i - 1] >= 256);
        last_len = r.prompt.len;
        const reply = try a.alloc(u8, n);
        @memset(reply, 'w');
        const said = try message(a, "assistant", reply);
        warm.reply(srv, input, said, false, null, r.prompt.len);
        try messages.append(a, said);
        try messages.append(a, try message(a, "user", "Next question?"));
    }
    try t.expect(gained >= 2); // the long replies' warm passes reach cuts past the reply's start
}
