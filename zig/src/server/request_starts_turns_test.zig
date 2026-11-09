//! Six actual chat preparations keep each history end, even when turns are closer than the default chunk minimum.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fixture = @import("control_tokens_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const chat = @import("chat.zig");
const openai = @import("openai.zig");
const pc = api.prompt_cache;
const a = std.testing.allocator;
const t = std.testing;

const Fake = struct {
    at: u32 = 0,
    sum: u64 = 0,
    saves: usize = 0,
    minimum: u32,
    ends: bool = false,
    capture: bool = false,
    captured: ?struct { ids: []u32, cuts: []u32 } = null,
    const Saved = struct { at: u32, sum: u64 };
    fn engine(f: *@This()) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ptr: *anyopaque) api.Info {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        var out: api.Info = .{ .context_window = 32768, .prefill_step = 4096, .prompt_cache = true, .prefill_turn_ends = f.ends };
        if (@hasField(api.Info, "prefill_min_chunk")) out.prefill_min_chunk = f.minimum;
        return out;
    }
    fn submit(ptr: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        if (!f.capture) return error.Closed;
        const ids = a.dupe(u32, r.prompt) catch return error.Busy;
        const cuts = a.dupe(u32, r.chunks) catch {
            a.free(ids);
            return error.Busy;
        };
        f.captured = .{ .ids = ids, .cuts = cuts };
        const e: api.Event = .{ .finished = .{ .reason = .length, .stats = .{} } };
        sink.event(sink.ctx, id, &e);
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return @sizeOf(Saved);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        try t.expectEqual(at, f.at);
        const s = try a.create(Saved);
        s.* = .{ .at = at, .sum = f.sum };
        f.saves += 1;
        return s;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        const s: *Saved = @ptrCast(@alignCast(saved));
        f.at = s.at;
        f.sum = s.sum;
    }
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        a.destroy(@as(*Saved, @ptrCast(@alignCast(saved))));
    }
    fn snapshots(f: *@This()) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};
const Never = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};
fn fresh(ids: []const u32) u64 {
    var sum: u64 = 0;
    for (ids) |id| sum = sum *% 31 +% id;
    return sum;
}
fn message(alloc: std.mem.Allocator, role: []const u8, content: []const u8) !json.Value {
    const out = try json.newObject(alloc);
    try out.put(alloc, "role", .{ .string = role });
    try out.put(alloc, "content", .{ .string = content });
    return .{ .object = out };
}

test "turn starts: native metadata reaches six chat histories and a one-slot kept store" {
    for ([_]u32{ 256, 1 }) |minimum| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const alloc = arena.allocator();
        var files = try fixture.Fixture.init(alloc, 0);
        defer files.deinit();
        var problem: []const u8 = "";
        const hf = try HfText.load(alloc, t.io, files.path, null, alloc, &problem);
        defer hf.deinit();
        var f: Fake = .{ .minimum = minimum };
        const srv = try Server.init(alloc, t.io, f.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
        defer srv.deinit();
        var store = pc.Store.init(a, f.snapshots(), .{ .planned = true, .grid = 4096, .min_prompt = 0 }, @sizeOf(Fake.Saved));
        defer store.deinit();
        store.most = 1;
        var messages: std.ArrayList(json.Value) = .empty;
        const document = try alloc.alloc(u8, 1450);
        @memset(document, 'x');
        try messages.append(alloc, try message(alloc, "user", document));
        var previous: u32 = 0;
        var first: u32 = 0;
        for (0..6) |turn| {
            const body = try json.newObject(alloc);
            try body.put(alloc, "messages", .{ .array = messages.items });
            try body.put(alloc, "max_tokens", try json.intValue(alloc, 1));
            var cx: errors.Cx = .{ .a = alloc };
            const input = try openai.chatInput(srv, &cx, .{ .object = body });
            const prepared = try chat.prepare(srv, &cx, input, Never{});
            defer srv.release(prepared.preparing, prepared.request);
            const r = prepared.request;
            if (turn == 0) first = r.history_len;
            if (turn > 0) try t.expect(r.history_len > previous and r.history_len - first < 256);
            const plan = try store.beginUser(alloc, r.prompt, r.history_len, r.rewind_len, r.user_end, r.shared_prefixes, r.chunks, null, r.decode_spans);
            try t.expectEqual(if (turn == 0) 0 else if (minimum == 1) previous else first, plan.from);
            const wanted = if (minimum == 1) r.history_len else first;
            if (minimum == 1 or turn == 0) try t.expectEqualSlices(u32, &.{wanted}, plan.marks);
            if (plan.from == 0) {
                f.at = 0;
                f.sum = 0;
            }
            while (f.at < r.prompt.len) {
                f.sum = f.sum *% 31 +% r.prompt[f.at];
                f.at += 1;
                if (std.mem.indexOfScalar(u32, plan.marks, f.at) != null) try t.expect(store.keep(r.prompt, f.at, null, r.chunks, r.decode_spans));
            }
            try t.expectEqual(fresh(r.prompt), f.sum);
            try t.expectEqual(@as(usize, 1), store.entries.items.len);
            try t.expectEqual(wanted, store.entries.items[0].at);
            try t.expect(store.held <= store.budget);
            previous = r.history_len;
            try messages.append(alloc, try message(alloc, "assistant", "Fixed reply."));
            try messages.append(alloc, try message(alloc, "user", "Next question?"));
        }
        try t.expectEqual(@as(usize, if (minimum == 1) 6 else 1), f.saves);
    }
}

// A minimal reasoning wrapper exercises production Jinja/tokenization without a checkpoint or borrowed template.
const reasoning_template =
    "{% for m in messages %}<|im_start|>{{ m.role }}\n" ++
    "{% if m.role == 'assistant' %}<think>\n{{ m.reasoning_content | default('') }}\n</think>\n\n{% endif %}" ++
    "{{ m.content }}<|im_end|>\n{% endfor %}" ++
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}";

test "turn starts: dropped reasoning shares the user end but not the reply endpoint" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var files = try fixture.Fixture.init(alloc, 0);
    defer files.deinit();
    try files.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = reasoning_template });
    var problem: []const u8 = "";
    const hf = try HfText.load(alloc, t.io, files.path, null, alloc, &problem);
    defer hf.deinit();
    var f: Fake = .{ .minimum = 1 };
    const srv = try Server.init(alloc, t.io, f.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    const user = try message(alloc, "user", "A long document and its question.");
    var initial = [_]json.Value{user};
    var cx: errors.Cx = .{ .a = alloc };
    const prompt = @import("prompt.zig");
    const before = try prompt.prepare(srv, &cx, .{ .messages = .{ .array = &initial }, .fields = .null }, true, null);
    try t.expectEqual(before.history_len, before.user_end);
    try t.expect(before.history_len > 0 and before.history_len < before.ids.len);
    var endpoints: [2][]const u32 = undefined;
    for ([_]bool{ false, true }, 0..) |kept, i| {
        const reply = try message(alloc, "assistant", "Fixed reply.");
        if (kept) try reply.object.put(alloc, "reasoning_content", .{ .string = "Retained reasoning." });
        var messages = [_]json.Value{ user, reply, try message(alloc, "user", "Next question?") };
        const later = try prompt.prepare(srv, &cx, .{ .messages = .{ .array = &messages }, .fields = .null }, true, null);
        try t.expectEqualSlices(u32, before.ids[0..before.history_len], later.ids[0..before.history_len]);
        endpoints[i] = later.ids;
    }
    const same = std.mem.indexOfDiff(u32, endpoints[0], endpoints[1]).?;
    try t.expect(same >= before.history_len and same < endpoints[1].len);
}

test "turn ends: rendered warm and chat requests share the inferred complete user-header cut" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var files = try fixture.Fixture.init(alloc, 0);
    defer files.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(alloc, t.io, files.path, null, alloc, &problem);
    defer hf.deinit();
    var f: Fake = .{ .minimum = 1, .ends = true, .capture = true };
    defer if (f.captured) |c| {
        a.free(c.ids);
        a.free(c.cuts);
    };
    const srv = try Server.init(alloc, t.io, f.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    srv.info.warm_turns = true;
    var messages: std.ArrayList(json.Value) = .empty;
    try messages.append(alloc, try message(alloc, "user", "First question?"));
    const reply = try message(alloc, "assistant", "Fixed reply.");
    const body = try json.newObject(alloc);
    try body.put(alloc, "messages", .{ .array = messages.items });
    var cx: errors.Cx = .{ .a = alloc };
    const input = try openai.chatInput(srv, &cx, .{ .object = body });
    const initial = try chat.prepare(srv, &cx, input, Never{});
    defer srv.release(initial.preparing, initial.request);
    try t.expect(std.mem.indexOfScalar(u32, initial.request.chunks, @intCast(initial.request.prompt.len)) != null);
    @import("warm.zig").reply(srv, input, reply, false, null, initial.request.prompt.len);
    const warm = f.captured orelse return error.MissingWarm;
    try t.expect(std.mem.indexOfScalar(u32, warm.cuts, @intCast(warm.ids.len)) != null);
    try messages.append(alloc, reply);
    try messages.append(alloc, try message(alloc, "user", "Next question?"));
    try body.put(alloc, "messages", .{ .array = messages.items });
    const next_input = try openai.chatInput(srv, &cx, .{ .object = body });
    const next = try chat.prepare(srv, &cx, next_input, Never{});
    defer srv.release(next.preparing, next.request);
    try t.expectEqualSlices(u32, warm.ids, next.request.prompt[0..warm.ids.len]);
    var n: usize = 0;
    for (next.request.chunks) |cut| {
        if (cut <= warm.ids.len) n += 1;
    }
    try t.expectEqualSlices(u32, warm.cuts, next.request.chunks[0..n]);
}
