//! Real chat histories through the real prepare path: an edited tool result resumes at its message's start (#3679).
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
    const Saved = struct { at: u32, sum: u64 };
    fn engine(f: *@This()) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 65536, .prefill_step = 2048, .prompt_cache = true, .prefill_min_chunk = 1 };
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
    fn bytes(_: *anyopaque, _: u32) u64 {
        return @sizeOf(Saved);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f: *@This() = @ptrCast(@alignCast(ptr));
        try t.expectEqual(at, f.at);
        const s = try a.create(Saved);
        s.* = .{ .at = at, .sum = f.sum };
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

/// system, user, one assistant call of four tools, four tool results, then the final question.
fn history(alloc: std.mem.Allocator, last_value: []const u8, qwen: bool) !json.Value {
    var out: std.ArrayList(u8) = .empty;
    const w = &out;
    try w.appendSlice(alloc, "[{\"role\":\"system\",\"content\":\"You are a careful agent.\"},{\"role\":\"user\",\"content\":\"Read the four logs.\"},");
    try w.appendSlice(alloc, "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[");
    for (0..4) |i| try w.print(alloc, "{s}{{\"id\":\"call_{d}\",\"type\":\"function\",\"function\":{{\"name\":\"read_file\",\"arguments\":{s}}}}}", .{ if (i > 0) "," else "", i, if (qwen) "{\"path\":\"a.log\"}" else "\"{}\"" });
    try w.appendSlice(alloc, "]}");
    for (0..4) |i| {
        try w.print(alloc, ",{{\"role\":\"tool\",\"tool_call_id\":\"call_{d}\",\"content\":\"log {d} value ", .{ i, i });
        try w.appendSlice(alloc, if (i == 3) last_value else "11");
        for (0..400) |k| try w.print(alloc, " line{d}x{d}", .{ i, k });
        try w.appendSlice(alloc, "\"}");
    }
    try w.appendSlice(alloc, ",{\"role\":\"user\",\"content\":\"What did the logs say?\"}]");
    return (try json.parse(alloc, out.items)).ok;
}

const Run = struct { from: u32, starts: []const u32, prompt: []const u32, marks: []const u32 };

fn prepare(srv: *Server, alloc: std.mem.Allocator, f: *Fake, store: *pc.Store, value: []const u8, qwen: bool) !Run {
    var cx: errors.Cx = .{ .a = alloc };
    const body = try json.newObject(alloc);
    try body.put(alloc, "messages", try history(alloc, value, qwen));
    try body.put(alloc, "max_tokens", try json.intValue(alloc, 1));
    const input = try openai.chatInput(srv, &cx, .{ .object = body });
    const prepared = try chat.prepare(srv, &cx, input, Never{});
    defer srv.release(prepared.preparing, prepared.request);
    const r = prepared.request;
    const previous = store.message_starts;
    store.message_starts = r.message_starts; // as the lane host scopes it around the lookup
    defer store.message_starts = previous;
    const plan = try store.beginUser(alloc, r.prompt, r.history_len, r.rewind_len, r.user_end, r.shared_prefixes, r.chunks, null, r.decode_spans);
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
    return .{ .from = plan.from, .starts = try alloc.dupe(u32, r.message_starts), .prompt = try alloc.dupe(u32, r.prompt), .marks = try alloc.dupe(u32, plan.marks) };
}

const Outcome = struct { from: u32, edit_start: u32, first_diff: u32, in_starts: bool, head: [32]u8 };
const Template = enum { chatml, qwen38 };

/// Serve the history, then the same one with its last tool result edited, under `message_starts` kept starts.
fn drive(template: Template, message_starts: u8) !Outcome {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    const family = @import("thinking_family_fixture.zig");
    var files = if (template == .qwen38)
        try family.init(a, t.io, @embedFile("fixtures/thinking/qwen38.jinja"), @embedFile("fixtures/thinking/qwen38-tokenizer_config.json"), "qwen3_5")
    else
        try family.init(a, t.io, @embedFile("fixtures/control_tokens/chatml.jinja"), "{}", "qwen3_5");
    defer files.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(alloc, t.io, files.path, null, alloc, &problem);
    defer hf.deinit();
    var f: Fake = .{};
    const srv = try Server.init(alloc, t.io, f.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    // The Qwen3.8-27B Metal rules of qwen27_served.cacheRules(2048, message_starts): native is not imported here.
    const rules: pc.Rules = .{ .warm = .on, .chunk = 2048, .chunk_marks = false, .message_starts = message_starts };
    var store = pc.Store.init(a, f.snapshots(), rules, @sizeOf(Fake.Saved) * 16);
    defer store.deinit();
    store.most = 16;
    const qwen = template == .qwen38;
    const first = try prepare(srv, alloc, &f, &store, "17", qwen);
    try t.expectEqual(@as(u32, 0), first.from);
    const edited = try prepare(srv, alloc, &f, &store, "19", qwen);
    const diff: u32 = @intCast(std.mem.indexOfDiff(u32, first.prompt, edited.prompt).?);
    var start: u32 = 0;
    for (edited.starts) |s| {
        if (s <= diff and s > start) start = s;
    }
    var out: Outcome = .{ .from = edited.from, .edit_start = start, .first_diff = diff, .head = @splat(0), .in_starts = std.mem.indexOfScalar(u32, edited.starts, edited.from) != null };
    const head = try srv.text.decode(alloc, edited.prompt[start..@min(edited.prompt.len, start + out.head.len)]);
    @memcpy(out.head[0..head.len], head);
    return out;
}

fn expectResume(template: Template, header: []const u8) !void {
    const o = try drive(template, 4);
    try t.expect(o.edit_start > 0 and o.edit_start < o.first_diff);
    try t.expect(std.mem.startsWith(u8, &o.head, header));
    try t.expectEqual(o.edit_start, o.from);
    try t.expect(o.in_starts);
    const off = try drive(template, 0); // the negative control: main's own marks, none of them at the message
    try t.expectEqual(o.edit_start, off.edit_start);
    try t.expect(off.from < off.edit_start);
}

test "tool edit starts: an edited tool result resumes at that tool message's start" {
    try expectResume(.chatml, "<|im_start|>tool\n");
}

test "tool edit starts: Qwen3.8's template resumes an edited tool result at its recorded start" {
    try expectResume(.qwen38, "\n<tool_response>\nlog 3 value 19");
}
