//! Host checks for opt-in compaction. The fake engine records each prompt and answers OK.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const compact = @import("compact.zig");
const openai = @import("openai.zig");
const server_mod = @import("server.zig");
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const io = std.testing.io;

const Stay = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
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

    fn render(_: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
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
        return buf.items;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

const Eng = struct {
    window: u32,
    prompts: std.ArrayList([]u8) = .empty,
    opened: bool = false,
    early: usize = 0,
    late: usize = 0,

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
        if (e.opened) e.late += 1 else e.early += 1;
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

fn start(eng: *Eng, text: *Text, at: ?server_mod.CompactAt, keep: ?u32, mem: ?[]const u8) !*server_mod.Server {
    return server_mod.Server.init(gpa, io, eng.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = true,
        .default_max_tokens = 32,
        .compact_at = at,
        .compact_keep = keep,
        .compact_memory = mem,
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

fn noteOf(reply: chat.Reply) ?[]const u8 {
    const c = reply.runtime.get("compaction") orelse return null;
    const n = c.get("note") orelse return null;
    return if (n == .string) n.string else null;
}

fn seenFrom(e: *Eng, from: usize, needle: []const u8) bool {
    for (e.prompts.items[from..]) |p| if (std.mem.indexOf(u8, p, needle) != null) return true;
    return false;
}

fn countFrom(e: *Eng, from: usize, needle: []const u8) usize {
    var n: usize = 0;
    for (e.prompts.items[from..]) |p| {
        if (std.mem.indexOf(u8, p, needle) != null) n += 1;
    }
    return n;
}

fn tmpDir(a: Allocator, tmp: std.testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, a);
    defer a.free(cwd);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "auto reserve, note budget and keep tokens" {
    try std.testing.expectEqual(@as(i64, 16384), compact.reserveTokens(1000, .auto));
    try std.testing.expectEqual(@as(i64, 30000), compact.reserveTokens(200000, .auto));
    try std.testing.expectEqual(@as(i64, 8192), compact.noteBudgetTokens(200000, .auto));
    try std.testing.expectEqual(@as(i64, 1000), compact.reserveTokens(2000, .{ .fraction = 0.5 }));
    try std.testing.expectEqual(@as(i64, 800), compact.noteBudgetTokens(2000, .{ .fraction = 0.5 }));
    try std.testing.expectEqual(@as(usize, 20000), compact.keepTokens(100000, null));
    try std.testing.expectEqual(@as(usize, 100), compact.keepTokens(400, null));
    try std.testing.expectEqual(@as(usize, 7), compact.keepTokens(400, 7));
}

test "off leaves the runtime object unchanged and a short request only reports the window" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try start(&eng, &text, null, null, null);
    defer off.deinit();
    var cx: errors.Cx = .{ .a = a };
    const plain = try go(off, a, &.{try userMsg(a, "hello")}, 8, &cx);
    try std.testing.expect(plain.runtime.get("context_window") == null);
    try std.testing.expect(plain.runtime.get("compaction") == null);
    const on = try start(&eng, &text, .{ .fraction = 0.9 }, null, null);
    defer on.deinit();
    const light = try go(on, a, &.{try userMsg(a, "hello")}, 8, &cx);
    try std.testing.expect(light.runtime.get("context_window") != null);
    try std.testing.expect(light.runtime.get("context_used") != null);
    try std.testing.expect(light.runtime.get("compaction") == null);
}

test "the kept span never starts between a tool call and its results" {
    var eng: Eng = .{ .window = 200 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, null, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try json.newObject(a);
    try args.put(a, "path", .{ .string = "src/app.zig" });
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = "read" });
    try function.put(a, "arguments", .{ .object = args });
    const call = try json.newObject(a);
    try call.put(a, "function", .{ .object = function });
    const calls = try a.alloc(Value, 1);
    calls[0] = .{ .object = call };
    const assistant = try json.newObject(a);
    try assistant.put(a, "role", .{ .string = "assistant" });
    try assistant.put(a, "content", .{ .string = "aa" });
    try assistant.put(a, "tool_calls", .{ .array = calls });
    const tool_a = try json.newObject(a);
    try tool_a.put(a, "role", .{ .string = "tool" });
    try tool_a.put(a, "content", .{ .string = "t" });
    const tool_b = try json.newObject(a);
    try tool_b.put(a, "role", .{ .string = "tool" });
    try tool_b.put(a, "content", .{ .string = "t" });
    const msgs = [_]Value{ try userMsg(a, "uuuu"), .{ .object = assistant }, .{ .object = tool_a }, .{ .object = tool_b }, try userMsg(a, "z") };
    try std.testing.expectEqual(@as(usize, 1), try compact.keptStart(srv, a, &msgs, 8, null));
}

test "the compacted prompt fits and the same request compacts the same way" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const msgs = [_]Value{ try userMsg(a, try pads(a, "BIG-", 1200)), try userMsg(a, "TAIL") };
    var cx: errors.Cx = .{ .a = a };
    const first = try go(srv, a, &msgs, 32, &cx);
    const note = noteOf(first).?;
    try std.testing.expect(first.prompt_tokens + 32 <= 2000);
    try std.testing.expect(first.runtime.get("compaction") != null);
    const second = try go(srv, a, &msgs, 32, &cx);
    try std.testing.expectEqualStrings(note, noteOf(second).?);
}

test "a request that cannot fit after compaction is still context_length_exceeded" {
    var eng: Eng = .{ .window = 80 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, null, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const msgs = [_]Value{try userMsg(a, try pads(a, "NO-", 200))};
    try std.testing.expectError(error.Refused, go(srv, a, &msgs, 32, &cx));
    try std.testing.expectEqual(errors.Kind.context_length, cx.kind);
    try std.testing.expect(std.mem.startsWith(u8, cx.message, errors.context_limit));
}

test "a split turn is summarized on its own and placed before the kept reply" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 20, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const assistant = try json.newObject(a);
    try assistant.put(a, "role", .{ .string = "assistant" });
    try assistant.put(a, "content", .{ .string = "KEPT-ASSISTANT" });
    const msgs = [_]Value{ try userMsg(a, try pads(a, "REQUEST-ALPHA-", 1200)), .{ .object = assistant } };
    var cx: errors.Cx = .{ .a = a };
    _ = try go(srv, a, &msgs, 32, &cx);
    try std.testing.expect(seenFrom(&eng, 0, "Summarize the cut part"));
    try std.testing.expect(seenFrom(&eng, 0, "REQUEST-ALPHA-"));
    try std.testing.expect(seenFrom(&eng, 0, "Earlier in this turn:"));
    try std.testing.expect(seenFrom(&eng, 0, "KEPT-ASSISTANT"));
}

test "the files section is filled from tool call arguments" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try json.newObject(a);
    try args.put(a, "path", .{ .string = "src/app.zig" });
    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = "read" });
    try function.put(a, "arguments", .{ .object = args });
    const call = try json.newObject(a);
    try call.put(a, "function", .{ .object = function });
    const raw = try json.newObject(a);
    try raw.put(a, "name", .{ .string = "write" });
    try raw.put(a, "arguments", .{ .string = "{\"file_path\":\"src/lib.zig\"}" });
    const call_b = try json.newObject(a);
    try call_b.put(a, "function", .{ .object = raw });
    const calls = try a.alloc(Value, 2);
    calls[0] = .{ .object = call };
    calls[1] = .{ .object = call_b };
    const assistant = try json.newObject(a);
    try assistant.put(a, "role", .{ .string = "assistant" });
    try assistant.put(a, "content", .{ .string = "looked" });
    try assistant.put(a, "tool_calls", .{ .array = calls });
    const tool = try json.newObject(a);
    try tool.put(a, "role", .{ .string = "tool" });
    try tool.put(a, "content", .{ .string = "data" });
    const msgs = [_]Value{ try userMsg(a, try pads(a, "READ-", 1200)), .{ .object = assistant }, .{ .object = tool }, try userMsg(a, "TAIL") };
    var cx: errors.Cx = .{ .a = a };
    const reply = try go(srv, a, &msgs, 32, &cx);
    const note = noteOf(reply).?;
    try std.testing.expect(std.mem.indexOf(u8, note, "Files read and changed") != null);
    try std.testing.expect(std.mem.indexOf(u8, note, "src/app.zig") != null);
    try std.testing.expect(std.mem.indexOf(u8, note, "src/lib.zig") != null);
}

test "a returned note is updated from the new middle" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const marked = try std.fmt.allocPrint(a, "{s}\nSTORED-NOTE", .{compact.marker_line});
    const msgs = [_]Value{ try userMsg(a, "ANCIENT-TEXT"), try userMsg(a, marked), try userMsg(a, try pads(a, "NEW-CUT-", 1200)), try userMsg(a, "TAIL") };
    var cx: errors.Cx = .{ .a = a };
    _ = try go(srv, a, &msgs, 32, &cx);
    try std.testing.expect(seenFrom(&eng, 0, "STORED-NOTE"));
    try std.testing.expect(seenFrom(&eng, 0, "NEW-CUT-"));
    try std.testing.expect(!seenFrom(&eng, 0, "ANCIENT-TEXT"));
}

test "a long cut is summarized in chunks" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const msgs = [_]Value{ try userMsg(a, try pads(a, "CHUNK-ONE-", 700)), try userMsg(a, try pads(a, "CHUNK-TWO-", 700)), try userMsg(a, "TAIL") };
    var cx: errors.Cx = .{ .a = a };
    _ = try go(srv, a, &msgs, 32, &cx);
    try std.testing.expectEqual(@as(usize, 2), countFrom(&eng, 0, "Rewrite the memory note"));
    try std.testing.expect(seenFrom(&eng, 0, "CHUNK-ONE-"));
    try std.testing.expect(seenFrom(&eng, 0, "CHUNK-TWO-"));
    try std.testing.expect(seenFrom(&eng, 0, "Previous note:"));
}

test "an incremental request summarizes only the new middle and a user edit is the next note" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmpDir(a, tmp);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, dir);
    defer srv.deinit();
    const old = try userMsg(a, try pads(a, "OLD-MIDDLE-", 1200));
    const tail = try userMsg(a, "TAIL");
    var cx: errors.Cx = .{ .a = a };
    const first_msgs = [_]Value{ old, tail };
    _ = try go(srv, a, &first_msgs, 32, &cx);
    try std.testing.expect(seenFrom(&eng, 0, "OLD-MIDDLE-"));
    const key = try compact.conversationKey(a, &first_msgs);
    const path = try std.fmt.allocPrint(a, "{s}/{s}.md", .{ dir, key[0..] });
    const saved = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, saved, "OK") != null);
    const fresh = eng.prompts.items.len;
    const newer = [_]Value{ old, try userMsg(a, try pads(a, "NEW-MIDDLE-", 1200)), try userMsg(a, "TAIL2") };
    _ = try go(srv, a, &newer, 32, &cx);
    try std.testing.expect(seenFrom(&eng, fresh, "NEW-MIDDLE-"));
    try std.testing.expect(!seenFrom(&eng, fresh, "OLD-MIDDLE-"));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "USER-EDITED-NOTE" });
    const edited = eng.prompts.items.len;
    _ = try go(srv, a, &newer, 32, &cx);
    try std.testing.expect(seenFrom(&eng, edited, "USER-EDITED-NOTE"));
}

const Watch = struct {
    eng: *Eng,
    opened: bool = false,
    saw: bool = false,

    fn out(w: *Watch) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const w: *Watch = @ptrCast(@alignCast(ctx));
        w.opened = true;
        w.eng.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const w: *Watch = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        if (p.get("tensorfold")) |t| {
            if (t.get("compaction") != null) w.saw = true;
        }
    }

    fn reply(ctx: *anyopaque, _: u16, payload: Value) void {
        const w: *Watch = @ptrCast(@alignCast(ctx));
        if (payload.get("tensorfold")) |t| {
            if (t.get("compaction") != null) w.saw = true;
        }
    }
};

test "a stream compacts before it opens" {
    var eng: Eng = .{ .window = 2000 };
    var text: Text = .{};
    defer freePrompts(&eng);
    const srv = try start(&eng, &text, .{ .fraction = 0.5 }, 8, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const big = try pads(a, "STREAM-", 1200);
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"stream\":true,\"max_tokens\":32,\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}},{{\"role\":\"user\",\"content\":\"TAIL\"}}]}}", .{big});
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var watch: Watch = .{ .eng = &eng };
    openai.run(srv, a, watch.out(), .{ .conn = &conn }, true, raw);
    try std.testing.expect(watch.opened);
    try std.testing.expect(watch.saw);
    try std.testing.expect(eng.early >= 1);
    try std.testing.expect(eng.late >= 1);
}
