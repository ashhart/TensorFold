//! cache_salt end to end: the lane host resumes only the same salt's states; the routes salt the key, or leave it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const json = @import("json");
const server = @import("root.zig");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const errors = @import("errors.zig");
const cache_salt = @import("cache_salt.zig");
const Conn = @import("http_conn.zig").Conn;
const pc = api.prompt_cache;
const gpa = std.testing.allocator;
const io = std.testing.io;

const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores; // lane-core backends restore inside their own prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?api.Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs, .prompt_logprobs, .audio => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *@This()) api.Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// One request run to its end under `salt` (0: none): its tokens (the caller frees them) and where it resumed.
fn run(e: api.Engine, id: api.Id, prompt: []const u32, history: u32, max: u32, salt: u64) !struct { tokens: std.ArrayList(u32), cached: ?u32 } {
    const key: []const u32 = if (salt != 0) try cache_salt.salted(gpa, prompt, salt) else &.{};
    defer if (salt != 0) gpa.free(key);
    const r: api.Request = .{ .prompt = prompt, .max_tokens = max, .history_len = history, .cache_key = key, .salted = salt != 0 };
    var b: Box = .{};
    try e.submit(id, &r, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, b.wait());
    return .{ .tokens = b.tokens, .cached = b.cached };
}

const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const salt_a: u64 = 0xa11ce;
const salt_b: u64 = 0xb0b;

test "two salts never resume each other's states, the same salt resumes as an unsalted turn does, tokens unchanged" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const e = host.engine();
    try std.testing.expect(e.info().cache_salt);
    // the plain conversation, as on main: turn 2 resumes the 8 tokens turn 1 kept
    var p1 = try run(e, 1, &t1, 8, 6, 0);
    defer p1.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), p1.cached);
    const t2 = try std.mem.concat(gpa, u32, &.{ &t1, p1.tokens.items[0..4], &.{ 7, 7, 5, 3 } });
    defer gpa.free(t2);
    const h2: u32 = @intCast(t2.len - 2);
    var p2 = try run(e, 2, t2, h2, 12, 0);
    defer p2.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 8), p2.cached);
    // salt A's first turn: the plain states are not its own
    var a1 = try run(e, 3, &t1, 8, 6, salt_a);
    defer a1.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), a1.cached);
    try std.testing.expectEqualSlices(u32, p1.tokens.items, a1.tokens.items);
    // salt B's second turn: neither A's states nor the plain ones
    var b2 = try run(e, 4, t2, h2, 12, salt_b);
    defer b2.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), b2.cached);
    // salt A's second turn resumes A's first turn, at the same place the plain turn did
    var a2 = try run(e, 5, t2, h2, 12, salt_a);
    defer a2.tokens.deinit(gpa);
    try std.testing.expectEqual(p2.cached, a2.cached);
    // salts never change tokens
    try std.testing.expectEqualSlices(u32, p2.tokens.items, b2.tokens.items);
    try std.testing.expectEqualSlices(u32, p2.tokens.items, a2.tokens.items);
    // a third salt, and no salt, after all of them: each resumes only its own
    var c2 = try run(e, 6, t2, h2, 12, 0xc0ffee);
    defer c2.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), c2.cached);
    var again = try run(e, 7, t2, h2, 12, 0);
    defer again.tokens.deinit(gpa);
    try std.testing.expect(again.cached.? >= 8); // the plain turns' own states
}

test "a salted text prompt counts as a repeat under its own key, as a plain one does under its tokens" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0, .repeat_min = 4 }, 1 << 20);
    defer store.deinit();
    var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const key = try cache_salt.salted(gpa, &t1, salt_a);
    defer gpa.free(key);
    try std.testing.expectEqual(@as(u32, 0), store.repeat_log.mark(4, key, 0, 0, false));
    var once = try run(host.engine(), 1, &t1, 0, 6, salt_a);
    defer once.tokens.deinit(gpa);
    try std.testing.expect(store.repeat_log.mark(4, key, 0, 0, false) > 0); // its key, never its plain tokens
    try std.testing.expectEqual(@as(u32, 0), store.repeat_log.mark(4, &t1, 0, 0, false));
}

/// Bytes as tokens; every prompt renders as "prompt".
const Text = struct {
    fn text(x: *@This()) model_text.Text {
        return .{ .ctx = x, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .token_bytes = tokenBytes } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @truncate(id);
        return out;
    }
    fn tokenBytes(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error!?[]u8 {
        return try a.dupe(u8, &.{@as(u8, @truncate(id))});
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return a.dupe(u8, &.{@as(u8, @truncate(id))});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that keeps prompt states (keyed by cache key, or not) and records each request's cache key.
const Keys = struct {
    keys: bool = true,
    got: ?[]u32 = null,
    salted: bool = false,
    fn engine(k: *@This()) api.Engine {
        return .{ .ctx = k, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const k: *Keys = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .prompt_cache = true, .cache_salt = k.keys };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const k: *Keys = @ptrCast(@alignCast(ctx));
        if (k.got) |g| gpa.free(g);
        k.got = gpa.dupe(u32, r.cache_key) catch return error.Busy;
        k.salted = r.salted;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Sent = struct {
    status: u16 = 0,
    fn out(s: *@This()) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const s: *Sent = @ptrCast(@alignCast(ctx));
        s.status = status;
    }
};

/// The cache key a chat request reached the engine with (testing-allocator owned), and its reply's status.
fn keyOf(k: *Keys, body: []const u8, per_key: bool, digest: ?[32]u8) !struct { key: []u32, status: u16, salted: bool } {
    var text: Text = .{};
    var secret = @import("ids.zig").Key.of("one server's salt key"); // Server.init takes it: every call's server keys alike
    var srv = try server.Server.init(gpa, io, k.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .cache_salt_per_key = per_key, .tool_id_key = &secret }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .key_digest = digest };
    var sent: Sent = .{};
    if (k.got) |g| gpa.free(g);
    k.got = null;
    k.salted = false;
    openai.run(srv, a, sent.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    const got = k.got orelse try gpa.alloc(u32, 0);
    k.got = null;
    return .{ .key = got, .status = sent.status, .salted = k.salted };
}

const chat = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4";

test "the routes: no salt sends today's key, a salt keys the request, the per-key option isolates keys" {
    var k: Keys = .{};
    defer if (k.got) |g| gpa.free(g);
    const plain = try keyOf(&k, chat ++ "}", false, null);
    defer gpa.free(plain.key);
    try std.testing.expectEqual(@as(usize, 0), plain.key.len); // no salt: the request is as on main
    const a1 = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-a\"}", false, null);
    defer gpa.free(a1.key);
    const a2 = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-a\"}", false, null);
    defer gpa.free(a2.key);
    const b = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-b\"}", false, null);
    defer gpa.free(b.key);
    try std.testing.expectEqual(@as(usize, "prompt".len), a1.key.len);
    try std.testing.expectEqualSlices(u32, a1.key, a2.key); // the same salt, the same key
    try std.testing.expect(!std.mem.eql(u32, a1.key[0..4], b.key[0..4]));
    try std.testing.expectEqualSlices(u8, "pt", &.{ @as(u8, @truncate(a1.key[4])), @as(u8, @truncate(a1.key[5])) });
    for (a1.key[0..4]) |w| try std.testing.expect(w & 0x8000_0000 != 0);
    // --cache-salt-per-key: two API keys' requests keyed apart; one key's requests together; off, no key change
    const d1: [32]u8 = @splat(1);
    const d2: [32]u8 = @splat(2);
    const k1 = try keyOf(&k, chat ++ "}", true, d1);
    defer gpa.free(k1.key);
    const k1b = try keyOf(&k, chat ++ "}", true, d1);
    defer gpa.free(k1b.key);
    const k2 = try keyOf(&k, chat ++ "}", true, d2);
    defer gpa.free(k2.key);
    try std.testing.expectEqualSlices(u32, k1.key, k1b.key);
    try std.testing.expect(k1.key.len == 6 and !std.mem.eql(u32, k1.key[0..4], k2.key[0..4]));
    const off = try keyOf(&k, chat ++ "}", false, d1);
    defer gpa.free(off.key);
    try std.testing.expectEqual(@as(usize, 0), off.key.len);
    const both = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-a\"}", true, d1);
    defer gpa.free(both.key);
    try std.testing.expect(!std.mem.eql(u32, both.key[0..4], a1.key[0..4]) and !std.mem.eql(u32, both.key[0..4], k1.key[0..4]));
    // the same client salt on another key: apart from key 1's, and from key 2's own requests
    const both2 = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-a\"}", true, d2);
    defer gpa.free(both2.key);
    try std.testing.expect(!std.mem.eql(u32, both2.key[0..4], both.key[0..4]) and !std.mem.eql(u32, both2.key[0..4], k2.key[0..4]));
    // the engine is told which requests are salted (Recall skips them); an unsalted one is as on main
    try std.testing.expect(!plain.salted and !off.salted and a1.salted and k1.salted and both2.salted);
}

test "the routes: Messages passes the salt on, and an imprint ask looks up a salted request by its salted key" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const body = (try json.parse(cx.a, "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"cache_salt\":\"tenant-a\"}")).ok;
    const translated = try @import("anthropic_translate.zig").translate(&cx, body, false);
    try std.testing.expectEqualStrings("tenant-a", translated.strField("cache_salt").?);
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const key = try cache_salt.salted(cx.a, &prompt, salt_a);
    const imprint = @import("imprint_routes.zig");
    try std.testing.expectEqualSlices(u32, &prompt, imprint.askKey(.{ .prompt = &prompt, .max_tokens = 1 }));
    try std.testing.expectEqualSlices(u32, key, imprint.askKey(.{ .prompt = &prompt, .max_tokens = 1, .cache_key = key, .salted = true }));
}

test "the routes: an engine keyed by tokens alone refuses a salt, a malformed salt is refused, Responses passes it on" {
    var k: Keys = .{ .keys = false };
    defer if (k.got) |g| gpa.free(g);
    const refused = try keyOf(&k, chat ++ ",\"cache_salt\":\"tenant-a\"}", false, null);
    defer gpa.free(refused.key);
    try std.testing.expectEqual(@as(u16, 400), refused.status);
    const plain = try keyOf(&k, chat ++ "}", false, null); // without a salt it serves as before
    defer gpa.free(plain.key);
    try std.testing.expectEqual(@as(u16, 200), plain.status);
    k.keys = true;
    const empty = try keyOf(&k, chat ++ ",\"cache_salt\":\"\"}", false, null);
    defer gpa.free(empty.key);
    try std.testing.expectEqual(@as(u16, 400), empty.status);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const body = (try json.parse(cx.a, "{\"model\":\"m\",\"input\":\"x\",\"cache_salt\":\"tenant-a\"}")).ok;
    const NoHistory = struct {
        pub fn conversation(_: @This(), c: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
            return c.refuse("no stored responses in this test");
        }
    };
    const t = try @import("responses_translate.zig").translate(&cx, body, NoHistory{});
    try std.testing.expectEqualStrings("tenant-a", t.chat.strField("cache_salt").?);
}
