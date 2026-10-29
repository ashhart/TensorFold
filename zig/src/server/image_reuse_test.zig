//! Image chats through the server, the lane host and its prompt cache over the fake target: reuse and image identity.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const json = @import("json");
const fixture = @import("control_tokens_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const pc = api.prompt_cache;
const t = std.testing;
const Allocator = std.mem.Allocator;

/// A Qwen-like template: each message's text and image parts in order, an image part as its one `<|image|>`.
const template =
    \\{% for m in messages %}<|im_start|>{{ m['role'] }}
    \\{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'text' %}{{ p['text'] }}{% else %}<|image|>{% endif %}{% endfor %}{% endif %}<|im_end|>
    \\{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
    \\{% endif %}
;

/// A test picture as a data URL: PNG's signature, a name byte, then the rows it takes when its share allows.
fn picture(a: Allocator, name: u8, rows: u8) ![]const u8 {
    const raw = [_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', name, rows };
    const enc = std.base64.standard.Encoder;
    const out = try a.alloc(u8, enc.calcSize(raw.len));
    return std.fmt.allocPrint(a, "data:image/png;base64,{s}", .{enc.encode(out, &raw)});
}

/// The fake processor: at most `max_tokens` rows, named by its bytes and grid as core/picture/identity.zig names one.
fn prepare(_: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const rows: u32 = @min(bytes[9], max_tokens);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    h.update(std.mem.asBytes(&rows));
    var d: [32]u8 = undefined;
    h.final(&d);
    return .{ .pixels = try a.alloc(f32, 0), .gh = 2, .gw = 2 * rows, .tokens = rows, .hash = std.mem.readInt(u64, d[0..8], .little) & ((1 << 62) - 1) };
}

/// A user message with its text, then its pictures; `say` without pictures, `ok` the assistant's short reply.
fn user(a: Allocator, text: []const u8, pics: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"{s}\"}}", .{text});
    for (pics) |p| try out.print(a, ",{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{s}\"}}}}", .{p});
    try out.appendSlice(a, "]}");
    return out.items;
}
fn say(a: Allocator, text: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"role\":\"user\",\"content\":\"{s}\"}}", .{text});
}
const ok = "{\"role\":\"assistant\",\"content\":\"Seen, and noted.\"}";
const system = "{\"role\":\"system\",\"content\":\"You describe pictures carefully.\"}";

/// The messages of a conversation as the client resends them: every earlier turn, whole.
fn conv(a: Allocator, msgs: []const []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "[{s}]", .{try std.mem.join(a, ",", msgs)});
}

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const Turn = struct { cached: usize, prompt: usize, rows: usize, tokens: []const u32, finish: []const u8 };
const Options = struct { image_tokens: u32 = 64, budget: u64 = 1 << 20, strip: bool = false };

/// A served model: the real server and lane host, the fake target (image rows named by their image) and a prompt cache.
const Harness = struct {
    gpa: Allocator,
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    store: pc.Store = undefined,
    host: api.LaneHost = undefined,
    files: fixture.Fixture = undefined,
    hf: *HfText = undefined,
    srv: *Server = undefined,
    markers: [1]u32 = .{0},
    dummy: u8 = 0,
    strip: bool, // the planted control: requests reach the cache without their image key
    stripped: api.Request = undefined,
    cancel_at: ?usize = null, // the next image pass is cancelled at this prompt chunk

    fn init(gpa: Allocator, a: Allocator, o: Options) !*Harness {
        const h = try gpa.create(Harness);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7), .target = .{ .gpa = gpa, .image_words = true, .prefill_chunks = 3 }, .strip = o.strip };
        h.target.prefill_hook = hook;
        h.target.prefill_hook_ctx = h;
        h.core = lanes.Engine.init(gpa, &h.cfg, h.target.backend(), h.clock.clock());
        h.store = pc.Store.init(gpa, .{ .ptr = &h.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, o.budget);
        h.files = try fixture.Fixture.init(a, 0);
        try h.files.tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = template });
        var problem: []const u8 = "";
        h.hf = try HfText.load(gpa, t.io, h.files.path, null, a, &problem);
        h.markers = .{h.hf.text().tokenId("<|image|>").?};
        const vision: api.Vision = .{ .ctx = &h.dummy, .prepare = prepare, .image_token = h.markers[0], .markers = &h.markers, .image_tokens = o.image_tokens, .max_images = 4 };
        h.host = api.LaneHost.init(gpa, t.io, &h.core, .{ .lanes = 2, .vision = vision, .prompt_cache = true });
        h.host.cache = &h.store;
        try h.host.start();
        h.srv = try Server.init(gpa, t.io, h.engine(), h.hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
        return h;
    }

    /// The generation prompt's tokens: a later turn resumes at the earlier turn's prompt less these.
    fn opener(h: *Harness, a: Allocator) !usize {
        return (try h.hf.text().encode(a, "<|im_start|>assistant\n", false)).len;
    }

    fn deinit(h: *Harness) void {
        h.srv.deinit();
        h.host.stop();
        h.store.deinit();
        h.core.deinit();
        h.target.deinit();
        h.cfg.deinit(h.gpa);
        h.hf.deinit();
        h.files.deinit();
        h.gpa.destroy(h);
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores; // the lane backend restores inside its own prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }

    /// A prompt chunk of an image pass: the planted cancellation, as a client leaving mid-prompt does.
    fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        const at = h.cancel_at orelse return;
        if (s.images.len == 0 or chunk != at) return;
        h.cancel_at = null;
        const job: *api.LaneHost.Job = @ptrCast(@alignCast(s.cancel_check.?.ptr));
        h.host.engine().cancel(job.id);
    }

    fn engine(h: *Harness) api.Engine {
        return if (h.strip) .{ .ctx = h, .vtable = &stripped_vtable } else h.host.engine();
    }
    const stripped_vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
    fn info(ctx: *anyopaque) api.Info {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        return h.host.engine().info();
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.stripped = r.*;
        h.stripped.cache_key = &.{};
        return h.host.engine().submit(id, &h.stripped, sink);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.host.engine().cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, tokens: []u32) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.host.engine().status(out, tokens);
    }
    fn memory(ctx: *anyopaque, reset: bool) ?api.Memory {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        return h.host.engine().memory(reset);
    }

    /// One chat request: its cached prompt tokens, the image rows its pass fed, and its reply's token ids.
    fn ask(h: *Harness, a: Allocator, messages: []const u8) !Turn {
        const before = h.target.image_rows;
        var cx: Cx = .{ .a = a };
        const fields = try json.newObject(a);
        try fields.put(a, "return_token_ids", .{ .bool = true });
        const r = try chat.run(h.srv, &cx, .{ .messages = (try json.parse(a, messages)).ok, .max_tokens = 6, .fields = .{ .object = fields } }, null, Gone{});
        const ids: []const json.Value = if (r.runtime == .object) (if (r.runtime.get("token_ids")) |v| v.array else &.{}) else &.{};
        const out = try a.alloc(u32, ids.len);
        for (out, ids) |*o, v| o.* = @intCast(v.int64().?);
        return .{ .cached = r.cached_tokens, .prompt = r.prompt_tokens, .rows = h.target.image_rows - before, .tokens = out, .finish = r.finish_reason };
    }
};

/// The reply a new server gives the same messages from an empty cache.
fn fresh(a: Allocator, o: Options, messages: []const u8) !Turn {
    var no_strip = o;
    no_strip.strip = false;
    const h = try Harness.init(t.allocator, a, no_strip);
    defer h.deinit();
    const turn = try h.ask(a, messages);
    try t.expectEqual(@as(usize, 0), turn.cached);
    return turn;
}

/// `turn` resumed or not, its reply equals a fresh server's.
fn expectFresh(a: Allocator, o: Options, messages: []const u8, turn: Turn) !void {
    try t.expectEqualSlices(u32, (try fresh(a, o, messages)).tokens, turn.tokens);
}

test "image reuse: text turns after an image resume past it, feed no image rows and equal a fresh server" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{});
    defer h.deinit();
    const pic = try picture(a, 'a', 12);
    var msgs: std.ArrayList([]const u8) = .empty;
    try msgs.appendSlice(a, &.{ system, try user(a, "What is in this picture?", &.{pic}) });
    const first = try h.ask(a, try conv(a, msgs.items));
    try t.expectEqual(@as(usize, 0), first.cached);
    try t.expectEqual(@as(usize, 12), first.rows);
    var last = first.prompt;
    const gen = try h.opener(a);
    for ([_][]const u8{ "And the colours?", "Is anyone in it?", "Summarise it." }) |follow| {
        try msgs.appendSlice(a, &.{ ok, try say(a, follow) });
        const m = try conv(a, msgs.items);
        const turn = try h.ask(a, m);
        try t.expectEqual(@as(usize, 0), turn.rows); // the image is in the kept state: no row of it runs again
        try t.expectEqual(last - gen, turn.cached); // the whole earlier history resumes
        try expectFresh(a, .{}, m, turn);
        last = turn.prompt;
    }
}

test "image reuse: another picture under the same placeholders never resumes past them, in or across conversations" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{});
    defer h.deinit();
    const first = try h.ask(a, try conv(a, &.{ system, try user(a, "Look at this.", &.{try picture(a, 'a', 12)}) }));
    try t.expectEqual(@as(usize, 12), first.rows);
    // a new conversation: the same words, another picture of as many rows
    const other = try conv(a, &.{ system, try user(a, "Look at this.", &.{try picture(a, 'b', 12)}) });
    const second = try h.ask(a, other);
    try t.expectEqual(@as(usize, 12), second.rows);
    try t.expect(second.cached < first.prompt - 12);
    try expectFresh(a, .{}, other, second);
    // the first conversation's next turn with its picture edited out for a third one
    const edited = try conv(a, &.{ system, try user(a, "Look at this.", &.{try picture(a, 'c', 12)}), ok, try say(a, "Again?") });
    const third = try h.ask(a, edited);
    try t.expectEqual(@as(usize, 12), third.rows);
    try expectFresh(a, .{}, edited, third);
}

test "image reuse: a repeated picture, and two pictures in one turn, resume what is kept and run only new rows" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{});
    defer h.deinit();
    const pa = try picture(a, 'a', 10);
    const pb = try picture(a, 'b', 14);
    var msgs: std.ArrayList([]const u8) = .empty;
    try msgs.appendSlice(a, &.{ system, try user(a, "Two pictures.", &.{ pa, pb }) });
    try t.expectEqual(@as(usize, 24), (try h.ask(a, try conv(a, msgs.items))).rows);
    try msgs.appendSlice(a, &.{ ok, try say(a, "Which is brighter?") });
    const text_turn = try conv(a, msgs.items);
    const resumed = try h.ask(a, text_turn);
    try t.expectEqual(@as(usize, 0), resumed.rows);
    try expectFresh(a, .{}, text_turn, resumed);
    // the first picture sent again in a later turn: its new rows run, the earlier ones stay kept
    try msgs.appendSlice(a, &.{ ok, try user(a, "This one again.", &.{pa}) });
    const again = try conv(a, msgs.items);
    const repeated = try h.ask(a, again);
    try t.expectEqual(@as(usize, 10), repeated.rows);
    try t.expectEqual(resumed.prompt - try h.opener(a), repeated.cached);
    try expectFresh(a, .{}, again, repeated);
}

test "image reuse: a picture whose shared budget shrinks when a turn adds one is prepared again, not resumed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{ .image_tokens = 64 });
    defer h.deinit();
    const big = try picture(a, 'a', 40);
    var msgs: std.ArrayList([]const u8) = .empty;
    try msgs.appendSlice(a, &.{ system, try user(a, "One picture.", &.{big}) });
    try t.expectEqual(@as(usize, 40), (try h.ask(a, try conv(a, msgs.items))).rows);
    // a second picture halves the share (64 / 2): the first is 32 rows now, another grid and so another image
    try msgs.appendSlice(a, &.{ ok, try user(a, "And another.", &.{try picture(a, 'b', 20)}) });
    const m = try conv(a, msgs.items);
    const turn = try h.ask(a, m);
    try t.expectEqual(@as(usize, 32 + 20), turn.rows);
    try expectFresh(a, .{ .image_tokens = 64 }, m, turn);
}

test "image reuse: switching conversations and concurrent clients resume each its own images" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{});
    defer h.deinit();
    const names = [_]u8{ 'a', 'b' };
    var firsts: [2][]const u8 = undefined;
    var seconds: [2][]const u8 = undefined;
    for (names, &firsts, &seconds) |n, *f1, *f2| {
        const u = try user(a, "Describe it.", &.{try picture(a, n, 12)});
        f1.* = try conv(a, &.{ system, u });
        f2.* = try conv(a, &.{ system, u, ok, try say(a, "More detail.") });
    }
    // both conversations' first turns at once, each equal to its solo reply
    var turns: [2]Turn = undefined;
    var errs: [2]?anyerror = .{ null, null };
    var arenas: [2]std.heap.ArenaAllocator = .{ .init(t.allocator), .init(t.allocator) };
    defer for (&arenas) |*x| x.deinit();
    const Run = struct {
        fn go(hh: *Harness, al: *std.heap.ArenaAllocator, m: []const u8, out: *Turn, err: *?anyerror) void {
            out.* = hh.ask(al.allocator(), m) catch |e| {
                err.* = e;
                return;
            };
        }
    };
    var threads: [2]std.Thread = undefined;
    for (&threads, &arenas, firsts, &turns, &errs) |*th, *al, m, *out, *err| th.* = try std.Thread.spawn(.{}, Run.go, .{ h, al, m, out, err });
    for (threads) |th| th.join();
    for (errs) |e| if (e) |x| return x;
    for (firsts, turns) |m, turn| try expectFresh(a, .{}, m, turn);
    // b's next turn, then a's: each resumes its own picture's state, none of the other's
    for ([_]usize{ 1, 0 }) |i| {
        const turn = try h.ask(a, seconds[i]);
        try t.expectEqual(@as(usize, 0), turn.rows);
        try expectFresh(a, .{}, seconds[i], turn);
    }
}

test "image reuse: a cancelled image pass and an evicting cache leave later turns equal to a fresh server" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const small: Options = .{ .budget = 64 + 4 * 120 }; // one kept state of these prompts, not two
    const h = try Harness.init(t.allocator, a, small);
    defer h.deinit();
    const ua = try user(a, "Describe it.", &.{try picture(a, 'a', 12)});
    const ub = try user(a, "Describe it.", &.{try picture(a, 'b', 12)});
    const a1 = try conv(a, &.{ system, ua });
    h.cancel_at = 1; // the pass is cancelled in the middle of the image prompt, as a client leaving does
    const cancelled = try h.ask(a, a1);
    try t.expectEqualStrings("cancelled", cancelled.finish);
    try t.expectEqual(@as(usize, 0), cancelled.tokens.len);
    const retried = try h.ask(a, a1);
    try t.expectEqual(@as(usize, 12), retried.rows);
    try expectFresh(a, small, a1, retried);
    try t.expectEqual(@as(usize, 1), h.store.entries.items.len);
    // b's conversation takes the cache's room: a's next turn runs its picture again, never b's state
    _ = try h.ask(a, try conv(a, &.{ system, ub }));
    try t.expectEqual(@as(usize, 1), h.store.entries.items.len);
    const a2 = try conv(a, &.{ system, ua, ok, try say(a, "More detail.") });
    const turn = try h.ask(a, a2);
    try t.expectEqual(@as(usize, 12), turn.rows);
    try expectFresh(a, small, a2, turn);
}

test "image reuse: the planted control without image keys resumes another picture's state and the check sees it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const h = try Harness.init(t.allocator, a, .{ .strip = true });
    defer h.deinit();
    _ = try h.ask(a, try conv(a, &.{ system, try user(a, "Look at this.", &.{try picture(a, 'a', 12)}) }));
    const other = try conv(a, &.{ system, try user(a, "Look at this.", &.{try picture(a, 'b', 12)}) });
    const wrong = try h.ask(a, other);
    try t.expectEqual(@as(usize, 0), wrong.rows); // resumed past the placeholders: picture a's rows stood in for b's
    try t.expect(!std.mem.eql(u32, (try fresh(a, .{}, other)).tokens, wrong.tokens));
}
