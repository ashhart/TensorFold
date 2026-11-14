//! A DeepSeek-V4.1 request over the KV pool budget as its client sees it: queued behind a running one, or refused.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const lanes = @import("lanes");
const HfText = @import("hf_text.zig").HfText;
const f = @import("hf_eos_fixture.zig");
const tf = @import("control_tokens_fixture.zig");
const t = std.testing;

const alone = "the engine has no memory for this request's caches with nothing else running: lower --context or --parallel";

/// The reply an OpenAI route hands its connection: status and body, or a stream's events.
const Seen = struct {
    a: std.mem.Allocator,
    code: u16 = 0,
    body: []const u8 = "",
    fn out(s: *Seen) @import("openai.zig").Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, code: u16, v: json.Value) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.code = code;
        s.body = json.stringify(s.a, v, .{}) catch "";
    }
};

/// Ends the running request once the lane host has put the over-budget one back at its queue's head.
const Ender = struct {
    host: *api.LaneHost,
    requeued: u64 = 0,
    fn run(e: *Ender) void {
        for (0..2000) |_| {
            e.host.lock();
            e.requeued = e.host.load.requeued;
            e.host.unlock();
            if (e.requeued > 0) break;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        e.host.engine().cancel(running_id);
    }
};

const running_id: api.Id = 900; // apart from the ids the server gives its own requests

fn discard(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

/// `memory_lanes`: the lanes the pool has room for; past them prefill returns request_admission.zig's error.LaneMemory.
fn ask(memory_lanes: usize, running: bool) !struct { code: u16, body: []const u8, requeued: u64, arena: std.heap.ArenaAllocator } {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    var fx = try f.Fixture.init(a, t.io, null);
    defer fx.deinit();
    const tok = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
    try fx.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = tok });
    try fx.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = memory_lanes };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 2, .context_window = 32768 });
    try host.start();
    defer host.stop();
    const long: api.Request = .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_tokens = 100000 };
    var unused: u8 = 0;
    var ender: Ender = .{ .host = &host };
    var thread: ?std.Thread = null;
    if (running) {
        try host.engine().submit(running_id, &long, .{ .ctx = &unused, .event = discard });
        thread = try std.Thread.spawn(.{}, Ender.run, .{&ender});
    }
    const srv = try @import("server.zig").Server.init(a, t.io, host.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var seen: Seen = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 4));
    try body.put(a, "messages", try tf.messages(a, "H"));
    @import("openai.zig").run(srv, a, seen.out(), .{ .conn = &conn }, true, .{ .object = body });
    if (thread) |th| th.join();
    return .{ .code = seen.code, .body = seen.body, .requeued = ender.requeued, .arena = arena };
}

test "DeepSeek chat: a request over the free KV pool waits for the running one, then is served" {
    var got = try ask(1, true);
    defer got.arena.deinit();
    try t.expectEqual(@as(u64, 1), got.requeued);
    try t.expectEqual(@as(u16, 200), got.code);
    try t.expect(std.mem.indexOf(u8, got.body, "\"finish_reason\": \"length\"") != null);
}

test "DeepSeek chat: a request the KV pool cannot hold with nothing else running fails in words" {
    var got = try ask(0, false);
    defer got.arena.deinit();
    try t.expectEqual(@as(u16, 500), got.code);
    try t.expectEqualStrings("{\"error\": {\"message\": \"" ++ alone ++ "\"}}", got.body);
}
