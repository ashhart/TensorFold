//! Slot 0 after a hand-off turn (#1594 step 2b): the real lane host and qwen27_slots on a fake backend, no device.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const slots = @import("qwen27_slots.zig");
const remote = @import("qwen27_remote.zig");
const Metal = @import("qwen27_lanes.zig").Metal;
const Remote = api.LaneHost.Remote;
const Buffer = @import("metal").Buffer;
const t = std.testing;
const gpa = t.allocator;

/// A GB10 that fills a prompt's first `at` tokens: its copy is the fake backend's kept form, the prefix's tokens.
const Peer = struct {
    at: u32,
    fills: u32 = 0,
    running: std.atomic.Value(u32) = .init(0),
    kept: std.ArrayList(u32) = .empty,

    fn wants(ptr: *anyopaque, _: u64, cached: u64) bool {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        return p.at > cached;
    }
    fn fill(ptr: *anyopaque, prompt: []const u32, _: u64) anyerror!?Remote.Filled {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        p.fills += 1;
        p.kept.clearRetainingCapacity();
        try p.kept.appendSlice(gpa, prompt[0..p.at]);
        return .{ .saved = &p.kept, .at = p.at };
    }
    fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
};

/// What a reply heard: its finished stats, the server's prefilled and remote counts.
const Heard = struct {
    reason: ?api.Reason = null,
    remote_prefix: u32 = 0,
    pulled_prefix: u32 = 0,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Heard = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .finished => |f| {
                x.* = .{ .reason = f.reason, .remote_prefix = f.stats.remote_prefix, .pulled_prefix = f.stats.pulled_prefix };
                x.done.store(true, .release);
            },
            else => {},
        }
    }
    /// The same, without the atomic, for comparing.
    fn plain(x: Heard) Heard {
        return .{ .reason = x.reason, .remote_prefix = x.remote_prefix, .pulled_prefix = x.pulled_prefix };
    }
};

/// A Metal with no device: slot 0's record, rewind buffer flag and offsets, and nothing a GPU pass would touch.
const Mac = struct {
    runner: tf.qwen27.decode_round.Runner = undefined,
    offsets: [1]u32 = .{0},
    back: Metal = undefined,
    fake_prefill: ?*const fn (ptr: *anyopaque, s: *lanes.Stream) anyerror!void = null,
    resumed: u32 = 0, // where the last pass's opening resumed slot 0 (0: it ran as a GPU pass would, not here)
    released: std.atomic.Value(u32) = .init(0),
    context: std.ArrayList(u32) = .empty, // the last stream's prompt and reply

    fn deinit(m: *Mac) void {
        m.back.standing.deinit(gpa);
        m.context.deinit(gpa);
    }

    fn init(m: *Mac) void {
        m.runner.offsets = &m.offsets;
        m.runner.slots = 1;
        m.runner.ready = 0; // a pass that does not resume in slot 0 would reset it: RoundNotReady, never the GPU
        m.runner.capacity = 4096;
        m.runner.active = null;
        m.runner.failed = false;
        m.runner.rope = @splat(.{});
        m.runner.reading = @splat(null);
        const b = &m.back;
        b.gpa = gpa;
        b.runner = &m.runner;
        b.draft = null;
        b.chunk = 16;
        b.window = 16;
        b.owners = @splat(null);
        b.checkpoint_hook = null;
        b.vision = null;
        b.image_fill = null;
        b.standing = .{};
        b.rewind = null;
        b.slot_resume = true;
    }

    /// The pass: slot 0's own opening where it resumes there (no copy, no GPU), else as a pass from its restore.
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const m = current.?;
        m.resumed = 0;
        if (slots.resumeAt(&m.back, s.prompt(), s.decode_spans) > 0) {
            m.resumed = @intCast((try slots.opening(&m.back, s, 0)).from);
        } else {
            const at: usize = if (s.reuse.saved != null) s.reuse.at else 0;
            const point = slots.rewindPoint(&m.back, s, 0, at, .{});
            m.back.standing.deinit(gpa); // the pass takes slot 0 from 0 or the restore (takeStanding's .none)
            m.back.standing.rewind_at = point; // saveRewind's blit at the point, which needs a GPU, landed
        }
        try m.fake_prefill.?(ptr, s);
        if (m.resumed > 0) s.cached = m.resumed; // the fake fills from 0; the Metal stood there already
    }

    /// The stream leaves slot 0 one short of its reply's end, as unbind does.
    fn release(ctx: *anyopaque, s: *lanes.Stream) void {
        const m: *Mac = @ptrCast(@alignCast(ctx));
        m.offsets[0] = @intCast(s.context.items.len - 1);
        slots.leave(&m.back, s);
        m.context.clearRetainingCapacity();
        m.context.appendSlice(gpa, s.context.items) catch unreachable;
        _ = m.released.fetchAdd(1, .acq_rel);
    }
};

var current: ?*Mac = null;

/// One request on a fresh lane host over `m` (the same Mac across turns), TF_PREFILL_KEEP `keep`, the GB10 `peer`.
fn turn(m: *Mac, peer: *Peer, request: *const api.Request, keep: bool) !Heard {
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var fake: lanes.fake.Fake = .{ .gpa = gpa, .release_hook = Mac.release, .release_hook_ctx = m };
    defer fake.deinit();
    var backend = fake.backend();
    var vtable = backend.vtable.*;
    m.fake_prefill = vtable.prefill;
    vtable.prefill = Mac.prefill;
    backend.vtable = &vtable;
    current = m;
    defer current = null;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 1 });
    host.decoded_rows = true; // as the 27B's served host
    host.remote = .{ .ptr = peer, .wants = Peer.wants, .fill = Peer.fill, .drop = Peer.drop, .running = &peer.running };
    remote.keepRule(.{ .host = &host, .back = &m.back }, keep); // as attach, on --prefill-from
    host.resume_peek = .{ .ctx = &m.back, .at = slots.resumeAt };
    const left = m.released.load(.acquire);
    try host.start();
    defer host.stop();
    var heard: Heard = .{};
    try host.engine().submit(1, request, .{ .ctx = &heard, .event = Heard.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!heard.done.load(.acquire) or m.released.load(.acquire) == left or peer.running.load(.acquire) > 0) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.HandoffFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    return heard;
}

const p1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 20 }; // a history of 8 tokens, then 20 opens the reply

test "qwen27 hand-off: the follow-up resumes slot 0 where the pulled turn left it, pulls nothing and reports it remote" {
    var m: Mac = .{};
    m.init();
    defer m.deinit();
    m.back.rewind = @as(Buffer, undefined); // a rewind buffer: only its presence is read here
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(gpa);
    const first = try turn(&m, &peer, &.{ .prompt = &p1, .max_tokens = 3, .history_len = 8 }, true);
    try t.expectEqual(@as(u32, 1), peer.fills);
    try t.expect(first.remote_prefix == 4 and first.pulled_prefix == 4); // the GB10 filled 4 tokens just now
    const st = &m.back.standing;
    const stood = .{ st.keep_remote, st.remote, st.rewind_at, st.tokens.len }; // what turn 1 left in slot 0
    const back = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 20, 30, 31 }; // the reply rendered without its reasoning
    const back_at = slots.resumeAt(&m.back, &back, &.{.{ 9, 10 }});
    var follow: [p1.len + 5]u32 = undefined; // the reply sent back as generated (decoded rows 9-11), then a new turn
    @memcpy(follow[0 .. p1.len + 3], m.context.items);
    follow[p1.len + 3] = 40;
    follow[p1.len + 4] = 20;
    const next = try turn(&m, &peer, &.{ .prompt = &follow, .max_tokens = 3, .history_len = p1.len + 4, .decode_spans = &.{.{ p1.len, p1.len + 3 }} }, true);
    try t.expectEqual(@as(u32, 1), peer.fills); // nothing pulled again
    try t.expectEqual(@as(u32, p1.len + 2), m.resumed); // in place where turn 1 left slot 0, not at 0
    try t.expectEqual(Heard{ .reason = .length, .remote_prefix = p1.len + 2, .pulled_prefix = 0 }, next.plain()); // kept remote
    try t.expect(st.remote and st.rewind_at == 8 and st.tokens.len == follow.len + 2); // and it stands remote again
    try t.expectEqual(.{ true, true, @as(u32, 8), @as(usize, p1.len + 2) }, stood); // a remote record one short of the end
    try t.expectEqual(@as(u32, 8), back_at); // a turn without the reply's reasoning goes back to the history end
}

test "qwen27 hand-off: with TF_PREFILL_KEEP=0 the pulled turn leaves nothing in slot 0, so the follow-up pulls again" {
    var m: Mac = .{};
    m.init();
    defer m.deinit();
    m.back.rewind = @as(Buffer, undefined); // a rewind buffer: only its presence is read here
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(gpa);
    _ = try turn(&m, &peer, &.{ .prompt = &p1, .max_tokens = 3, .history_len = 8 }, false);
    const st = &m.back.standing;
    try t.expect(!st.keep_remote and st.tokens.len == 0 and st.rewind_at == 0);
    var follow: [p1.len + 5]u32 = undefined;
    @memcpy(follow[0 .. p1.len + 3], m.context.items);
    follow[p1.len + 3] = 40;
    follow[p1.len + 4] = 20;
    const next = try turn(&m, &peer, &.{ .prompt = &follow, .max_tokens = 3, .history_len = p1.len + 4, .decode_spans = &.{.{ p1.len, p1.len + 3 }} }, false);
    try t.expectEqual(@as(u32, 2), peer.fills);
    try t.expectEqual(@as(u32, 0), m.resumed);
    try t.expect(next.pulled_prefix == 4 and st.tokens.len == 0);
}

/// Mark hooks that must never run: opening only plans where its pass stops.
fn noMark(_: *anyopaque, _: *lanes.Stream, _: u32) void {
    unreachable;
}

test "qwen27 hand-off: a pass resuming slot 0's remote rows makes no unplanned checkpoint, and on its own rows it does" {
    var hook_ctx: u8 = 0;
    const hook: lanes.stream.MarkHook = .{ .ptr = &hook_ctx, .at = noMark };
    var ids: [232]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i + 1);
    for ([_]bool{ true, false }) |on_remote| {
        var m: Mac = .{};
        m.init();
        defer m.deinit();
        m.back.standing.keep_remote = true;
        m.back.checkpoint_hook = hook;
        var s0 = try lanes.Stream.init(gpa, .{ .id = "turn", .prompt = ids[0..32], .max_new = 1, .reuse = .{ .remote = on_remote } });
        defer s0.deinit(gpa);
        s0.fill_at.store(32, .monotonic); // a pass that yielded at a landed chunk's end
        m.offsets[0] = 32;
        slots.leave(&m.back, &s0);
        try t.expect(m.back.standing.tokens.len == 32 and m.back.standing.remote == on_remote);
        // its follow-up: a pull slot 0 overtook (lane_remote.ownFirst), a mark hook, a history end the cache keeps not
        var s = try lanes.Stream.init(gpa, .{ .id = "next", .prompt = &ids, .max_new = 2, .history_len = 200, .reuse = .{ .hook = hook, .pulled = true } });
        defer s.deinit(gpa);
        const f = try slots.opening(&m.back, &s, 0);
        try t.expect(f.from == 32 and s.cached == 32);
        try t.expect(s.reuse.remote == on_remote and !s.reuse.pulled); // its rows' origin, and no pull just now
        try t.expectEqual(@as(u32, if (on_remote) 0 else 48), f.checkpoint); // 48: the first chunk's end, on own rows
    }
}

test "qwen27 hand-off: attach's TF_PREFILL_KEEP reaches slot 0 as it reaches the lane host" {
    var m: Mac = .{};
    m.init();
    var host: struct { keep_remote: bool = false } = .{};
    remote.keepRule(.{ .host = &host, .back = &m.back }, true);
    try t.expect(host.keep_remote and m.back.standing.keep_remote);
    remote.keepRule(.{ .host = &host, .back = &m.back }, false);
    try t.expect(!host.keep_remote and !m.back.standing.keep_remote);
}
