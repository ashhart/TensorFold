//! A long prompt beside a live stream: split at the cap, the same tokens as one pass, the gap between rounds one chunk.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const ends = @import("lane_host_end.zig");
const t = std.testing;

const long_rows = 12288;
const live_tokens = 64;
const long_tokens = 8;

/// Each request's tokens as the host delivered them.
const Sink = struct {
    live: [live_tokens]u32 = undefined,
    long: [long_tokens]u32 = undefined,
    live_n: usize = 0,
    long_n: usize = 0,

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        if (e.* != .tokens) return;
        for (e.tokens) |token| {
            if (id == 1) {
                s.live[s.live_n] = token;
                s.live_n += 1;
            } else {
                s.long[s.long_n] = token;
                s.long_n += 1;
            }
        }
    }

    fn done(s: *const Sink) bool {
        return s.live_n == live_tokens and s.long_n == long_tokens;
    }
};

const Run = struct { sink: Sink = .{}, widest: u64 = 0, steps: usize = 0, rounds_during: usize = 0 };

/// The host's loop by hand: a live stream decoding, then a 12,288-row prompt; `share` is --decode-share.
fn twoStreams(share: ?f64, out: *Run) !void {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    core.backend.chunk_invariant = true;
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 4, .decode_share = share });
    defer {
        ends.closeAll(&host);
        host.stop();
    }
    const sink: api.Sink = .{ .ctx = &out.sink, .event = Sink.event };
    const live: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = live_tokens, .drafts = false };
    try host.engine().submit(1, &live, sink);
    try t.expect(host.admitOne());
    while (host.filling.items.len > 0) fill.run(&host);
    var prompt: [long_rows]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i % 89 + 2);
    const long: api.Request = .{ .prompt = &prompt, .max_tokens = long_tokens, .drafts = false };
    try host.engine().submit(2, &long, sink);
    try t.expect(host.admitOne());
    var visits: usize = 0;
    while (!out.sink.done()) : (visits += 1) {
        if (visits == 4096) return error.RunNeverEnded;
        host.share.debt_ns = 0; // the round below paid the last step's debt
        const job = if (host.filling.items.len > 0) host.filling.items[0] else null;
        const before = if (job) |j| j.stream.fill_at.load(.monotonic) else 0;
        const filling = job != null;
        fill.run(&host);
        if (filling) {
            const open = host.filling.items.len > 0;
            const after = if (open) host.filling.items[0].stream.fill_at.load(.monotonic) else long_rows;
            out.widest = @max(out.widest, after - before);
            out.steps += 1;
        }
        try core.sweep();
        ends.deliverPending(&host);
        if (core.activeCount() == 0) continue;
        if (filling and host.filling.items.len > 0) out.rounds_during += 1;
        try core.step();
    }
}

test "a long prompt beside a live stream splits at the cap, with the tokens of the unsplit pass" {
    var whole: Run = .{};
    try twoStreams(null, &whole);
    var split: Run = .{};
    try twoStreams(0.5, &split);
    try t.expectEqual(@as(u64, long_rows), whole.widest); // no flag: today's one pass
    try t.expectEqual(@as(usize, 1), whole.steps);
    try t.expectEqual(@as(u32, 512), fill.decode_cap_rows); // the cap these cases pin, written out so a change shows here
    try t.expectEqual(@as(u64, 512), split.widest);
    try t.expectEqual(@as(usize, long_rows / 512), split.steps);
    try t.expectEqualSlices(u32, &whole.sink.live, &split.sink.live);
    try t.expectEqualSlices(u32, &whole.sink.long, &split.sink.long);
}

test "the live stream gets a round after every chunk but the last: its gap is one chunk" {
    var split: Run = .{};
    try twoStreams(0.5, &split);
    try t.expectEqual(split.steps - 1, split.rounds_during);
    try t.expect(split.widest <= 512);
}
