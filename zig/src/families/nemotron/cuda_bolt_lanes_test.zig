//! Production admission facts with a fake target: lane-core scheduling, copy proposals and acceptance counters.
const std = @import("std");
const lanes = @import("lanes");
const Cuda = @import("cuda_lanes.zig").Cuda;
const Engine = @import("cuda_engine.zig").Engine;
const a = std.testing.allocator;
const Result = struct { tokens: [4][24]u32, proposed: u64, verified: u64, accepted: u64, rounds: u64, stream_rounds: u64 };

fn run(drafts: bool, global: bool) !Result {
    var e: Engine = undefined;
    e.c.latent = 128;
    var adapter: Cuda = undefined;
    adapter.e = &e;
    adapter.head = null;
    adapter.copy_drafts = global;
    adapter.cost_count = 0;
    adapter.mtp_ms = 0;
    var cfg = try lanes.Config.init(a, adapter.facts(), 16, 15);
    defer cfg.deinit(a);
    try std.testing.expect(cfg.family_streams);
    try std.testing.expect(!cfg.family_mtp);
    const prompt = [_]u32{ 11, 12, 13, 11, 12, 13, 11, 12, 13, 11, 12, 13 };
    var target: lanes.fake.Fake = .{ .gpa = a, .cycle_after = 0, .probe_prompt = prompt.len };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    var streams: [4]lanes.Stream = undefined;
    var proposers: [4]lanes.SuffixLookup = undefined;
    var live: usize = 0;
    defer {
        core.deinit();
        for (streams[0..live], proposers[0..live]) |*s, *p| {
            s.deinit(a);
            p.deinit();
        }
    }
    for (&streams, &proposers) |*s, *p| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 6 });
        s.* = lanes.Stream.init(a, .{ .id = "bolt", .prompt = &prompt, .max_new = 24, .drafts = drafts, .proposer = p.proposer(), .loop_guard = false }) catch |err| {
            p.deinit();
            return err;
        };
        live += 1;
        try core.addStream(s);
    }
    while (core.activeCount() > 0) try core.step();
    var result: Result = .{ .tokens = undefined, .proposed = 0, .verified = 0, .accepted = 0, .rounds = target.rounds, .stream_rounds = 0 };
    for (streams, proposers, &result.tokens) |s, p, *out| {
        try std.testing.expectEqual(@as(usize, 24), s.emitted().len);
        @memcpy(out, s.emitted());
        result.proposed += @intCast(p.proposed_tokens);
        result.stream_rounds += s.rounds;
        result.verified += s.drafted;
        result.accepted += s.accepted;
    }
    return result;
}

test "four latent streams share rounds with real copy proposals and plain replies match" {
    const drafted = try run(true, true);
    const plain = try run(false, true);
    const disabled = try run(true, false);
    try std.testing.expect(drafted.proposed > 0);
    try std.testing.expect(drafted.verified > 0);
    try std.testing.expect(drafted.accepted <= drafted.verified);
    std.debug.print("fake target: proposed={d} verified={d} accepted={d}\n", .{ drafted.proposed, drafted.verified, drafted.accepted });
    try std.testing.expectEqual(@as(u64, 0), plain.proposed);
    try std.testing.expectEqual(@as(u64, 0), disabled.proposed);
    try std.testing.expect(drafted.stream_rounds > drafted.rounds);
    try std.testing.expect(disabled.stream_rounds > disabled.rounds);
    try std.testing.expectEqual(plain.tokens, drafted.tokens);
    try std.testing.expectEqual(plain.tokens, disabled.tokens);
}
