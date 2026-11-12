//! The pair's refusals and its seam-sized chunk: the words, the guards, and the window the served pass may ask for.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const pair_mod = @import("qwen27_pair.zig");
const pair_flags = @import("qwen27_pair_flags.zig");
const served = @import("qwen27_served.zig");
const fabric = @import("fabric");
const Fingerprint = tf.pair_rows.Fingerprint;

test "qwen27 pair: rank 1 refuses a request by name before its backend moves round state" {
    try std.testing.expectEqualStrings("speed-up mode: this Mac runs rank 0's requests; send requests to rank 0", served.words(null, error.FollowsPeer).?);
    // `link` is the only `Metal` field the guard reads: the rest staying undefined proves no other step ran
    var pair: lb.Pair = undefined;
    pair.rank = 1;
    var l: lb.Link = undefined;
    l.pair = &pair;
    var m: lb.Metal = undefined;
    m.link = &l;
    try std.testing.expectError(error.FollowsPeer, pair_mod.refuseFollower(m.link));
    pair.rank = 0;
    try pair_mod.refuseFollower(m.link); // rank 0's own requests run on it
    m.link = null;
    try pair_mod.refuseFollower(m.link); // no pair: nothing to refuse for
}

test "qwen27 pair: a sampled or grammar request is refused by name at the head of prefill" {
    try std.testing.expectEqualStrings("speed-up mode: a pair answers greedy only; send the request with sampling and grammar off", served.words(null, error.PairGreedyOnly).?);
    // `link` is the only `Metal` field and the stream's two: the rest staying undefined proves no other step ran
    var pair: lb.Pair = undefined;
    pair.rank = 0;
    var l: lb.Link = undefined;
    l.pair = &pair;
    var m: lb.Metal = undefined;
    m.link = &l;
    var s: lanes.Stream = undefined;
    s.sampling = null;
    s.grammar = null;
    try pair_mod.refuseSampling(m.link, &s); // a greedy stream runs on a pair
    s.sampling = std.mem.zeroes(lanes.Sampling);
    try std.testing.expectError(error.PairGreedyOnly, pair_mod.refuseSampling(m.link, &s));
    s.sampling = null;
    var grammar: lanes.grammar.Constraint = undefined;
    s.grammar = &grammar;
    try std.testing.expectError(error.PairGreedyOnly, pair_mod.refuseSampling(m.link, &s));
    m.link = null;
    try pair_mod.refuseSampling(m.link, &s); // a single Mac samples and constrains freely
}

test "qwen27 pair: the served chunk from promptRows fits the window" {
    // the served pass asks for the generation's rows (`model.promptRows(generation)`); a pair caps them at the window
    const chunk = lb.pairChunk(q.model.promptRows(q.config.Config{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 124160, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 }, 0));
    try std.testing.expect(chunk <= lb.batch_rows);
    try std.testing.expect(lb.seamBytes(chunk) <= lb.pairBytes());
    // the uncapped pass would stage a wider seam than the window, so the cap is what keeps the exchange legal
    try std.testing.expect(lb.seamBytes(512) > lb.pairBytes());
}

test "qwen27 pair: a pair never resumes a slot in place, and one Mac keeps the switch it read" {
    // `attach` passes `Metal.init`'s switch through `slotResume`; a refusal that put it back on would turn this red
    var pair: lb.Pair = undefined;
    pair.rank = 0;
    try std.testing.expect(!lb.slotResume(true, &pair)); // rank 0, which would resume
    pair.rank = 1;
    try std.testing.expect(!lb.slotResume(true, &pair)); // rank 1, which resets
    // a single Mac keeps main's in-place resume where slot 0 stands (#1197), and `TF_QWEN27_SLOT_RESUME=0` stays off
    try std.testing.expect(lb.slotResume(true, null));
    try std.testing.expect(!lb.slotResume(false, null));
}

test "qwen27 pair: --vision is refused on a pair, by name" {
    var why: []const u8 = "";
    try std.testing.expectError(error.PairNoVision, pair_flags.refuse(false, false, false, true, &why));
    try std.testing.expect(std.mem.startsWith(u8, why, "--vision does not run on a pair"));
    try pair_flags.refuse(false, false, false, false, &why); // a single Mac sets none of the flags this refuses
}

test "qwen27 pair: --drafter runs on a pair: rank 0 drafts, rank 1 replays the tree rounds" {
    // `open` reads `.drafter` through this helper: the tree protocol replays rank 0's draft rounds
    const pair = served.Options{
        .context = 4096,
        .streams = 1,
        .pair = "rank0.json",
        .drafter = "z-lab/Qwen3.8-27B-DFlash2",
    };
    var why: []const u8 = "";
    try pair_flags.refuseOptions(true, pair, false, &why);
    // a drafter the caller already bound reaches `attach` with no flag in the options, and is accepted too
    var plain = pair;
    plain.drafter = null;
    try pair_flags.refuseOptions(true, plain, true, &why);
    // the other pair refusals still stand
    var cache = pair;
    cache.drafter = null;
    cache.cache_gib = 8;
    try std.testing.expectError(error.PairNoCache, pair_flags.refuseOptions(true, cache, false, &why));
    // one Mac keeps every flag: with no pair there is nothing to refuse
    try pair_flags.refuseOptions(false, pair, false, &why);
}

test "qwen27 pair: the fingerprint names the fact two Macs differ on" {
    const same: tf.pair_rows.Fingerprint = .{ .version = 7, .chip = 4, .checkpoint = 0xA11CE, .format = 64 };
    try std.testing.expectEqual(@as(?[]const u8, null), try tf.pair_rows.fingerprintProblem(std.testing.allocator, same, same)); // equal facts refuse nothing
    // each fact by name: the message says which one differed and both sides' words
    const cases = [_]tf.pair_rows.Fingerprint{
        .{ .version = 8, .chip = 4, .checkpoint = 0xA11CE, .format = 64 }, // another build
        .{ .version = 7, .chip = 5, .checkpoint = 0xA11CE, .format = 64 }, // an M5 beside an M4
        .{ .version = 7, .chip = 4, .checkpoint = 0xB0B, .format = 64 }, // another checkpoint (quantization, shapes)
        .{ .version = 7, .chip = 4, .checkpoint = 0xA11CE, .format = 128 }, // another weight format
    };
    const names = [_][]const u8{ "engine version", "GPU family", "checkpoint", "weight format" };
    for (cases, names) |peer, name| {
        const why = (try tf.pair_rows.fingerprintProblem(std.testing.allocator, same, peer)).?;
        defer std.testing.allocator.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, name) != null);
        try std.testing.expect(std.mem.indexOf(u8, why, "match neither Mac alone") != null);
    }
    // a uniform tag 0 travels the wire as 1 (ready cannot signal a 0); both sides normalize
    const uniform: tf.pair_rows.Fingerprint = .{ .version = 7, .chip = 4, .checkpoint = 0xA11CE, .format = 0 };
    const decoded: tf.pair_rows.Fingerprint = .{ .version = 7, .chip = 4, .checkpoint = 0xA11CE, .format = 1 };
    try std.testing.expectEqual(@as(?[]const u8, null), try tf.pair_rows.fingerprintProblem(std.testing.allocator, uniform, decoded));
    try std.testing.expectEqual(@as(?[]const u8, null), try tf.pair_rows.fingerprintProblem(std.testing.allocator, decoded, uniform));
}

/// One rank of a pair whose link is the fake fabric: the fields `ready` and `next` read are set, the rest unset.
fn fakeRank(c: *fabric.fake.Cluster, rank: u32, layout: @FieldType(lb.Pair, "l")) lb.Pair {
    var p: lb.Pair = undefined;
    p.rank = rank;
    p.peer = 1 - rank;
    p.l = layout;
    p.rd = c.endpoint(rank);
    p.win = c.windows[rank];
    p.stop = .init(false);
    p.broken = .init(false);
    p.cmd = 0;
    return p;
}

/// A rank's `ready` and what it answered: no error, or the refusal and its words.
const Ready = struct {
    p: lb.Pair,
    fp: Fingerprint,
    why: []const u8 = "",
    err: ?anyerror = null,

    fn run(r: *Ready) void {
        r.p.ready(r.fp, std.testing.allocator, &r.why) catch |err| {
            r.err = err;
        };
    }
};

/// Both ranks' `ready` at once over one fake link, rank 1 on its own thread as on its own Mac.
fn readyBoth(zero: Fingerprint, one: Fingerprint) ![2]Ready {
    const layout: @FieldType(lb.Pair, "l") = .{ .slot = 16384 };
    const c = try fabric.fake.Cluster.init(std.testing.allocator, 2, layout.window(), .immediate, null);
    defer c.deinit();
    var ranks = [2]Ready{ .{ .p = fakeRank(c, 0, layout), .fp = zero }, .{ .p = fakeRank(c, 1, layout), .fp = one } };
    const t = try std.Thread.spawn(.{}, Ready.run, .{&ranks[1]});
    ranks[0].run();
    t.join();
    return ranks;
}

test "qwen27 pair: ready refuses another checkpoint or GPU family by name, read through the link's words" {
    const m4: Fingerprint = .{ .version = 7, .chip = pair_mod.chipGeneration("Apple M4 Max"), .checkpoint = 0xA11CE, .format = 64 };
    for (try readyBoth(m4, m4)) |r| try std.testing.expectEqual(@as(?anyerror, null), r.err); // alike: both come up
    var m5 = m4;
    m5.chip = pair_mod.chipGeneration("Apple M5 Ultra");
    var other = m4;
    other.checkpoint = 0xB0B;
    for ([_]Fingerprint{ m5, other }, [_][]const u8{ "GPU family", "checkpoint" }) |peer, name| {
        // each rank compares its own facts with the words the other signalled, so both refuse, naming the fact
        const ranks = try readyBoth(m4, peer);
        for (ranks) |r| {
            defer std.testing.allocator.free(r.why);
            try std.testing.expectEqual(@as(?anyerror, error.PairFingerprintMismatch), r.err);
            try std.testing.expect(std.mem.indexOf(u8, r.why, name) != null);
        }
    }
}

test "qwen27 pair: the GPU family is the generation the Metal device names" {
    try std.testing.expectEqual(@as(u64, 4), pair_mod.chipGeneration("Apple M4 Max"));
    try std.testing.expectEqual(@as(u64, 3), pair_mod.chipGeneration("Apple M3 Ultra"));
    try std.testing.expectEqual(@as(u64, 5), pair_mod.chipGeneration("Apple M5"));
    try std.testing.expectEqual(@as(u64, 0), pair_mod.chipGeneration("Apple"));
    try std.testing.expectEqual(@as(u64, 0), pair_mod.chipGeneration("AMD Radeon Pro"));
}
