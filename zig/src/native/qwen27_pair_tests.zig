//! The pair's refusals and its seam-sized chunk: the words, the guards, and the window the served pass may ask for.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const pair_mod = @import("qwen27_pair.zig");
const pair_flags = @import("qwen27_pair_flags.zig");
const served = @import("qwen27_served.zig");

test "rank 1 refuses a request by name before its backend moves round state" {
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

test "a pair refuses a sampled or grammar request by name at the head of prefill" {
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

test "the served pair's chunk times the widest seam fits the window" {
    // the served pass asks for the generation's rows (`model.promptRows(generation)`); a pair caps them at the window
    const chunk = lb.pairChunk(q.model.promptRows(0));
    try std.testing.expect(chunk <= lb.batch_rows);
    try std.testing.expect(lb.seamBytes(chunk) <= lb.pairBytes());
    // the uncapped pass would stage a wider seam than the window, so the cap is what keeps the exchange legal
    try std.testing.expect(lb.seamBytes(512) > lb.pairBytes());
}

test "a pair never resumes a slot in place, and one Mac keeps the switch it read" {
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

test "--vision is refused on a pair, by name" {
    var why: []const u8 = "";
    try std.testing.expectError(error.PairNoVision, pair_flags.refuse(false, false, false, true, &why));
    try std.testing.expect(std.mem.startsWith(u8, why, "--vision does not run on a pair"));
    try pair_flags.refuse(false, false, false, false, &why); // a single Mac sets none of the flags this refuses
}

test "--drafter is refused at the front door, by name" {
    // `open` reads `.drafter` through this helper: a refusal that never fires here turns this test red
    const pair = served.Options{
        .context = 4096,
        .streams = 1,
        .pair = "rank0.json",
        .drafter = "z-lab/Qwen3.8-27B-DFlash2",
    };
    var why: []const u8 = "";
    try std.testing.expectError(error.PairNoDrafter, served.refusePairFlags(true, pair, false, &why));
    try std.testing.expect(std.mem.startsWith(u8, why, "--drafter does not run on a pair"));
    // a drafter the caller already bound reaches `attach` with no flag in the options, and is refused too
    var plain = pair;
    plain.drafter = null;
    try std.testing.expectError(error.PairNoDrafter, served.refusePairFlags(true, plain, true, &why));
    // one Mac keeps its drafter: with no pair there is nothing to refuse
    try served.refusePairFlags(false, pair, false, &why);
}
