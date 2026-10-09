// The GLM head's draw rules against the lane core's host references, plus the GLM ABI's own regressions.

const std = @import("std");
const fwd = @import("draw_rule.zig");
const lanes = @import("lanes");
const gpu_full = lanes.gpu_full;
const gpu_rule = lanes.gpu_rule;
const Sampling = lanes.Sampling;

const a = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

test "the GLM rule mirrors the host reference's plan inputs at the same position" {
    const s: Sampling = .{ .seed = 0x1234_5678_9ABC_DEF0, .temperature = 0.7, .top_k = 40, .top_p = 0.9, .min_p = 0.05 };
    const rule = fwd.ruleOf(s, 4096, 154880);
    try expectEqual(@as(u32, @truncate(s.seed)), rule.seed_lo);
    try expectEqual(@as(u32, @truncate(s.seed >> 32)), rule.seed_hi);
    try expectEqual(@as(u32, 4096), rule.position);
    try expectEqual(rule.inv_t, @as(f32, 1.0) / @as(f32, 0.7));
    try expectEqual(rule.top_p, @as(f32, 0.9));
    try expectEqual(rule.min_log, @as(f32, @floatCast(s.minLog())));
    try expectEqual(@as(u32, 40), rule.top_k);
    try expectEqual(@as(u32, 154880), rule.vocab);
}

test "the GLM rule's fields are the Metal payload's order: seed, position, top_k, filters, vocab" {
    // The shader reads the rule as u32/f32 words behind a 16-byte header; the offsets are the ABI.
    try expectEqual(@as(usize, 36), @sizeOf(fwd.Rule));
    try expectEqual(@as(usize, 0), @offsetOf(fwd.Rule, "seed_lo"));
    try expectEqual(@as(usize, 4), @offsetOf(fwd.Rule, "seed_hi"));
    try expectEqual(@as(usize, 8), @offsetOf(fwd.Rule, "position"));
    try expectEqual(@as(usize, 12), @offsetOf(fwd.Rule, "top_k"));
    try expectEqual(@as(usize, 16), @offsetOf(fwd.Rule, "inv_t"));
    try expectEqual(@as(usize, 20), @offsetOf(fwd.Rule, "top_p"));
    try expectEqual(@as(usize, 24), @offsetOf(fwd.Rule, "near"));
    try expectEqual(@as(usize, 28), @offsetOf(fwd.Rule, "min_log"));
    try expectEqual(@as(usize, 32), @offsetOf(fwd.Rule, "vocab"));
    // A normal temperature-1 rule's words: the u32 fields first, then the float filters.
    const rule = fwd.ruleOf(Sampling{ .seed = 1, .temperature = 1.0, .top_k = 7, .top_p = 0.5, .min_p = 0.1 }, 3, 2048);
    const words: [9]u32 = std.mem.bytesToValue([9]u32, std.mem.asBytes(&rule));
    try expectEqual(@as(u32, 1), words[0]); // seed_lo
    try expectEqual(@as(u32, 0), words[1]); // seed_hi
    try expectEqual(@as(u32, 3), words[2]); // position
    try expectEqual(@as(u32, 7), words[3]); // top_k, not a float's bits
    try expectEqual(@as(u32, 2048), words[8]); // vocab, an integer the shader indexes with
}

test "a greedy marker's row argmaxes, and ruleOf's rows never carry one" {
    const rule = fwd.ruleOf(Sampling{ .seed = 2, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 9, 512);
    try expect(!rule.isGreedy());
    const greedy = fwd.Rule.greedy(512);
    try expect(greedy.isGreedy());
    try expectEqual(fwd.Rule.greedy_position, greedy.position);
    try expectEqual(@as(u32, 512), greedy.vocab); // the greedy argmax still reads the row's vocabulary
}

test "draws.at reads each row's rule and falls back to greedy past n" {
    var d: fwd.Draws = .{};
    d.rules[0] = fwd.ruleOf(Sampling{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 7, 100);
    d.rules[1] = fwd.ruleOf(Sampling{ .seed = 2, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 8, 100);
    d.n = 2;
    try expect((d.at(0) orelse return error.Test).position == 7);
    try expect((d.at(1) orelse return error.Test).position == 8);
    try expect(d.at(2) == null);
    try expect(d.at(1000) == null);
    try expect(d.sampled(0) and d.sampled(1) and !d.sampled(2));
}

test "the prompt's first-token rule sits in row 0 keyed at the prompt length (the head's one output row)" {
    // The review's defect: a rule in rules[15] read by a one-row head call at d.at(0) drew greedy.
    const P: u32 = 33;
    const d = fwd.promptDraws(Sampling{ .seed = 5, .temperature = 0.8, .top_k = 0, .top_p = 1.0 }, P, 2048);
    const first = d.at(0) orelse return error.Test; // the head's one-row call reads row 0
    try expectEqual(P, first.position); // keyed at the prompt length, the lane core's first() contract
    try expect(!first.isGreedy());
    try expectEqual(@as(usize, 1), d.n); // one rule: the prompt pass arms only the first reply token
}

test "a window's rules fill every row: sampled streams their rule, greedy streams the marker" {
    // The review's defect: out.n counted unsampled rows as rules, exposing undefined data to Draws.at.
    var sampled_stream: lanes.Stream = undefined;
    sampled_stream.sampling = Sampling{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    var greedy_stream: lanes.Stream = undefined;
    greedy_stream.sampling = null;
    const positions = [_]u64{ 11, 12, 13 };
    const be = lanes.backend;
    const windows = [_]be.Window{
        .{ .stream = &sampled_stream, .pending = 1, .held = 0, .tokens = &.{6}, .parents = null, .positions = positions[0..2] },
        .{ .stream = &greedy_stream, .pending = 2, .held = 0, .tokens = &.{}, .parents = null, .positions = positions[2..3] },
    };
    const d = fwd.windowDraws(&windows, 154880);
    try expectEqual(@as(usize, 3), d.n); // every row transmitted, sampled or not (1 + 1 token + 1)
    try expect(d.sampled(0)); // the sampled stream's pending row
    try expect(d.sampled(1)); // the sampled stream's token row
    try expect(!d.sampled(2)); // the greedy stream's row carries the marker, not undefined data
    try expectEqual(@as(u64, 11), d.at(0).?.position);
    try expectEqual(@as(u64, 12), d.at(1).?.position);
    try expect(d.at(2).?.isGreedy());
}

test "the configuration parser rejects the review's malformed values instead of sampling them" {
    // The review's fixtures: temperature=inf, top_p=2, min_p=1.5 were accepted; a trailing field too.
    try expectError(error.BadSampling, fwd.parseSampling("1,inf,0,1,0"));
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,2,0"));
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,1,1.5"));
    try expectEqual(@as(f64, 0), (try fwd.parseSampling("1,0,0,1,0")).temperature);
    try expectError(error.BadSampling, fwd.parseSampling("1,-1,0,1,0"));
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,0,0")); // top_p 0
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,1,1")); // min_p 1
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,1,0,9")); // a sixth field
    try expectError(error.BadSampling, fwd.parseSampling("x,1,0,1,0")); // a bad seed
    try expectError(error.BadSampling, fwd.parseSampling("1,nan,0,1,0"));
    try expectError(error.BadSampling, fwd.parseSampling("1,1,0,inf,0"));
    // A valid configuration survives, defaults and all.
    const full = try fwd.parseSampling("123,0.7,40,0.9,0.05");
    try expectEqual(@as(u64, 123), full.seed);
    try expectEqual(@as(f64, 0.7), full.temperature);
    try expectEqual(@as(u32, 40), full.top_k);
    try expectEqual(@as(f64, 0.9), full.top_p);
    try expectEqual(@as(f64, 0.05), full.min_p);
    const defaults = try fwd.parseSampling("7");
    try expectEqual(@as(f64, 1.0), defaults.temperature);
    try expectEqual(@as(u32, 0), defaults.top_k);
    try expectEqual(@as(f64, 1.0), defaults.top_p);
    try expectEqual(@as(f64, 0.0), defaults.min_p);
}

test "the host reference's plan keeps its contract: the GLM fixtures hold against gpu_full" {
    // Reference fixtures (the expected results the implementation must reach), kept from the review.
    var equal_logits: [2048]f32 = @splat(0.0);
    const p = try gpu_full.plan(a, &equal_logits, Sampling{ .seed = 1, .temperature = 1, .top_k = 0, .top_p = 0.75, .min_p = 0 });
    try expectEqual(@as(u32, 1535), p.cut.id); // the nucleus spans 1,536 of 2,048 equal tokens: id 1535
    var big: [4096]f32 = @splat(0.0);
    const p2 = try gpu_full.plan(a, &big, Sampling{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 0.5 });
    try expect(p2.cut.id >= 2047); // the nucleus spans 2,048 tokens of a 4,096-token row
}

test "host reference: top_k 1 with min_p keeps only argmax" {
    // Vocabulary 2048, logits [0, -0.1, ...], top_k=1, min_p=0.99: counting sees one token.
    var logits: [2048]f32 = @splat(-0.1);
    logits[0] = 0.0;
    const s: Sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1, .top_p = 1.0, .min_p = 0.99 };
    const p = try gpu_full.plan(a, &logits, s);
    for (0..16) |pos| {
        const id = gpu_full.race(&logits, p, s.seed, @intCast(pos), null);
        try expectEqual(@as(u32, 0), id);
    }
}

test "the review's equal-logit fixture draws across the whole vocabulary, not its first 1,024" {
    var logits: [2048]f32 = @splat(0.0);
    const s: Sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    const p = try gpu_full.plan(a, &logits, s);
    var tail: u32 = 0;
    for (0..200) |pos| {
        const id = gpu_full.race(&logits, p, s.seed, @intCast(pos), null);
        try expect(id < 2048);
        tail += @intFromBool(id >= 1024);
    }
    try expect(tail >= 50 and tail <= 150); // a uniform draw over 2,048 lands past 1,024 about half the time
}

test "host reference: large top_k boundary is not truncated" {
    var logits: [4096]f32 = @splat(0.0);
    const s: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 3000, .top_p = 1.0 };
    const p = try gpu_full.plan(a, &logits, s);
    try expectEqual(@as(u32, 2999), p.top.id);
    for (0..64) |pos| {
        const id = gpu_full.race(&logits, p, s.seed, @intCast(pos), null);
        try expect(id < 3000);
    }
}

test "the seeded rule covers both halves of the 64-bit seed" {
    const hi: Sampling = .{ .seed = 0x8000_0000_0000_0001, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    const rule = fwd.ruleOf(hi, 1, 2048);
    try expectEqual(@as(u32, 1), rule.seed_lo);
    try expectEqual(@as(u32, 0x8000_0000), rule.seed_hi);
    const lo: Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    const rule_lo = fwd.ruleOf(lo, 1, 2048);
    try expectEqual(@as(u32, 3), rule_lo.seed_lo);
    try expectEqual(@as(u32, 0), rule_lo.seed_hi);
}

test "standalone rulesFor advances one position a row and fills only the window's rows" {
    const s: Sampling = .{ .seed = 0xABCD, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    const d = fwd.rulesFor(s, 100, 154880, 4);
    try expectEqual(@as(usize, 4), d.n);
    for (0..4) |r| try expectEqual(@as(u32, 100 + @as(u32, @intCast(r))), d.at(r).?.position);
    try expect(d.at(4) == null); // past the window's rows: greedy, not a stale rule
}

test "production payload initializes trailing rows and preserves sampled rules" {
    const d = fwd.promptDraws(.{ .seed = 9, .temperature = 1, .top_k = 0, .top_p = 1 }, 4096, 2049);
    const p = fwd.Payload.init(&d, 4, 2049);
    try expectEqual(@as(u32, 4), p.header.n);
    try expectEqual(@as(usize, 16), @offsetOf(fwd.Payload, "rules"));
    try expectEqual(@as(usize, 592), @sizeOf(fwd.Payload));
    try expect(p.sampled());
    try expectEqual(@as(u32, 4096), p.rules[0].position);
    for (p.rules[1..]) |rule| try expect(rule.isGreedy());
    const greedy = fwd.Payload.init(null, 4, 2049);
    try expect(!greedy.sampled());
}

test "production prompt construction treats an explicit zero temperature as greedy" {
    const s = try fwd.parseSampling("7,0,0,1,0");
    const d = fwd.promptDraws(s, 17, 3);
    try expectEqual(@as(usize, 0), d.n);
    try expect(d.at(0) == null);
}
