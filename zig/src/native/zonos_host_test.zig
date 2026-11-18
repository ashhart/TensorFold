//! The ZONOS3 host's sizes: what a load is admitted for.
const std = @import("std");
const z3 = @import("zonos3");
const host = @import("zonos_host.zig");

const cfg = z3.config;
const need = host.need;
const budget = host.budget;
const laneBytes = host.laneBytes;
const twinBytes = host.sizes.twinBytes;

test "need: the weights, then state that grows with the context and with the frame budget" {
    const t = std.testing;
    const base = need(4096, 2000, .keep2_plain);
    try t.expectEqual(@as(u64, 3_125_611_008), base);
    // the full-size decoder as the default: its head in place of KEEP-2's, and more layers' caches
    try t.expect(need(4096, 2000, .full_mr) > base + 3_400_000_000);
    try t.expect(need(8192, 2000, .keep2_plain) > base and need(4096, 1000, .keep2_plain) < base);
    // two attention layers keep a key row and a value row a position, and a masked head's scratch one of each
    try t.expectEqual(@as(u64, (2 * 2 + 2) * 4096 * cfg.kv_width * 4), need(8192, 2000, .keep2_plain) - base);
}

test "budget: the release's frame budget where the context holds it, the rest for the prompt" {
    const t = std.testing;
    const d = try budget(4096, null);
    try t.expectEqual(@as(u32, 2000), d.frames);
    try t.expectEqual(@as(u32, 2096), d.prompt);
    try t.expectEqual(@as(u32, 512), (try budget(1025, null)).frames);
    try t.expectError(error.BadZonosContext, budget(1024, null));
    try t.expectError(error.BadZonosContext, budget(32769, null));
    try t.expectError(error.BadZonosFrames, budget(4096, 4096));
    try t.expectError(error.BadZonosFrames, budget(4096, 5));
}

test "laneBytes: each request at once past the first adds a backbone lane and a decoder lane, no weights" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 495_527_936), laneBytes(4096, 2000, .keep2_plain));
    try t.expectEqual(@as(u64, 10_913_792), laneBytes(4096, 2000, .keep2_plain) - z3.Codec.laneBytes(.{ .max_frames = 2000 }));
    try t.expect(laneBytes(4096, 1000, .keep2_plain) < laneBytes(4096, 2000, .keep2_plain));
}

test "twinBytes: a guided request's second stream is the caches and recurrent state again, no weights" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 10_913_792), twinBytes(.{ .max_ctx = 4096 }));
    try t.expectEqual(@as(u64, 2 * 2 * 4096 * cfg.kv_width * 4), twinBytes(.{ .max_ctx = 8192 }) - twinBytes(.{ .max_ctx = 4096 }));
}
