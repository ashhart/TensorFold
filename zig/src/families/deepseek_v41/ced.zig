//! CED bounded replay prefill (`TF_DSV41_PREFILL=replay`): the replay rules and the stash bookkeeping (host only).

const std = @import("std");
const Config = @import("config.zig").Config;
const ft = @import("forward_types.zig");

pub const replay_rows = ft.replay_rows;
pub const keep = ft.keep;
pub const ring = ft.ring;
/// snapshot and piece alignment
pub const grid: u64 = 16;
/// a prompt snapshot needs this many tokens
pub const session_min: u64 = 64;
/// session tag prefill field: the fast prefill path, prompt snapshots only; exact-tag (0) turn entries never resume
pub const tag_fast: u32 = 1 << 8;
/// with the CED field (replay mode index << 4): replay entries never resume full ones
pub const tag: u32 = tag_fast | 1 << 4;

pub const Error = error{ NoDecoder, EngramInDecoder, StashGap };

/// The first ratio-1 layer: must be a kv source, and no decoder layer may be an Engram layer.
pub fn decoderStart(cfg: *const Config) Error!u32 {
    var d: u32 = cfg.layers;
    for (0..cfg.layers) |i| if (cfg.compressRatio(@intCast(i)) == 1) {
        d = @intCast(i);
        break;
    };
    if (d >= cfg.layers or !cfg.isKvSource(d)) return error.NoDecoder;
    for (cfg.engram_layers.items()) |L| if (L >= d) return error.EngramInDecoder;
    return d;
}

/// The decoder replay of a prompt whose prefilled part ends at `end` (= n - 1): rows [r0, end), m = end - r0.
pub const Tail = struct { r0: u64, m: u32 };

pub fn tailOf(end: u64) Tail {
    const m: u32 = @intCast(@min(end, keep));
    return .{ .r0 = end - m, .m = m };
}

/// The last grid point strictly before a prompt's end.
pub fn snapshotPoint(n: u64) u64 {
    return if (n > 0) (n - 1) / grid * grid else 0;
}

/// Where a prompt of `n` tokens resumed at `cached` is snapshotted, or null.
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s = snapshotPoint(n);
    return if (s > cached and s >= session_min) s else null;
}

pub const Stash = ft.Stash;

/// The drafter hand-off before the first verify row: one ingest of the last `window` taps rows from `row0` at `start`.
pub const Handoff = struct { start: u64, row0: u32, n: u32 };

pub fn handoff(at: u64, rows: u32, window: u32) ?Handoff {
    if (rows == 0 or window == 0) return null;
    const skip = rows -| window;
    return .{ .start = at + skip, .row0 = skip, .n = rows - skip };
}

/// One contiguous run of positions in the ring: rows [row, row + n) hold positions [at, at + n) of a range.
pub const Span = struct { row: u32, n: u32, at: u32 };

/// Positions [a, b) (at most `ring`) as at most two contiguous ring runs; `at` counts from a.
pub fn spans(a: u64, b: u64, out: *[2]Span) []const Span {
    std.debug.assert(b >= a and b - a <= ring);
    const total: u32 = @intCast(b - a);
    if (total == 0) return out[0..0];
    const r: u32 = @intCast(a % ring);
    const first = @min(total, ring - r);
    out[0] = .{ .row = r, .n = first, .at = 0 };
    if (first == total) return out[0..1];
    out[1] = .{ .row = 0, .n = total - first, .at = first };
    return out[0..2];
}

/// The stash roles (streams [ring, 4 D], normed input, fp32 pre / post [ring, 4], comb [ring, 16]) in glue-step order.
pub const roles = [_][]const u8{ "s.ced.x", "s.ced.out", "s.ced.pre", "s.ced.post", "s.ced.comb" };

test "replay rules as replay.py / sessions.py / batch.py" {
    const cfg: Config = .{};
    try std.testing.expectEqual(@as(u32, 20), try decoderStart(&cfg));
    // n <= 128: R0 = 0, the decoder sees the whole prefilled part
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 0 }, tailOf(0));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 44 }, tailOf(44));
    try std.testing.expectEqual(Tail{ .r0 = 0, .m = 127 }, tailOf(127));
    // n = 129 (end 128): R0 = 1 = n - 128
    try std.testing.expectEqual(Tail{ .r0 = 1, .m = 127 }, tailOf(128));
    try std.testing.expectEqual(Tail{ .r0 = 32768 - 128, .m = 127 }, tailOf(32767));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(33));
    try std.testing.expectEqual(@as(u64, 32), snapshotPoint(48));
    try std.testing.expectEqual(@as(u64, 48), snapshotPoint(49));
    try std.testing.expectEqual(@as(?u64, null), savePoint(60, 0)); // S = 48 < session_min
    try std.testing.expectEqual(@as(?u64, 64), savePoint(65, 0));
    try std.testing.expectEqual(@as(?u64, null), savePoint(65, 64)); // nothing past the resumed position
    try std.testing.expectEqual(@as(?u64, 4096), savePoint(4100, 1024));
    // a decoder layer with Engram, or no ratio-1 layer: refused
    var bad = cfg;
    bad.engram_layers = @TypeOf(cfg.engram_layers).of(&.{ 1, 24 });
    try std.testing.expectError(error.EngramInDecoder, decoderStart(&bad));
}

test "the DSpark hand-off: the decoder pass's rows whole, a longer run's last window" {
    // replay, n = 1,536: the tail [1408, 1535), 127 rows, one ingest at R0
    try std.testing.expectEqual(Handoff{ .start = 1408, .row0 = 0, .n = 127 }, handoff(tailOf(1535).r0, tailOf(1535).m, 128).?);
    // a prompt of 45 tokens: its 44 prefilled rows from 0
    try std.testing.expectEqual(Handoff{ .start = 0, .row0 = 0, .n = 44 }, handoff(tailOf(44).r0, tailOf(44).m, 128).?);
    // full mode, a last segment of 2,048 rows at 4,096: its last 128
    try std.testing.expectEqual(Handoff{ .start = 4096 + 1920, .row0 = 1920, .n = 128 }, handoff(4096, 2048, 128).?);
    try std.testing.expectEqual(@as(?Handoff, null), handoff(0, 0, 128));
}

test "the stash: the last rows of every segment, a gap restarts it, two ring runs at most" {
    var s: Stash = .{};
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    s.extend(0, ids[0..100]);
    try std.testing.expect(s.covers(0, 100));
    s.extend(100, ids[100..300]);
    try std.testing.expectEqual(@as(u64, 300 - keep), s.lo());
    try std.testing.expect(s.covers(173, 300));
    try std.testing.expect(!s.covers(172, 300));
    try std.testing.expect(!s.covers(173, 299));
    var got: [keep]u32 = undefined;
    s.tail(173, 300, &got);
    try std.testing.expectEqualSlices(u32, ids[173..300], &got);
    // a segment that does not start at hi: the stash starts over there (a restored slot whose stash ended elsewhere)
    s.extend(500, ids[0..10]);
    try std.testing.expectEqual(@as(u64, 500), s.lo());
    try std.testing.expect(!s.covers(495, 510));
    var sp: [2]Span = undefined;
    try std.testing.expectEqualSlices(Span, &.{ .{ .row = 44, .n = 84, .at = 0 }, .{ .row = 0, .n = 43, .at = 84 } }, spans(300, 427, &sp));
    try std.testing.expectEqualSlices(Span, &.{.{ .row = 0, .n = 127, .at = 0 }}, spans(256, 383, &sp));
    try std.testing.expectEqual(@as(usize, 0), spans(7, 7, &sp).len);
}
