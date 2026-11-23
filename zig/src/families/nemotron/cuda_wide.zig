//! TF_NEMOTRON_WIDE_ROUNDS: shared rounds of up to 16 streams and 32 rows where each row keeps its 16-row bits.
const std = @import("std");
const Config = @import("config.zig").Config;

/// Streams a shared round holds: today's 8, or 16 with the switch.
pub const narrow_streams = 8;
pub const wide_streams = 16;

/// The switch; default off. The startup check refused every malformed value already, so a plain default stands here.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_NEMOTRON_WIDE_ROUNDS", false) catch false;
}

/// Where a 17-32-row round keeps each row's bits: GB10, one rank, non-latent, not Bolt, no e4m3/bf16 experts, no slide.
pub fn allowed(c: Config, gb10: bool, slide: bool) bool {
    return refusal(c, gb10, slide) == null;
}

/// Why the switch leaves rounds at 8 streams here (null: it applies), for the line the engine logs when it is set.
pub fn refusal(c: Config, gb10: bool, slide: bool) ?[]const u8 {
    if (!gb10) return "only GB10's affine-4 has the 32-row verify";
    if (c.latent > 0) return "a latent checkpoint keeps its own 32-row window";
    if (c.bolt or c.ck or c.salm) return "Bolt, e4m3 checkpoint rows and bf16 experts are not proved at 17-32 rows";
    if (c.world != 1) return "it runs on one rank only";
    if (slide) return "Sliding Weights route lessons to 8 streams a pass";
    return null;
}

/// Lessons attached after init (saved ones, which load without --slide): rounds back to 8 streams, and why; else null.
pub fn withLessons(c: *Config) ?[]const u8 {
    if (!c.wide_rounds) return null;
    c.wide_rounds = false; // every buffer was sized for 16 streams: 8 use less of each
    return "saved lessons route to 8 streams a pass";
}

/// Whether a shared window of `rows` rows fits (its streams' Mamba rows stay at windowRows each, the scan's stride).
pub fn windowFits(c: Config, rows: usize) bool {
    return rows <= @import("cuda_state.zig").sharedRows(c);
}

/// The shared check's rows for stream `k`: uneven splits, within the round's room and the stream's own window.
pub fn checkRows(round: usize, k: usize, room: usize, waiting: usize, window: usize, left: usize) usize {
    const cap = @min(room - waiting, window);
    return @min(1 + (round * 7 + k * 5 + round / 3) % cap, left);
}

/// Whether Head.chains batches one more stream of `more` rows: the switch's streams (main's 8 off) and shared rows.
pub fn batches(c: Config, batched: usize, rows: usize, more: usize) bool {
    return batched < streams(c) and rows + more <= @import("cuda_state.zig").sharedRows(c);
}

/// A shared round's streams at most.
pub fn streams(c: Config) usize {
    return if (c.wide_rounds) wide_streams else narrow_streams;
}

/// The shared expert's 16-row items at `rows` (1-32), halves at members 0 and `half`; then (items, experts).
pub fn sharedItems(experts: u32, half: u32, rows: usize) [14]u32 {
    var w: [14]u32 = @splat(0);
    const lo: u32 = @intCast(@min(rows, 16));
    const hi: u32 = @intCast(rows - lo);
    if (hi == 0) {
        w[0..6].* = .{ experts, 0, lo, experts + 1, half, lo };
        w[12..14].* = .{ 2, 2 };
    } else {
        w[0..12].* = .{ experts, 0, 16, experts, 16, hi, experts + 1, half, 16, experts + 1, half + 16, hi };
        w[12..14].* = .{ 4, 2 };
    }
    return w;
}

test "wide rounds: shared plan items keep 16-row tiles, today's two items at 16 rows or fewer" {
    for (1..17) |rows| {
        const w = sharedItems(40, 32, rows);
        const n: u32 = @intCast(rows);
        try std.testing.expectEqualSlices(u32, &.{ 40, 0, n, 41, 32, n }, w[0..6]);
        try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, w[12..14]);
    }
    for (17..33) |rows| {
        const w = sharedItems(40, 32, rows);
        const hi: u32 = @intCast(rows - 16);
        try std.testing.expectEqualSlices(u32, &.{ 40, 0, 16, 40, 16, hi, 41, 32, 16, 41, 48, hi, 4, 2 }, &w);
    }
}

const small: Config = .{ .hidden = 8, .vocab = 16, .layers = 2, .mamba_heads = 2, .mamba_head_dim = 4, .groups = 1, .state = 4, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 4, .experts = 4, .top_k = 2, .expert_width = 4, .shared_width = 8, .routed_scaling = 1, .eps = 1e-5 };

test "wide rounds: 16 streams only with the switch, and only where each row's kernels are proved" {
    var c: Config = small;
    try std.testing.expectEqual(@as(usize, 8), streams(c));
    c.wide_rounds = true;
    try std.testing.expectEqual(@as(usize, 16), streams(c));
    c = small;
    try std.testing.expect(allowed(c, true, false));
    try std.testing.expect(!allowed(c, false, false)); // off GB10 affine-4 has no 32-row verify
    inline for (.{ "bolt", "ck", "salm" }) |f| {
        var x: Config = small;
        @field(x, f) = true;
        try std.testing.expect(!allowed(x, true, false));
    }
    var lat: Config = small;
    lat.latent = 1024;
    try std.testing.expect(!allowed(lat, true, false)); // Super keeps its own 32-row window
    var tp: Config = small;
    tp.world = 2;
    try std.testing.expect(!allowed(tp, true, false));
    try std.testing.expect(!allowed(small, true, true)); // lessons reach 8 streams' rows a pass (REV60 2)
}

test "wide rounds: the shared-widths check's rounds reach 17 to 32 rows with the switch, a stream's within 16 (REV60 5)" {
    for ([_]bool{ false, true }) |switched| {
        var c: Config = small;
        c.wide_rounds = switched;
        const room = @import("cuda_state.zig").sharedRows(c);
        const window = @import("cuda_state.zig").windowRows(c);
        for (2..streams(c) + 1) |n| {
            var widest: usize = 0;
            for (0..64) |round| {
                var left = room;
                var total: usize = 0;
                for (0..n) |k| {
                    const rows = checkRows(round, k, left, n - 1 - k, window, 1000);
                    try std.testing.expect(rows >= 1 and rows <= window);
                    if (!switched) try std.testing.expectEqual(1 + (round * 7 + k * 5 + round / 3) % (left - (n - 1 - k)), rows); // as before
                    left -= rows;
                    total += rows;
                }
                try std.testing.expect(total <= room);
                widest = @max(widest, total);
            }
            if (switched) try std.testing.expect(widest > window); // the check now sends the rounds the switch allows
        }
    }
}

test "wide rounds: the switch set where it cannot apply says why, so it is never silently ignored (REV60 7)" {
    try std.testing.expect(refusal(small, true, false) == null);
    try std.testing.expect(refusal(small, false, false) != null);
    try std.testing.expect(refusal(small, true, true) != null);
    inline for (.{ "bolt", "ck", "salm" }) |f| {
        var x: Config = small;
        @field(x, f) = true;
        try std.testing.expect(refusal(x, true, false) != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_engine.zig"), "if (wide.on()) if (wide.refusal(") != null);
}

test "wide rounds: saved lessons loaded after init put rounds back at 8 streams and say why (REV74 1)" {
    var c: Config = small;
    try std.testing.expect(withLessons(&c) == null); // the switch off: nothing to undo
    c.wide_rounds = true;
    try std.testing.expect(withLessons(&c) != null);
    try std.testing.expect(!c.wide_rounds);
    try std.testing.expectEqual(@as(usize, 8), streams(c));
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_slide.zig"), "if (@import(\"cuda_wide.zig\").withLessons(&e.c)) |why|") != null);
}

test "wide rounds: a draft call batches the switch's streams, main's 8 with it off (REV83 1)" {
    var c: Config = small;
    try std.testing.expect(batches(c, 7, 0, 1));
    try std.testing.expect(!batches(c, 8, 0, 1)); // the ninth drafts alone, as on main
    try std.testing.expect(!batches(c, 0, 16, 1)); // and past the shared rows
    c.wide_rounds = true;
    try std.testing.expect(batches(c, 15, 0, 1) and !batches(c, 16, 0, 1));
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_lanes.zig"), "!@import(\"cuda_wide.zig\").batches(self.e.c, batched, rows, follow.len)") != null);
}
