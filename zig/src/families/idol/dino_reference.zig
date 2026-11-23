//! The DINOv3 encoder's gate: ours against the float32 ruler within twice torch bf16's distance, cosine and worst row.
const std = @import("std");
const ref = @import("reference.zig");
const cfg = @import("dino_config.zig");

pub const Match = ref.Match;
pub const match = ref.match;

/// Ours and torch's against the ruler, and whether ours passes.
pub const Gate = struct { ours: Match, torch: Match, pass: bool };

/// Ours and torch's rows [n, 1280] each against the ruler's: worst row and 1 - cosine within 2x torch's.
pub fn gate(ours: []const f32, torch: []const f32, ruler: []const f32) Gate {
    const o = ref.match(ours, ruler, cfg.width);
    const t = ref.match(torch, ruler, cfg.width);
    return .{ .ours = o, .torch = t, .pass = ref.within(o, t) };
}

/// The row holding the largest relative L2 error against `theirs`, and that error.
pub fn worstRow(ours: []const f32, theirs: []const f32) struct { row: usize, err: f64 } {
    var worst: f64 = 0;
    var at: usize = 0;
    var row: usize = 0;
    while (row * cfg.width < ours.len) : (row += 1) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row * cfg.width ..][0..cfg.width], theirs[row * cfg.width ..][0..cfg.width]) |a, b| {
            rd += (@as(f64, a) - b) * (@as(f64, a) - b);
            rb += @as(f64, b) * b;
        }
        const e = @sqrt(rd / @max(rb, 1e-30));
        if (e > worst) {
            worst = e;
            at = row;
        }
    }
    return .{ .row = at, .err = worst };
}

test "a planted 2% error in one of 64 feature rows fails the gate, though its cosine alone would pass" {
    const a = std.testing.allocator;
    const n = 64 * cfg.width;
    const ruler = try a.alloc(f32, n);
    defer a.free(ruler);
    const torch = try a.alloc(f32, n);
    defer a.free(torch);
    const ours = try a.alloc(f32, n);
    defer a.free(ours);
    var rng = std.Random.DefaultPrng.init(11);
    for (ruler, torch, ours) |*r, *t, *o| {
        r.* = rng.random().floatNorm(f32);
        t.* = r.* * (1 + 4e-3 * rng.random().floatNorm(f32));
        o.* = r.* * (1 + 3e-3 * rng.random().floatNorm(f32));
    }
    try std.testing.expect(gate(ours, torch, ruler).pass);
    for (ours[37 * cfg.width ..][0..cfg.width], ruler[37 * cfg.width ..][0..cfg.width]) |*o, r| o.* += 0.02 * r;
    const g = gate(ours, torch, ruler);
    try std.testing.expect(!g.pass);
    try std.testing.expect(ref.withinCosine(g.ours, g.torch));
    try std.testing.expectEqual(@as(usize, 37), worstRow(ours, ruler).row);
}

test "a NaN anywhere in ours fails the gate" {
    var ruler: [2 * cfg.width]f32 = @splat(1);
    var ours = ruler;
    ours[cfg.width + 3] = std.math.nan(f32);
    try std.testing.expect(!gate(&ours, &ruler, &ruler).pass);
}
