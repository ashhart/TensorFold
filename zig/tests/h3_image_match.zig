//! The H3-Image selftests' comparison of a GPU run's bf16 values against a float64 host reference.
const std = @import("std");
const bf16_round = @import("bf16_round");

/// Ours against the float64 reference: cosine, relative error, the share bit-equal, the worst value by its row's RMS.
pub const Match = struct { cosine: f64, relative: f64, equal: f64, worst: f64 };

pub fn match(ours: []const u16, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var equal: usize = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rb: f64 = 0;
        var rd: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |o, t| {
            const x: f64 = bf16_round.value(o);
            dot += x * t;
            aa += x * x;
            rb += @as(f64, t) * t;
            rd = @max(rd, @abs(x - t));
            dd += (x - t) * (x - t);
            equal += @intFromBool(o == bf16_round.bits(t));
        }
        bb += rb;
        worst = @max(worst, rd / @sqrt(@max(rb / @as(f64, @floatFromInt(width)), 1e-300)));
    }
    return .{ .cosine = dot / @sqrt(@max(aa * bb, 1e-300)), .relative = @sqrt(dd / @max(bb, 1e-300)), .equal = @as(f64, @floatFromInt(equal)) / @as(f64, @floatFromInt(ours.len)), .worst = worst };
}

pub fn report(label: []const u8, m: Match, pass: bool) void {
    std.debug.print("{s}: cosine {d:.7}, relative error {e:.2}, {d:.4} bit-equal, worst value {e:.2} of its row's RMS ({s})\n", .{ label, m.cosine, m.relative, m.equal, m.worst, if (pass) "pass" else "FAIL" });
}

/// A whole run `m` beside the float32-sum host run's `noise`: within `factor` of its distance, cosine at `floor`.
pub fn within(m: Match, noise: Match, factor: f64, floor: f64) bool {
    return m.cosine >= floor and 1 - m.cosine <= factor * (1 - noise.cosine) and m.relative <= factor * noise.relative;
}

/// `vals` (bf16 values held in fp32) as their bf16 bits.
pub fn bf16s(a: std.mem.Allocator, vals: []const f32) ![]u16 {
    const out = try a.alloc(u16, vals.len);
    for (out, vals) |*o, v| o.* = bf16_round.bits(v);
    return out;
}
