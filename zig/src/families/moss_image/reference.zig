//! A run's rows against exact ones, gated by the reference implementation's own distance from them.
const std = @import("std");
const host = @import("host.zig");

pub const Match = struct { max_abs: f64, cosine: f64, worst_row: f64, worst_at: usize };

/// Largest absolute difference, cosine, and the largest relative L2 error of one row of `width` (and its row).
pub fn match(comptime T: type, ours: []const T, theirs: anytype, width: usize) Match {
    return measure(T, ours, theirs, width, false);
}

/// `theirs` rounded to bf16 against itself: the closest a bf16 result can be.
pub fn rounding(theirs: []const f32, width: usize) Match {
    return measure(f32, theirs, theirs, width, true);
}

fn measure(comptime T: type, ours: []const T, theirs: anytype, width: usize, comptime rounded: bool) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var m: Match = .{ .max_abs = 0, .cosine = 0, .worst_row = 0, .worst_at = 0 };
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |h, b| {
            const a: f64 = if (rounded) host.round(h) else if (T == u16) host.widen(h) else h;
            const d = a - b;
            dot += a * b;
            aa += a * a;
            bb += @as(f64, b) * b;
            rd += d * d;
            rb += @as(f64, b) * b;
            m.max_abs = @max(m.max_abs, @abs(d));
        }
        const rel = @sqrt(rd / @max(rb, 1e-30));
        if (rel > m.worst_row) {
            m.worst_row = rel;
            m.worst_at = row / width;
        }
    }
    m.cosine = dot / @sqrt(aa * bb);
    return m;
}

/// How far a result may be from exact: this many times the reference's distance (or bf16 rounding's, for bf16).
pub const factor = 2.0;

pub const Verdict = struct { pass: bool, cosine_err: f64, cosine_limit: f64, row: f64, row_at: usize, row_limit: f64 };

/// Ours against float32: its cosine error and worst row each within `factor` of the bf16 reference's or rounding's.
pub fn gate(ours: Match, reference: Match, rounded: Match) Verdict {
    return within(ours, .{ .max_abs = 0, .cosine = @min(reference.cosine, rounded.cosine), .worst_row = @max(reference.worst_row, rounded.worst_row), .worst_at = 0 });
}

/// Ours against exact rows: its cosine error and worst row each within `factor` of the reference's.
pub fn within(ours: Match, reference: Match) Verdict {
    const cosine_limit = factor * (1 - reference.cosine);
    const row_limit = factor * reference.worst_row;
    const cosine_err = 1 - ours.cosine;
    return .{
        .pass = cosine_err <= cosine_limit and ours.worst_row <= row_limit,
        .cosine_err = cosine_err,
        .cosine_limit = cosine_limit,
        .row = ours.worst_row,
        .row_at = ours.worst_at,
        .row_limit = row_limit,
    };
}

/// A check passes only when every gate passes and the repeat gave the same bytes.
pub fn accept(verdicts: []const Verdict, repeat: bool) bool {
    for (verdicts) |v| if (!v.pass) return false;
    return repeat;
}

const cols = 64;
const n = 8;

fn synthetic(out: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*x| x.* = prng.random().floatNorm(f32) * 3;
}

fn bf16Of(out: []u16, from: []const f32) void {
    for (out, from) |*o, x| o.* = host.bf16(x);
}

test "a result as close as the bf16 reference passes" {
    var exact: [cols * n]f32 = undefined;
    synthetic(&exact, 1);
    var ref: [cols * n]f32 = undefined;
    for (&ref, exact, 0..) |*r, x, i| r.* = host.round(x * (1 + 0.004 * @as(f32, if (i % 3 == 0) 1 else -1)));
    var ours: [cols * n]u16 = undefined;
    bf16Of(&ours, &exact);
    const v = gate(match(u16, &ours, &exact, cols), match(f32, &ref, &exact, cols), rounding(&exact, cols));
    try std.testing.expect(v.pass);
    try std.testing.expect(accept(&.{v}, true));
}

test "a planted mismatch fails even when its repeat is identical" {
    var exact: [cols * n]f32 = undefined;
    synthetic(&exact, 2);
    var ref: [cols * n]f32 = undefined;
    for (&ref, exact, 0..) |*r, x, i| r.* = host.round(x * (1 + 0.004 * @as(f32, if (i % 3 == 0) 1 else -1)));
    var ours: [cols * n]u16 = undefined;
    bf16Of(&ours, &exact);
    // one row off by 5%, the rest exact: the cosine stays near 1, the worst row does not
    for (ours[5 * cols ..][0..cols], exact[5 * cols ..][0..cols]) |*o, x| o.* = host.bf16(x * 1.05);
    var again: [cols * n]u16 = ours;
    const repeat = std.mem.eql(u16, &ours, &again);
    const m = match(u16, &ours, &exact, cols);
    const v = gate(m, match(f32, &ref, &exact, cols), rounding(&exact, cols));
    try std.testing.expect(repeat);
    try std.testing.expect(!v.pass);
    try std.testing.expectEqual(@as(usize, 5), m.worst_at);
    try std.testing.expect(!accept(&.{v}, repeat));
}

test "a different repeat fails even when every gate passes" {
    const v: Verdict = .{ .pass = true, .cosine_err = 0, .cosine_limit = 1, .row = 0, .row_at = 0, .row_limit = 1 };
    try std.testing.expect(!accept(&.{ v, v }, false));
}

test "against float64, a row further off than the reference allows fails" {
    var exact: [cols * n]f64 = undefined;
    var f: [cols * n]f32 = undefined;
    synthetic(&f, 3);
    for (&exact, f) |*e, x| e.* = x;
    var theirs: [cols * n]f32 = undefined;
    var ours: [cols * n]f32 = undefined;
    for (&theirs, &ours, exact, 0..) |*t, *o, x, i| {
        const s: f64 = if (i % 2 == 0) 1 else -1;
        t.* = @floatCast(x * (1 + 1e-5 * s));
        o.* = @floatCast(x * (1 - 1e-5 * s));
    }
    try std.testing.expect(within(match(f32, &ours, &exact, cols), match(f32, &theirs, &exact, cols)).pass);
    for (ours[3 * cols ..][0..cols], exact[3 * cols ..][0..cols]) |*o, x| o.* = @floatCast(x * (1 + 1e-4));
    const v = within(match(f32, &ours, &exact, cols), match(f32, &theirs, &exact, cols));
    try std.testing.expect(!v.pass);
    try std.testing.expectEqual(@as(usize, 3), v.row_at);
}
